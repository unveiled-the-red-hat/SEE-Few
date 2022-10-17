from ast import arg
import json
from collections import OrderedDict
import math
from models.model_ner import SEE_Few
from utils import common, sampling
from utils.args import get_arg_parser
import logging
import sys
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.data_loader import NerDataProcessor
from utils.entities import Document, LabelSpan
from utils.data_loader import Dataset
from torch.nn import DataParallel
from transformers import AdamW, AutoConfig, PreTrainedModel, PreTrainedTokenizer
from utils.evaluator import Evaluator
import time
from utils.focalloss import FocalLoss


def get_optimizer_params(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_params


def extract_span(
    args,
    batch: dict,
    seeding_scores,
    offsets,
    data_processor: NerDataProcessor,
    is_eval: bool,
):
    batch_size = batch["input_ids"].shape[0]
    round_offsets = torch.round(offsets).long()
    span_dataset = []
    if is_eval:
        span_dataset = set()
    for i in range(batch_size):
        old_boundaries = batch["entity_spans_token"][i]
        new_boundaries = old_boundaries + round_offsets[i]
        if not is_eval:
            gt_iofs = batch["iofs"][i]

        token_count = batch["token_masks_bool"][i].long().sum()

        new_boundaries[:, 0][new_boundaries[:, 0] < 0] = 0
        new_boundaries[:, 1][new_boundaries[:, 1] > token_count] = token_count

        gt_span_types = dict()
        span_set = dict()

        if not is_eval:
            gt_boundaries = (
                old_boundaries
                + torch.stack((batch["l_offsets"][i], batch["r_offsets"][i])).T.long()
            )
            for idx, span in enumerate(gt_boundaries):
                start = span[0].item()
                end = span[1].item()
                e_type = batch["entity_types"][i][idx].item()
                if start >= end:
                    continue
                gt_span_types[(start, end)] = e_type
                if e_type != 0:
                    span_set[(start, end)] = e_type

        for idx, span in enumerate(new_boundaries):
            start = span[0].item()
            end = span[1].item()
            if is_eval:
                iof = seeding_scores[i][idx].item()
            else:
                iof = gt_iofs[idx]
            old_start = old_boundaries[idx][0].item()
            old_end = old_boundaries[idx][1].item()
            if old_start >= old_end:
                continue
            if (start, end) in gt_span_types.keys():
                gt_type = gt_span_types[(start, end)]
            else:
                gt_type = 0

            if (iof >= args.seed_threshold) and start < end:
                span_set[(start, end)] = gt_type

            if is_eval:
                doc_id = batch["_doc"][i].doc_id
                old_span = " ".join(
                    [
                        batch["_doc"][i]._tokens[j].phrase
                        for j in range(old_start, old_end)
                    ]
                )
                new_span = " ".join(
                    [batch["_doc"][i]._tokens[j].phrase for j in range(start, end)]
                )
                # self._log_csv("test", "span_extraction", doc_id, iof, old_span, new_span, old_start, old_end, start, end)

        if not is_eval:
            for k in gt_span_types.keys():
                if gt_span_types[k] != 0 and k not in span_set:
                    span_set[k] = gt_span_types[k]

        for k in span_set.keys():
            doc: Document = batch["_doc"][i]
            if is_eval:
                span_dataset.add((doc.doc_id, k[0], k[1]))
            else:
                for span_type in range(data_processor.entity_type_count):
                    span_data = {
                        "doc_id": doc.doc_id,
                        "span_type": span_type,
                        "input_ids": doc.encodings,
                        "tokens": doc.tokens,
                        "batch_idx": i,
                        "label": 0 if span_type == span_set[k] else 1,
                        "start_idx": k[0],
                        "end_idx": k[1],
                        "template_ids": data_processor.template_encoding[span_type],
                    }
                    span_dataset.append(span_data)

    return span_dataset


def train(args, model: torch.nn.Module):
    train_label = "train"
    valid_label = "valid"
    dataset_paths = {train_label: args.train_path, valid_label: args.valid_path}

    data_processor = NerDataProcessor(args)
    data_processor.read(dataset_paths=dataset_paths)

    train_dataset: Dataset = data_processor.get_dataset(train_label)
    train_sample_count = train_dataset.document_count

    updates_epoch = train_sample_count // args.se_train_batch_size
    updates_total = updates_epoch * args.epochs

    optimizer_params = get_optimizer_params(args, model)
    optimizer = AdamW(
        optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False
    )

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup * updates_total,
        num_training_steps=updates_total,
    )

    best_f1 = 0
    best_epoch = 0

    entailing_loss_func = FocalLoss(
        class_num=2, reduction="none", gamma=args.focal_loss_gamma
    )
    smooth_l1_loss_func = torch.nn.SmoothL1Loss(reduction="none")

    for epoch in range(args.epochs):
        common.logger.info(f"Train epoch {epoch + 1}")
        train_dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(
            train_dataset,
            batch_size=args.se_train_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        model.zero_grad()

        iteration = 0
        total = train_dataset.document_count // args.se_train_batch_size
        epoch_loss = 0.0

        train_bar = tqdm(data_loader, total=total, desc=f"Train epoch {epoch + 1}")
        for batch in train_bar:
            model.train()
            batch = common.to_device(batch, args.device)
            # forward step
            p_seed, p_offset = model(batch, cls_stage=False)

            span_dataset = extract_span(
                args, batch, p_seed, p_offset, data_processor, is_eval=False
            )

            span_dataloader = DataLoader(
                dataset=span_dataset,
                batch_size=data_processor.entity_type_count
                * args.entail_train_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=sampling.collate_batch_span_classification_data,
            )

            entailing_loss_sum = 0

            # entailing loss
            for span_batch in span_dataloader:
                entailing_result = model(span_batch, cls_stage=True)
                entailing_label = torch.tensor(
                    span_batch["label"], dtype=torch.long
                ).to(device=model.device)
                entailing_loss = (
                    entailing_loss_func(entailing_result, entailing_label).sum()
                    * args.entailing_loss_weight
                )

                entailing_loss.backward()
                entailing_loss_sum += entailing_loss.item()

            # seeding loss
            y_iofs = batch["iofs"]
            iof_sample_masks = batch["iof_sample_masks"]
            seeding_loss = (
                smooth_l1_loss_func(p_seed, y_iofs) * iof_sample_masks
            ).sum()

            # expansion loss
            y_l_offsets = batch["l_offsets"]
            y_r_offsets = batch["r_offsets"]
            offset_sample_masks = batch["offset_sample_masks"]

            l_offset_loss = smooth_l1_loss_func(
                p_offset[:, :, 0].squeeze(-1), y_l_offsets
            )
            r_offset_loss = smooth_l1_loss_func(
                p_offset[:, :, 1].squeeze(-1), y_r_offsets
            )

            l_offset_loss[torch.isinf(l_offset_loss)] = 0
            r_offset_loss[torch.isinf(r_offset_loss)] = 0

            l_offset_loss = (l_offset_loss * offset_sample_masks).sum()
            r_offset_loss = (r_offset_loss * offset_sample_masks).sum()

            expansion_loss = l_offset_loss + r_offset_loss
            se_loss = (
                seeding_loss * args.seeding_loss_weight
                + expansion_loss * args.expansion_loss_weight
            )

            se_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            # logging
            iteration += 1
            epoch_loss += se_loss.item() + entailing_loss_sum

            # print(entailing_loss_sum + se_loss.item())

            train_bar.set_description(
                "Train epoch {}, loss:{:.6}".format(epoch + 1, epoch_loss / iteration)
            )

        common.logger.info(
            "Train epoch {}, loss:{:.6}".format(epoch + 1, epoch_loss / iteration)
        )

        f1, examples = eval(args, model, data_processor, valid_label)
        if best_f1 <= f1[2]:
            common.logger.info(f"Best F1 score update, from {best_f1} to {f1[2]}")
            best_f1 = f1[2]
            best_epoch = epoch + 1
            _save_model(args.save_path, "best_model", model, model.tokenizer)

            result_path = os.path.join(args.save_path, "valid_result.txt")
            with open(result_path, "w", encoding="utf-8") as fw:
                for line in examples:
                    fw.write(json.dumps(line, indent=4, ensure_ascii=False) + "\n")

        common.logger.info(
            f"Best F1 score: {best_f1}, achieved at Epoch: {best_epoch}\n"
        )

    if args.do_eval:
        data_processor.read(dataset_paths={"test": args.test_path})

        best_model_path = os.path.join(args.save_path, "best_model")
        config = AutoConfig.from_pretrained(best_model_path)
        model = SEE_Few.from_pretrained(
            pretrained_model_name_or_path=best_model_path, config=config, args=args
        )
        model.to(args.device)

        eval(args, model, data_processor, "test")

    common.logger.info("Logged in: %s" % args.log_path)
    common.logger.info("Saved in: %s" % args.save_path)


def eval(
    args, model: torch.nn.Module, data_processor: NerDataProcessor, dataset_label: str
):

    common.logger.info(f"***** Evaluating: {dataset_label} *****")

    dataset = data_processor.get_dataset(dataset_label)
    # create evaluator
    evaluator = Evaluator(dataset)

    # create data loader
    dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(
        dataset,
        batch_size=args.se_eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.sampling_processes,
        collate_fn=sampling.collate_fn_padding,
    )

    with torch.no_grad():
        model.eval()

        total = math.ceil(dataset.document_count / args.se_eval_batch_size)
        span_set = set()
        docs = OrderedDict()
        batch_doc_id_set = set()
        for batch in tqdm(data_loader, total=total, desc="Seeding & Expanding"):
            batch = common.to_device(batch, args.device)
            p_seed, p_offset = model(batch, cls_stage=False)

            batch_spans = extract_span(
                args, batch, p_seed, p_offset, data_processor, is_eval=True
            )

            span_set.update(batch_spans)
            for d in batch["_doc"]:
                docs[d.doc_id] = d
                batch_doc_id_set.add(d.doc_id)

        span_dataset = []
        for span in span_set:
            doc_id = span[0]
            doc: Document = docs[doc_id]
            for span_type_idx in range(data_processor.entity_type_count):
                span_data = {
                    "doc_id": doc_id,
                    "span_type": 0,
                    "input_ids": doc.encodings,
                    "tokens": doc.tokens,
                    "batch_idx": 0,
                    "label": 0,
                    "start_idx": span[1],
                    "end_idx": span[2],
                    "template_ids": data_processor.template_encoding[span_type_idx],
                }
                span_dataset.append(span_data)

        entail_eval_batch_size = (
            data_processor.entity_type_count * args.entail_eval_batch_size
        )

        span_dataloader = DataLoader(
            dataset=span_dataset,
            batch_size=entail_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=sampling.collate_batch_span_classification_data,
        )

        type_count = data_processor.entity_type_count

        batch_result = []
        common.logger.info("num of candidate spans: %s" % len(span_set))
        total = math.ceil(len(span_dataset) / entail_eval_batch_size)

        for span_batch in tqdm(span_dataloader, total=total, desc="Entailing"):
            p_entail = model(span_batch, cls_stage=True)
            entail_scores = torch.softmax(p_entail, dim=-1)[:, 0]

            for idx in range(entail_scores.shape[0]):
                e_score = entail_scores[idx].item()
                doc_id = span_batch["doc_id"][idx]
                span = span_batch["span"][idx]
                infer_type = idx % type_count
                candidate = " ".join(
                    [
                        span_batch["tokens"][idx]._tokens[j].phrase
                        for j in range(span[0], span[1])
                    ]
                )
                # self._log_csv("test", "entailment", doc_id, infer_type, e_score, span[0], span[1], candidate)

            entail_scores = entail_scores.view(-1, type_count)  # [bs, type_count]
            entail_types = entail_scores.argmax(dim=-1)

            assert len(span_batch["label"]) % type_count == 0
            for idx in range(len(span_batch["label"]) // type_count):
                doc_id = span_batch["doc_id"][idx * type_count]
                infer_type = entail_types[idx].item()
                if infer_type == 0:  # non-entity
                    continue
                cls_score = entail_scores[idx][infer_type].item()
                span = span_batch["span"][idx * type_count]
                batch_result.append(
                    LabelSpan(
                        doc_id,
                        span[0],
                        span[1],
                        data_processor.get_entity_type(infer_type),
                        cls_score,
                    )
                )

        evaluator.update_batch(batch_doc_id_set, batch_result)

    ner_eval = evaluator.compute_scores()
    examples = evaluator.get_examples()

    if dataset_label == "test":
        path = os.path.join(args.save_path, "test_result.txt")
        with open(path, "w", encoding="utf-8") as fw:
            for line in examples:
                fw.write(json.dumps(line, indent=4, ensure_ascii=False) + "\n")

    return ner_eval, examples


def _save_model(
    save_path: str,
    name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    dir_path = os.path.join(save_path, name)
    common.create_directories_dir(dir_path)

    # save model
    if isinstance(model, DataParallel):
        model.module.save_pretrained(dir_path)
    else:
        model.save_pretrained(dir_path)

    # save vocabulary
    tokenizer.save_pretrained(dir_path)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    arg_parser = get_arg_parser()
    args, _ = arg_parser.parse_known_args()

    if (
        os.path.exists(args.save_path)
        and os.listdir(args.save_path)
        and args.do_train
        and not args.overwrite_save_path
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_save_path to overcome.".format(
                args.save_path
            )
        )

    common.create_directories_dir(args.save_path)
    args.log_path = os.path.join(args.save_path, "logs")
    common.create_directories_dir(args.log_path)

    common.logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s  %(message)s")
    common.reset_logger(common.logger)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    log_file = os.path.join(
        args.log_path, "{}-{}.log".format("train" if args.do_train else "eval", time_)
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    common.logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    common.logger.addHandler(console_handler)

    common.print_arguments(args)

    if args.cpu or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:0")

    if args.seed is not None:
        common.set_seed(args.seed)

    if args.do_train:
        config = AutoConfig.from_pretrained(args.plm_path)
        model = SEE_Few.from_pretrained(
            pretrained_model_name_or_path=args.plm_path, config=config, args=args
        )
        model.to(args.device)
        train(args, model)
    elif args.do_eval:
        data_processor = NerDataProcessor(args)
        data_processor.read(dataset_paths={"test": args.test_path})
        model_path = os.path.join(args.save_path, "best_model")
        if not os.path.exists(model_path):
            common.logger.info(
                "Can not find the best model saved in args.save_path! Load model from args.plm_path. "
            )
            model_path = args.plm_path
        config = AutoConfig.from_pretrained(model_path)
        model = SEE_Few.from_pretrained(
            pretrained_model_name_or_path=model_path, config=config, args=args
        )
        model.to(args.device)

        eval(args, model, data_processor, "test")
    else:
        raise Exception("Please run with 'do_train' or 'do_eval'")
