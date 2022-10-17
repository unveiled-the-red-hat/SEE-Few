from .entities import LabelSpan, Token, Document
from typing import List, OrderedDict, Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils import common
from utils.data_loader import Dataset


class Evaluator:
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
        self._gt_entities = OrderedDict()  # ground truth
        self._pred_entities = OrderedDict()  # prediction

        self._convert_gt(self._dataset.documents)

    def update_batch(self, batch_doc_id_set, batch_result: List[LabelSpan]):
        for doc_id in batch_doc_id_set:
            self._pred_entities[doc_id] = []

        for item in batch_result:
            self._pred_entities[item.doc_id].append(item)

    def compute_scores(self):
        for doc_id, doc_entities in self._pred_entities.items():
            preds = []
            unique = set()

            for i, span in enumerate(doc_entities):
                span: LabelSpan
                if (span.start, span.end) in unique:
                    continue
                max_cls_score = span.cls_score
                for j in range(i + 1, len(doc_entities)):
                    other: LabelSpan = doc_entities[j]
                    if (
                        span.start == other.start
                        and span.end == other.end
                        and span.entity_type == other.entity_type
                    ):
                        max_cls_score = max(max_cls_score, other.cls_score)

                preds.append(
                    LabelSpan(
                        span.doc_id,
                        span.start,
                        span.end,
                        span.entity_type,
                        max_cls_score,
                    )
                )
                unique.add((span.start, span.end))

            preds = sorted(preds, key=lambda x: x.cls_score, reverse=True)
            preds = self._remove_partial_overlapping(preds)
            preds = self._remove_overlapping(preds)
            self._pred_entities[doc_id] = preds

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(
            self._gt_entities.values(), self._pred_entities.values()
        ):
            converted_gt.append([(s.start, s.end, s.entity_type) for s in sample_gt])
            converted_pred.append(
                [(s.start, s.end, s.entity_type) for s in sample_pred]
            )

        ner_eval = self._score(converted_gt, converted_pred, print_results=True)

        return ner_eval

    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            self._gt_entities[doc.doc_id] = []
            for entity in doc.entities:
                e = entity.as_label_span()
                self._gt_entities[doc.doc_id].append(e)

    def _remove_overlapping(self, entities):
        non_overlapping_entities = []
        for i, entity in enumerate(entities):
            if not self._is_overlapping(entity, non_overlapping_entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _remove_partial_overlapping(self, entities):
        non_overlapping_entities = []
        for entity in entities:
            if not self._is_partial_overlapping(entity, entities):
                non_overlapping_entities.append(entity)

        return non_overlapping_entities

    def _is_partial_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_partial_overlap(e1, e2):
                return True

        return False

    def _is_overlapping(self, e1, entities):
        for e2 in entities:
            if self._check_overlap(e1, e2):
                return True

        return False

    def _check_overlap(self, e1: LabelSpan, e2: LabelSpan):
        if e1 == e2 or e1.end <= e2.start or e2.end <= e1.start:
            return False
        else:
            return True

    def _check_partial_overlap(self, e1: LabelSpan, e2: LabelSpan):
        if (e1.start < e2.start and e2.start < e1.end and e1.end < e2.end) or (
            e2.start < e1.start and e1.start < e2.end and e2.end < e1.end
        ):
            return True
        else:
            return False

    def _score(
        self,
        gt: List[List[Tuple]],
        pred: List[List[Tuple]],
        print_results: bool = False,
    ):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t.index)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t.index)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None)
        micro = prfs(gt_all, pred_all, labels=labels, average="micro")[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average="macro")[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(
                per_type,
                list(micro) + [total_support],
                list(macro) + [total_support],
                types,
            )

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ("type", "precision", "recall", "f1-score", "support")
        max_len = max(max([len(x.short_name) for x in types]), 5)
        row_fmt = f"%{max_len}s" + (" %12s" * (len(columns) - 1))
        results = ["***** Eval results *****", row_fmt % columns]

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            # results.append("\n")

        results.append("")

        # micro
        results.append(row_fmt % self._get_row(micro, "micro"))
        # results.append("\n")

        # macro
        results.append(row_fmt % self._get_row(macro, "macro"))

        # results.append("\n")

        for line in results:
            common.logger.info(line)

        # with open(
        #     os.path.join(self._valid_score_path, f"{self._epoch}.txt"), "a+"
        # ) as file:
        #     file.write(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _prettify(self, text: str):
        text = (
            text.replace("_start_", "")
            .replace("_classify_", "")
            .replace("<unk>", "")
            .replace("⁇", "")
        )
        text = text.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
        return text

    def _token_to_text(self, tokens: List[Token], start: int, end: int) -> str:
        phrases = []
        for i in range(start, end):
            phrases.append(tokens[i].phrase)
        text = " ".join(phrases)
        text = self._prettify(text)
        return text

    def get_examples(self):
        outputs = []

        for i, doc in enumerate(self._dataset.documents):
            doc: Document
            text = self._token_to_text(doc.tokens, 0, len(doc.tokens))
            infer_list = [
                {
                    "text": self._token_to_text(doc.tokens, infer.start, infer.end),
                    "label": infer.entity_type.short_name,
                }
                for infer in self._pred_entities[i]
            ]

            outputs.append({"text": text, "infer_entities": infer_list})

            converted_gt = [
                (s.start, s.end, s.entity_type) for s in self._gt_entities[i]
            ]
            converted_pred = [
                (s.start, s.end, s.entity_type) for s in self._pred_entities[i]
            ]

            join_set = set(converted_gt) & set(converted_pred)
            lack = set(converted_gt) - join_set
            new = set(converted_pred) - join_set

            outputs[-1]["golden_entities"] = [
                {
                    "text": self._token_to_text(doc.tokens, item.start, item.end),
                    "label": item.entity_type.short_name,
                }
                for item in self._gt_entities[i]
            ]
            outputs[-1]["lack"] = [
                {
                    "text": self._token_to_text(doc.tokens, item[0], item[1]),
                    "label": item[2].short_name,
                }
                for item in lack
            ]
            outputs[-1]["new"] = [
                {
                    "text": self._token_to_text(doc.tokens, item[0], item[1]),
                    "label": item[2].short_name,
                }
                for item in new
            ]

        return outputs
