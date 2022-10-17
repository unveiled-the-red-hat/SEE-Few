import torch
from utils.entities import Document
from utils import common


def create_train_sample(
    doc: Document, offset_limit: int = 4, seed_threshold: float = 0.7
):

    input_ids = doc.encodings
    token_count = len(doc.tokens)
    context_size = len(input_ids)

    # all tokens
    token_spans, word_mask, token_sizes = [], [], []
    for t in doc.tokens:
        token_spans.append(t.span)
        word_mask.append(create_mask(*t.span, context_size))
        token_sizes.append(t.span_end - t.span_start)

    token_sample_mask = torch.ones([len(token_spans)], dtype=torch.bool)

    gt_entities_spans_token = []
    gt_entities_spans = []
    gt_entity_types = []
    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entities_spans.append(e.span)
        gt_entity_types.append(e.entity_type.index)

    (
        pos_iofs,
        pos_l_offsets,
        pos_r_offsets,
        pos_offset_sample_masks,
        pos_iof_sample_masks,
    ) = ([], [], [], [], [])

    (
        neg_iofs,
        neg_l_offsets,
        neg_r_offsets,
        neg_offset_sample_masks,
        neg_iof_sample_masks,
    ) = ([], [], [], [], [])

    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = (
        [],
        [],
        [],
        [],
    )
    pos_entity_spans_token, pos_span_mask = [], []

    neg_entity_spans, neg_entity_types, neg_entity_masks, neg_entity_sizes = (
        [],
        [],
        [],
        [],
    )
    neg_entity_spans_token, neg_span_mask = [], []

    for seed_size in [1, 2]:
        for i in range(0, token_count):
            w_left = i
            w_right = min(token_count, i + seed_size)
            span = doc.tokens[w_left:w_right].span
            span_token = doc.tokens[w_left:w_right].span_token
            if (
                span_token not in pos_entity_spans_token
                and span_token not in neg_entity_spans_token
            ):
                flag_max_iof = 0
                ty, left, right = 0, 0, 0

                for i, gt_entities_span_token in enumerate(gt_entities_spans_token):
                    iof = common.iof(span_token, gt_entities_span_token)
                    if iof > flag_max_iof:
                        flag_max_iof = iof
                        ty = gt_entity_types[i]
                        left = gt_entities_span_token[0] - span_token[0]
                        right = gt_entities_span_token[1] - span_token[1]

                if flag_max_iof > seed_threshold:
                    pos_iofs.append(flag_max_iof)
                    pos_entity_types.append(ty)
                    pos_entity_spans.append(span)
                    pos_entity_spans_token.append(span_token)
                    pos_entity_sizes.append(w_right - w_left)
                    pos_l_offsets.append(left)
                    pos_r_offsets.append(right)
                    pos_offset_sample_masks.append(1)
                    pos_iof_sample_masks.append(1)
                else:
                    neg_iofs.append(flag_max_iof)
                    neg_entity_types.append(0)  # 0: non-entity
                    neg_entity_spans.append(span)
                    neg_entity_spans_token.append(span_token)
                    neg_entity_sizes.append(w_right - w_left)
                    neg_l_offsets.append(0)
                    neg_r_offsets.append(0)
                    neg_offset_sample_masks.append(0)
                    neg_iof_sample_masks.append(1)

    for i, gt_entities_span_token in enumerate(gt_entities_spans_token):
        if gt_entities_span_token not in pos_entity_spans_token:
            pos_entity_spans_token.append(gt_entities_span_token)
            pos_entity_sizes.append(
                gt_entities_span_token[1] - gt_entities_span_token[0]
            )
            pos_l_offsets.append(0)
            pos_r_offsets.append(0)
            pos_entity_spans.append(gt_entities_spans[i])
            pos_iofs.append(1)
            pos_offset_sample_masks.append(
                0
            )  # only for training the entailing module, so mask the offset and iof
            pos_iof_sample_masks.append(0)
            pos_entity_types.append(gt_entity_types[i])

    pos_span_mask = [create_mask(*span, token_count) for span in pos_entity_spans_token]
    pos_window_mask = [
        create_mask(
            max(0, span[0] - offset_limit * 2),
            min(token_count, span[1] + offset_limit * 2),
            token_count,
        )
        for span in pos_entity_spans_token
    ]
    pos_entity_masks = [create_mask(*span, context_size) for span in pos_entity_spans]
    neg_span_mask = [create_mask(*span, token_count) for span in neg_entity_spans_token]
    neg_window_mask = [
        create_mask(
            max(0, span[0] - offset_limit * 2),
            min(token_count, span[1] + offset_limit * 2),
            token_count,
        )
        for span in neg_entity_spans_token
    ]
    neg_entity_masks = [create_mask(*span, context_size) for span in neg_entity_spans]

    iofs = pos_iofs + neg_iofs
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + neg_entity_sizes
    l_offsets = pos_l_offsets + neg_l_offsets
    r_offsets = pos_r_offsets + neg_r_offsets
    offset_sample_masks = pos_offset_sample_masks + neg_offset_sample_masks
    iof_sample_masks = pos_iof_sample_masks + neg_iof_sample_masks
    span_mask = pos_span_mask + neg_span_mask
    window_mask = pos_window_mask + neg_window_mask
    entity_spans_token = pos_entity_spans_token + neg_entity_spans_token

    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(span_mask)

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    # masking of tokens
    attention_mask = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    word_mask = torch.stack(word_mask)

    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)

        span_mask = torch.stack(span_mask)
        window_mask = torch.stack(window_mask)
        entity_spans_token = torch.tensor(entity_spans_token)

        l_offsets = torch.tensor(l_offsets, dtype=torch.float)
        r_offsets = torch.tensor(r_offsets, dtype=torch.float)
        offset_sample_masks = torch.tensor(offset_sample_masks, dtype=torch.bool)
        iof_sample_masks = torch.tensor(iof_sample_masks, dtype=torch.bool)
        iofs = torch.tensor(iofs, dtype=torch.float)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

        span_mask = torch.zeros([1, token_count], dtype=torch.bool)
        window_mask = torch.zeros([1, token_count], dtype=torch.bool)
        entity_spans_token = torch.zeros([1, 2], dtype=torch.long)

        l_offsets = torch.tensor([0], dtype=torch.float)
        r_offsets = torch.tensor([0], dtype=torch.float)
        offset_sample_masks = torch.tensor([0], dtype=torch.bool)
        iof_sample_masks = torch.tensor([0], dtype=torch.bool)
        iofs = torch.tensor([0], dtype=torch.float)

    return dict(
        _doc=doc,
        input_ids=input_ids,
        attention_mask=attention_mask,
        word_mask=word_mask,
        span_mask=span_mask,
        entity_spans_token=entity_spans_token,
        entity_sizes=entity_sizes,
        entity_types=entity_types,
        token_sample_masks=token_sample_mask,
        entity_sample_masks=entity_sample_masks,
        l_offsets=l_offsets,
        r_offsets=r_offsets,
        offset_sample_masks=offset_sample_masks,
        iofs=iofs,
        iof_sample_masks=iof_sample_masks,
        window_mask=window_mask,
        token_masks_bool=token_masks_bool,
    )


def create_eval_sample(doc: Document, offset_limit: int = 4):
    input_ids = doc.encodings
    token_count = len(doc.tokens)
    context_size = len(input_ids)

    # all tokens
    token_spans, word_mask, token_sizes = [], [], []
    for t in doc.tokens:
        token_spans.append(t.span)
        word_mask.append(create_mask(*t.span, context_size))
        token_sizes.append(t.span_end - t.span_start)

    word_mask = torch.stack(word_mask)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    entity_spans_token = []
    span_mask = []
    window_mask = []

    for seed_size in [1, 2]:
        for i in range(0, token_count):
            w_left = i
            w_right = min(token_count, i + seed_size)
            span = doc.tokens[w_left:w_right].span
            span_token = doc.tokens[w_left:w_right].span_token
            if span not in entity_spans:
                entity_spans.append(span)
                entity_spans_token.append(span_token)
                entity_masks.append(create_mask(*span, context_size))
                span_mask.append(create_mask(*span_token, token_count))
                window_mask.append(
                    create_mask(
                        max(0, span_token[0] - offset_limit * 2),
                        min(token_count, span_token[1] + offset_limit * 2),
                        token_count,
                    )
                )
                entity_sizes.append(w_right - w_left)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    attention_mask = torch.ones(context_size, dtype=torch.bool)

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        span_mask = torch.stack(span_mask)
        window_mask = torch.stack(window_mask)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)
        entity_spans_token = torch.tensor(entity_spans_token, dtype=torch.long)

        entity_sample_masks = torch.tensor(
            [1] * entity_masks.shape[0], dtype=torch.bool
        )
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        span_mask = torch.zeros([1, token_count], dtype=torch.bool)
        window_mask = torch.zeros([1, token_count], dtype=torch.bool)

    return dict(
        _doc=doc,
        input_ids=input_ids,
        attention_mask=attention_mask,
        word_mask=word_mask,
        entity_masks=entity_masks,
        span_mask=span_mask,
        entity_sizes=entity_sizes,
        entity_spans=entity_spans,
        entity_spans_token=entity_spans_token,
        entity_sample_masks=entity_sample_masks,
        window_mask=window_mask,
        token_masks_bool=token_masks_bool,
    )


def create_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        if key[0] == "_":
            padded_batch[key] = [s[key] for s in batch]
            continue
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = common.padded_stack([s[key] for s in batch])

    return padded_batch


def collate_batch_span_classification_data(batch):

    b_span_type = []
    b_span = []
    b_encodings = []
    b_tokens = []
    b_batch_idx = []
    b_label = []
    b_template_encoding = []
    b_doc_id = []

    for item in batch:
        b_doc_id.append(item["doc_id"])
        b_span.append((item["start_idx"], item["end_idx"]))
        b_encodings.append(item["input_ids"])
        b_tokens.append(item["tokens"])
        if "span_type" in item.keys():
            b_span_type.append(item["span_type"])
            b_template_encoding.append(item["template_ids"])
        b_batch_idx.append(item["batch_idx"])
        b_label.append(item["label"])

    results = {
        "span_type": b_span_type,
        "doc_id": b_doc_id,
        "span": b_span,
        "input_ids": b_encodings,
        "tokens": b_tokens,
        "label": b_label,
        "batch_idx": b_batch_idx,
        "template_ids": b_template_encoding,
    }
    return results
