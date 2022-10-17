import torch
from torch import nn as nn
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, AutoConfig
from utils import common


class SEE_Few(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SEE_Few, self).__init__(config)

        config.output_hidden_states = True

        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_path, do_lower_case=args.lowercase
        )
        self.offset_limit = args.offset_limit

        self.seeding = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.entailing = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
        )

        self.expanding = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(args.prop_drop)

        self.init_weights()

    def _span_extraction_forward(self, data):
        bert_output = self.bert(
            input_ids=data["input_ids"], attention_mask=data["attention_mask"]
        )
        h_token = bert_output[0]

        h_word = common.pooling(h_token, data["word_mask"])
        h_span = common.pooling(h_word, data["span_mask"])
        h_window = common.pooling(h_word, data["window_mask"])

        h_seed = [h_span]
        h_expand = [h_span]

        h_cls = h_token[:, 0, :]
        h_cls = h_cls.unsqueeze(1).repeat(1, h_span.shape[1], 1)

        h_seed.append(h_cls)
        h_expand.append(h_window)

        h_seed = torch.cat(h_seed, dim=-1)
        h_seed = self.dropout(h_seed)

        h_expand = torch.cat(h_expand, dim=-1)
        h_expand = self.dropout(h_expand)

        p_seed = self.seeding(h_seed).squeeze(-1)
        p_offset = 2 * self.expanding(h_expand) - 1
        p_offset = p_offset * self.offset_limit

        return p_seed, p_offset

    def _span_classification_forward(self, batch):
        b_token_type_ids = []
        b_attention_masks = []
        b_input_ids = []

        tokens_batch = batch["tokens"]
        span_batch = batch["span"]
        input_ids_batch = batch["input_ids"]
        template_ids_batch = batch["template_ids"]

        for idx, span in enumerate(span_batch):
            candidate = tokens_batch[idx]._tokens[span[0] : span[1]]
            assert len(candidate) != 0

            # Todo:
            candidate_ids = input_ids_batch[idx][:]
            for t in candidate:
                t_ids = self.tokenizer.encode(t.phrase, add_special_tokens=False)
                candidate_ids += t_ids

            input_ids = candidate_ids + template_ids_batch[idx]
            attention_mask = torch.ones(len(input_ids), dtype=torch.bool)

            token_type_ids = []
            for i in range(len(input_ids)):
                if i < len(input_ids_batch[idx]):
                    token_type_ids.append(0)
                else:
                    token_type_ids.append(1)

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_attention_masks.append(attention_mask)
            b_token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.long))

        b_input_ids = common.padded_stack(b_input_ids).to(device=self.device)
        b_attention_masks = (
            common.padded_stack(b_attention_masks).float().to(device=self.device)
        )
        b_token_type_ids = common.padded_stack(b_token_type_ids).to(device=self.device)

        bert_output = self.bert(
            input_ids=b_input_ids,
            attention_mask=b_attention_masks,
            token_type_ids=b_token_type_ids,
        )

        h_cls = bert_output[0][:, 0, :]
        h_cls = self.dropout(h_cls)

        return self.entailing(h_cls)

    def forward(self, data, cls_stage: bool):
        if cls_stage:
            return self._span_classification_forward(data)
        else:
            return self._span_extraction_forward(data)
