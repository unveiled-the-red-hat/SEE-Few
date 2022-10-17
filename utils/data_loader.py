import json
from collections import OrderedDict
from logging import Logger
import os
from typing import Iterable, List
import numpy as np
import string
from torch.utils import data

from tqdm import tqdm
from transformers import BertTokenizer
from utils.entities import EntityType, Entity, Document, Token
from torch.utils.data import Dataset as TorchDataset
from utils import sampling


class Dataset(TorchDataset):
    TRAIN_MODE = "train"
    EVAL_MODE = "eval"

    def __init__(self, label, offset_limit=4, seed_threshold=0.7):
        self._label = label
        self._mode = Dataset.TRAIN_MODE

        self._documents = OrderedDict()
        self._entities = OrderedDict()

        # current ids
        self._doc_id = 0
        self._eid = 0
        self._tid = 0

        self._offset_limit = offset_limit
        self._seed_threshold = seed_threshold

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, entity_mentions, doc_encoding) -> Document:
        document = Document(self._doc_id, tokens, entity_mentions, doc_encoding)
        self._documents[self._doc_id] = document
        self._doc_id += 1
        return document

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(
                doc, self._offset_limit, self._seed_threshold
            )
        else:
            return sampling.create_eval_sample(doc, self._offset_limit)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def document_count(self):
        return len(self._documents)


class NerDataProcessor(object):
    def __init__(self, args):
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_path, do_lower_case=args.lowercase
        )

        self.args = args

        types = json.load(open(args.types_path), object_pairs_hook=OrderedDict)
        self.entity_types = OrderedDict()
        self.idx2entity_type = OrderedDict()

        template = []

        for i, (key, v) in enumerate(types["entities"].items()):
            entity_type = EntityType(key, i, v["short"], v["verbose"])
            self.entity_types[key] = entity_type
            self.idx2entity_type[i] = entity_type

            template.append(v["template"])

        self.template_encoding = [self.tokenizer.encode(x)[1:] for x in template]

        self.datasets = dict()

    def read(self, dataset_paths: dict):
        for dataset_label, dataset_path in dataset_paths.items():
            dataset = Dataset(
                dataset_label,
                offset_limit=self.args.offset_limit,
                seed_threshold=self.args.seed_threshold,
            )
            self._parse_dataset(dataset_path, dataset)
            self.datasets[dataset_label] = dataset

    def get_dataset(self, label) -> Dataset:
        return self.datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self.idx2entity_type[idx]
        return entity

    def _parse_dataset(self, dataset_path: str, dataset: Dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc: dict, dataset: Dataset):
        tokens = doc["tokens"]
        entities = doc["entities"]

        # parse tokens
        doc_tokens, doc_encoding = self._parse_tokens(tokens, dataset)

        # parse entity mentions
        entities = self._parse_entities(entities, doc_tokens, dataset)

        # create document
        dataset.create_document(doc_tokens, entities, doc_encoding)

    def _parse_tokens(self, raw_tokens, dataset: Dataset):
        doc_tokens = []
        doc_encoding = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
        for i, token_phrase in enumerate(raw_tokens):
            token_encoding = self.tokenizer.encode(
                token_phrase, add_special_tokens=False
            )
            span_start, span_end = (
                len(doc_encoding),
                len(doc_encoding) + len(token_encoding),
            )
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding

        doc_encoding += [self.tokenizer.convert_tokens_to_ids("[SEP]")]

        return doc_tokens, doc_encoding

    def _parse_entities(
        self, raw_entities, doc_tokens, dataset: Dataset
    ) -> List[Entity]:
        entities = []

        for raw_entity in raw_entities:
            entity_type = self.entity_types[raw_entity["type"]]
            start, end = raw_entity["start"], raw_entity["end"]

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    @property
    def entity_type_count(self):
        return len(self.entity_types)

    def __str__(self):
        string = ""
        for dataset in self.datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()
