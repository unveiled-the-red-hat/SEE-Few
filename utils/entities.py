from typing import List
from collections import namedtuple

# end is exclusive
LabelSpan = namedtuple("LabelSpan", ["doc_id", "start", "end", "entity_type", "cls_score"])


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)

    def __str__(self) -> str:
        return self._identifier + "=" + self._verbose_name


class Token:
    def __init__(
        self, tid: int, index: int, span_start: int, span_end: int, phrase: str
    ):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        # start of token span in document (inclusive)
        self._span_start = span_start
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self._tokens[0].index, self._tokens[-1].index + 1

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start : s.stop : s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __str__(self) -> str:
        return " ".join([str(t) for t in self._tokens])


class Entity:
    def __init__(
        self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str
    ):
        self._eid = eid  # ID within the corresponding dataset

        self._entity_type = entity_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    def as_tuple_token(self):
        return self._tokens[0].index, self._tokens[-1].index + 1, self._entity_type

    def as_label_span(self):
        return LabelSpan(self._eid, self._tokens[0].index, self._tokens[-1].index + 1, self._entity_type, 1)

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self._tokens[0].index, self._tokens[-1].index + 1

    @property
    def tokens_len(self):
        return self._tokens[-1].index + 1 - self._tokens[0].index

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Document:
    def __init__(
        self,
        doc_id: int,
        tokens: List[Token],
        entities: List[Entity],
        encodings: List[int],
    ):
        self._doc_id = doc_id  # ID within the corresponding dataset
        self._tokens = tokens
        self._entities = entities
        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encodings = encodings

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    # @property
    # def token_text(self):
    #     return self._token_text

    @property
    def encodings(self):
        return self._encodings

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)
