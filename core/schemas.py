"""Validated records shared by collection, scoring and saved JSON."""

from dataclasses import asdict, field
import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, StringConstraints, model_serializer
from pydantic.dataclasses import dataclass


CONFIG = {"extra": "forbid", "validate_assignment": True}
Text = Annotated[str, StringConstraints(strict=True, pattern=r"\S")]
Messages = list[dict[str, StrictStr]]


@dataclass(config=CONFIG)
class Prompts:
    instruction_policy: Text
    a_instruction: Text
    b_instruction: Text
    judge_rubric: Text


@dataclass(config={"extra": "ignore"})
class DatasetRow:
    id: StrictInt
    question: Text
    chosen: Text
    rejected: Text
    system: StrictStr = ""


@dataclass(config=CONFIG)
class Judgment:
    choice: Literal["option1", "option2", "tie"]
    presented_order: list[Literal["y1", "y2", "y3"]] = field(default_factory=list)
    winner: Literal["y1", "y2", "y3"] | None = None


@dataclass(config=CONFIG)
class IntermediatePreference:
    id: StrictInt
    x: Text
    system: StrictStr
    a_messages: Messages
    y_chosen: Text
    y_rejected: Text
    finish_reasons: dict[str, StrictStr]

    def __post_init__(self):
        if self.y_chosen == self.y_rejected:
            raise ValueError("Intermediate preferences need distinct candidates")


@dataclass(config=CONFIG)
class ScoringRequest:
    id: StrictInt
    a_messages: Messages
    intermediates: Annotated[list[Text], Field(min_length=2, max_length=3)]
    b_messages: Annotated[list[Messages], Field(min_length=2, max_length=3)]
    final_answers: Annotated[list[Text], Field(min_length=2, max_length=2)]
    candidate_mask: Annotated[list[StrictBool], Field(min_length=2, max_length=3)]
    reference_intermediates: Annotated[list[Text], Field(min_length=2, max_length=2)]
    reference_b_messages: Annotated[list[Messages], Field(min_length=2, max_length=2)]
    reference_candidate_mask: Annotated[list[StrictBool], Field(min_length=2, max_length=2)]
    request_sha256: str = ""

    def __post_init__(self):
        if len(self.intermediates) != len(self.b_messages) or len(self.intermediates) != len(self.candidate_mask):
            raise ValueError("Candidates, B contexts and mask must have matching lengths")
        if not any(self.candidate_mask):
            raise ValueError("Each question needs a usable intermediate candidate")
        if not any(self.reference_candidate_mask):
            raise ValueError("Each question needs a usable reference candidate")

        payload = asdict(self)
        payload.pop("request_sha256")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        if self.request_sha256 and self.request_sha256 != digest:
            raise ValueError("Scoring request hash does not match its contents")
        self.request_sha256 = digest


@dataclass(config=CONFIG)
class QuestionRecord:
    id: StrictInt
    x: Text
    system: StrictStr
    z_chosen: Text
    z_rejected: Text
    a_messages: Messages
    finish_reasons: dict[str, StrictStr] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    y1: str | None = None
    y2: str | None = None
    y3: str | None = None
    reference_intermediates: list[str] | None = None
    intermediate_judgments: list[Judgment] = field(default_factory=list)
    preference_edges: list[tuple[str, str]] = field(default_factory=list)
    comparisons_accepted: bool = False
    discard_reason: str | None = None
    scoring_request: ScoringRequest | None = None

    @model_serializer(mode="wrap")
    def serialize(self, handler):
        # Preserve absent stages in existing question JSON; keep nested null winners.
        return {key: value for key, value in handler(self).items() if value is not None}
