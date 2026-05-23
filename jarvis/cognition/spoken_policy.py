from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Any

from pydantic import Field, field_validator

from jarvis.cognition.models import (
    CognitionModel,
    CognitionResponse,
    CognitionResponseKind,
    SpokenResponseStyle,
    new_id,
)
from jarvis.runtime.observability.structured_logger import get_logger


class SpokenDialogueAct(StrEnum):
    """
    Type of spoken response being prepared.
    """

    DIRECT_REPLY = "direct_reply"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"
    FAILURE_FALLBACK = "failure_fallback"
    ACKNOWLEDGEMENT = "acknowledgement"


class SpokenDialogueTone(StrEnum):
    """
    Voice-native tone profile.
    """

    CALM = "calm"
    DIRECT = "direct"
    SUPPORTIVE = "supportive"


class SpokenDialoguePolicyDecision(CognitionModel):
    """
    Final spoken policy decision for one response.
    """

    decision_id: str = Field(default_factory=new_id)
    request_id: str | None = None
    original_text: str
    spoken_text: str
    style: SpokenResponseStyle = SpokenResponseStyle.CONCISE
    act: SpokenDialogueAct = SpokenDialogueAct.DIRECT_REPLY
    tone: SpokenDialogueTone = SpokenDialogueTone.DIRECT
    truncated: bool = False
    sentence_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("spoken_text")
    @classmethod
    def _spoken_text_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("spoken_text cannot be empty.")

        return value


@dataclass(frozen=True, slots=True)
class SpokenDialoguePolicyConfig:
    """
    Configuration for spoken dialogue shaping.

    The defaults are intentionally concise because real-time voice assistants
    must respond quickly and naturally.
    """

    name: str = "spoken_dialogue_policy"
    concise_max_chars: int = 280
    normal_max_chars: int = 700
    detailed_max_chars: int = 1_600
    concise_max_sentences: int = 2
    normal_max_sentences: int = 5
    fallback_text: str = "I understand, sir."
    clarification_text: str = "What should I focus on, sir?"
    refusal_text: str = "I cannot help with that, sir."
    failure_fallback_text: str = "I had trouble thinking that through, sir."
    acknowledgement_text: str = "Yes sir."
    remove_markdown: bool = True
    ensure_terminal_punctuation: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name cannot be empty.")

        if self.concise_max_chars <= 0:
            raise ValueError("concise_max_chars must be greater than zero.")

        if self.normal_max_chars <= 0:
            raise ValueError("normal_max_chars must be greater than zero.")

        if self.detailed_max_chars <= 0:
            raise ValueError("detailed_max_chars must be greater than zero.")

        if self.concise_max_sentences <= 0:
            raise ValueError("concise_max_sentences must be greater than zero.")

        if self.normal_max_sentences <= 0:
            raise ValueError("normal_max_sentences must be greater than zero.")

        self._validate_text("fallback_text", self.fallback_text)
        self._validate_text("clarification_text", self.clarification_text)
        self._validate_text("refusal_text", self.refusal_text)
        self._validate_text("failure_fallback_text", self.failure_fallback_text)
        self._validate_text("acknowledgement_text", self.acknowledgement_text)

    @staticmethod
    def _validate_text(name: str, value: str) -> None:
        if not value.strip():
            raise ValueError(f"{name} cannot be empty.")


@dataclass(frozen=True, slots=True)
class SpokenDialoguePolicySnapshot:
    """
    Observable policy diagnostics.
    """

    name: str
    prepared_count: int
    concise_count: int
    normal_count: int
    detailed_count: int
    truncated_count: int
    fallback_count: int
    last_request_id: str | None
    last_act: SpokenDialogueAct | None
    last_style: SpokenResponseStyle | None
    last_error: str | None


class SpokenDialoguePolicy:
    """
    Voice-native response shaping policy.

    Responsibilities:
    - normalize generated text for speech
    - remove markdown/code/list artifacts
    - enforce style-specific length limits
    - enforce concise spoken sentence limits
    - produce safe fallback text
    - update CognitionResponse text without changing response ownership

    Non-responsibilities:
    - no LLM calls
    - no TTS/playback
    - no memory lookup
    - no tool execution
    """

    _whitespace_pattern = re.compile(r"\s+")
    _markdown_token_pattern = re.compile(r"[*_`#>]+")
    _list_marker_pattern = re.compile(r"(^|\n)\s*[-*+]\s+")
    _numbered_marker_pattern = re.compile(r"(^|\n)\s*\d+[.)]\s+")

    def __init__(
        self,
        *,
        config: SpokenDialoguePolicyConfig | None = None,
    ) -> None:
        self._config = config or SpokenDialoguePolicyConfig()
        self._config.validate()

        self._lock = RLock()
        self._logger = get_logger("cognition.spoken_policy")

        self._prepared_count = 0
        self._concise_count = 0
        self._normal_count = 0
        self._detailed_count = 0
        self._truncated_count = 0
        self._fallback_count = 0
        self._last_request_id: str | None = None
        self._last_act: SpokenDialogueAct | None = None
        self._last_style: SpokenResponseStyle | None = None
        self._last_error: str | None = None

    @property
    def name(self) -> str:
        return self._config.name

    def prepare(
        self,
        text: str,
        *,
        request_id: str | None = None,
        style: SpokenResponseStyle = SpokenResponseStyle.CONCISE,
        act: SpokenDialogueAct = SpokenDialogueAct.DIRECT_REPLY,
        tone: SpokenDialogueTone = SpokenDialogueTone.DIRECT,
        metadata: dict[str, Any] | None = None,
    ) -> SpokenDialoguePolicyDecision:
        """
        Prepare raw text for voice-native speech.
        """

        original_text = text
        working_text = self._fallback_for_act(act) if not text.strip() else text

        if not text.strip():
            self._increment_fallback_count()

        if self._config.remove_markdown:
            working_text = self._remove_markdown(working_text)

        working_text = self._normalize_whitespace(working_text)
        working_text = self._limit_sentences(
            text=working_text,
            style=style,
        )

        if self._config.ensure_terminal_punctuation:
            working_text = self._ensure_terminal_punctuation(working_text)

        bounded_text = self._bounded_text(
            text=working_text,
            max_chars=self._max_chars_for_style(style),
        )
        truncated = bounded_text != working_text
        sentence_count = self._sentence_count(bounded_text)

        decision = SpokenDialoguePolicyDecision(
            request_id=request_id,
            original_text=original_text,
            spoken_text=bounded_text,
            style=style,
            act=act,
            tone=tone,
            truncated=truncated,
            sentence_count=sentence_count,
            metadata={
                "policy": self.name,
                **(metadata or {}),
            },
        )

        self._record_decision(decision)

        self._logger.info(
            "spoken_dialogue_policy_prepared",
            policy=self.name,
            request_id=request_id,
            style=style.value,
            act=act.value,
            truncated=truncated,
            sentence_count=sentence_count,
        )

        return decision

    def apply_to_response(
        self,
        response: CognitionResponse,
        *,
        style: SpokenResponseStyle | None = None,
        act: SpokenDialogueAct | None = None,
        tone: SpokenDialogueTone = SpokenDialogueTone.DIRECT,
    ) -> CognitionResponse:
        """
        Return a copy of CognitionResponse with voice-shaped text.
        """

        selected_style = style or self._style_from_response(response)
        selected_act = act or self._act_from_response(response)

        decision = self.prepare(
            response.text,
            request_id=response.request_id,
            style=selected_style,
            act=selected_act,
            tone=tone,
            metadata={
                "response_id": response.response_id,
                "response_kind": response.kind.value,
            },
        )

        return response.model_copy(
            update={
                "text": decision.spoken_text,
                "metadata": {
                    **response.metadata,
                    "spoken_policy": self.name,
                    "spoken_decision_id": decision.decision_id,
                    "spoken_style": decision.style.value,
                    "spoken_act": decision.act.value,
                    "spoken_truncated": decision.truncated,
                    "spoken_sentence_count": decision.sentence_count,
                },
            }
        )

    def snapshot(self) -> SpokenDialoguePolicySnapshot:
        """
        Return policy diagnostics.
        """

        with self._lock:
            return SpokenDialoguePolicySnapshot(
                name=self.name,
                prepared_count=self._prepared_count,
                concise_count=self._concise_count,
                normal_count=self._normal_count,
                detailed_count=self._detailed_count,
                truncated_count=self._truncated_count,
                fallback_count=self._fallback_count,
                last_request_id=self._last_request_id,
                last_act=self._last_act,
                last_style=self._last_style,
                last_error=self._last_error,
            )

    def reset(self) -> None:
        """
        Reset counters.
        """

        with self._lock:
            self._prepared_count = 0
            self._concise_count = 0
            self._normal_count = 0
            self._detailed_count = 0
            self._truncated_count = 0
            self._fallback_count = 0
            self._last_request_id = None
            self._last_act = None
            self._last_style = None
            self._last_error = None

        self._logger.info("spoken_dialogue_policy_reset", policy=self.name)

    def _record_decision(
        self,
        decision: SpokenDialoguePolicyDecision,
    ) -> None:
        with self._lock:
            self._prepared_count += 1
            self._last_request_id = decision.request_id
            self._last_act = decision.act
            self._last_style = decision.style
            self._last_error = None

            if decision.style == SpokenResponseStyle.CONCISE:
                self._concise_count += 1

            elif decision.style == SpokenResponseStyle.NORMAL:
                self._normal_count += 1

            elif decision.style == SpokenResponseStyle.DETAILED:
                self._detailed_count += 1

            if decision.truncated:
                self._truncated_count += 1

    def _increment_fallback_count(self) -> None:
        with self._lock:
            self._fallback_count += 1

    def _fallback_for_act(
        self,
        act: SpokenDialogueAct,
    ) -> str:
        if act == SpokenDialogueAct.CLARIFICATION:
            return self._config.clarification_text

        if act == SpokenDialogueAct.REFUSAL:
            return self._config.refusal_text

        if act == SpokenDialogueAct.FAILURE_FALLBACK:
            return self._config.failure_fallback_text

        if act == SpokenDialogueAct.ACKNOWLEDGEMENT:
            return self._config.acknowledgement_text

        return self._config.fallback_text

    def _style_from_response(
        self,
        response: CognitionResponse,
    ) -> SpokenResponseStyle:
        value = response.metadata.get("spoken_style")

        if isinstance(value, str):
            try:
                return SpokenResponseStyle(value)

            except ValueError:
                return SpokenResponseStyle.CONCISE

        return SpokenResponseStyle.CONCISE

    @staticmethod
    def _act_from_response(
        response: CognitionResponse,
    ) -> SpokenDialogueAct:
        if response.kind == CognitionResponseKind.CLARIFICATION:
            return SpokenDialogueAct.CLARIFICATION

        if response.kind == CognitionResponseKind.REFUSAL:
            return SpokenDialogueAct.REFUSAL

        if response.kind == CognitionResponseKind.ERROR_FALLBACK:
            return SpokenDialogueAct.FAILURE_FALLBACK

        return SpokenDialogueAct.DIRECT_REPLY

    def _remove_markdown(
        self,
        text: str,
    ) -> str:
        cleaned = text.replace("```", " ")
        cleaned = self._list_marker_pattern.sub(" ", cleaned)
        cleaned = self._numbered_marker_pattern.sub(" ", cleaned)
        cleaned = self._markdown_token_pattern.sub("", cleaned)

        return cleaned

    def _normalize_whitespace(
        self,
        text: str,
    ) -> str:
        return self._whitespace_pattern.sub(" ", text).strip()

    def _limit_sentences(
        self,
        *,
        text: str,
        style: SpokenResponseStyle,
    ) -> str:
        if style == SpokenResponseStyle.DETAILED:
            return text

        max_sentences = (
            self._config.concise_max_sentences
            if style == SpokenResponseStyle.CONCISE
            else self._config.normal_max_sentences
        )

        sentences = self._split_sentences(text)

        if len(sentences) <= max_sentences:
            return text

        return " ".join(sentences[:max_sentences]).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences: list[str] = []
        current: list[str] = []

        for character in text:
            current.append(character)

            if character in ".!?":
                sentence = "".join(current).strip()

                if sentence:
                    sentences.append(sentence)

                current = []

        remaining = "".join(current).strip()

        if remaining:
            sentences.append(remaining)

        return sentences

    def _max_chars_for_style(
        self,
        style: SpokenResponseStyle,
    ) -> int:
        if style == SpokenResponseStyle.DETAILED:
            return self._config.detailed_max_chars

        if style == SpokenResponseStyle.NORMAL:
            return self._config.normal_max_chars

        return self._config.concise_max_chars

    @staticmethod
    def _ensure_terminal_punctuation(text: str) -> str:
        if not text:
            return text

        if text.endswith((".", "!", "?", "...")):
            return text

        return f"{text}."

    @staticmethod
    def _bounded_text(
        *,
        text: str,
        max_chars: int,
    ) -> str:
        if len(text) <= max_chars:
            return text

        if max_chars <= 3:
            return text[:max_chars]

        return f"{text[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _sentence_count(text: str) -> int:
        sentences = SpokenDialoguePolicy._split_sentences(text)

        return len(sentences)