"""Defensive parsing for structured model output."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

T = TypeVar("T")


class StructuredOutputError(ValueError):
    """Raised when model output cannot be decoded or validated."""


def _validate(data: Any, schema: type[T]) -> T:
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)  # type: ignore[attr-defined,no-any-return]
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)  # type: ignore[attr-defined,no-any-return]
    if isinstance(data, schema):
        return data
    raise StructuredOutputError(f"Schema {schema!r} is not Pydantic-compatible")


def _json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(item.strip() for item in fenced)
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1].strip())
                    break
    return list(dict.fromkeys(candidates))


def parse_json(text: str) -> Any:
    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as error:
            last_error = error
    raise StructuredOutputError(f"No valid JSON object found: {last_error}")


def parse_structured_response(text: str, schema: type[T]) -> T:
    try:
        data = parse_json(text)
        return _validate(data, schema)
    except StructuredOutputError:
        raise
    except Exception as error:
        raise StructuredOutputError(f"Structured response validation failed: {error}") from error
