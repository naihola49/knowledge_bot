"""Validation helpers for node boundaries."""

from __future__ import annotations

from pydantic import ValidationError

from book_companion.schema.models import (
    ComprehensionInputStateModel,
    GraphStateModel,
    Output1Model,
    Output2Model,
)


def _raise_validation_error(context: str, err: ValidationError) -> None:
    raise ValueError(f"{context} schema validation failed: {err}") from err


def validate_graph_state(state: dict, *, context: str) -> dict:
    try:
        return GraphStateModel.model_validate(state).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error(context, err)


def validate_comprehension_input_state(state: dict) -> dict:
    try:
        return ComprehensionInputStateModel.model_validate(state).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("comprehension input", err)


def validate_output_1(output_1: dict) -> dict:
    try:
        return Output1Model.model_validate(output_1).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("output_1", err)


def validate_output_2(output_2: dict) -> dict:
    try:
        return Output2Model.model_validate(output_2).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("output_2", err)
