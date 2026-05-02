"""Validation helpers for node boundaries."""

from __future__ import annotations

from pydantic import ValidationError

from book_companion.schema.models import (
    ComprehensionInputStateModel,
    GraphStateModel,
    Output1Model,
    Output2Model,
    Output3Model,
    PremiseDocModel,
    PremiseIngestionRequestModel,
    QuerySpecModel,
    RunConfigModel, # entry from cli/yaml 
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


def validate_run_config(data: dict) -> RunConfigModel:
    try:
        return RunConfigModel.model_validate(data)
    except ValidationError as err:
        _raise_validation_error("run_config", err)


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


def validate_output_3(output_3: dict) -> dict:
    try:
        return Output3Model.model_validate(output_3).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("output_3", err)


def validate_premise_ingestion_request(request: dict) -> dict:
    try:
        return PremiseIngestionRequestModel.model_validate(request).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("premise_ingestion_request", err)


def validate_query_spec(spec: dict) -> dict:
    try:
        return QuerySpecModel.model_validate(spec).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("query_spec", err)


def validate_premise_doc(doc: dict) -> dict:
    try:
        return PremiseDocModel.model_validate(doc).model_dump(exclude_none=False)
    except ValidationError as err:
        _raise_validation_error("premise_doc", err)
