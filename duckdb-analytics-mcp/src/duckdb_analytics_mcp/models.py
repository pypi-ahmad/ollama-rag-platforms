"""Pydantic models for tool inputs and outputs."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResponseFormat(StrEnum):
    """Allowed tool response formats."""

    MARKDOWN = "markdown"
    JSON = "json"


class ListDatasetsRequest(BaseModel):
    """Input model for listing datasets."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    limit: int = Field(default=25, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)
    response_format: ResponseFormat = ResponseFormat.MARKDOWN


class DescribeDatasetRequest(BaseModel):
    """Input model for describing a dataset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset: str = Field(..., min_length=1, max_length=256)
    sample_rows: int = Field(default=10, ge=1, le=200)
    response_format: ResponseFormat = ResponseFormat.MARKDOWN

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        if value.startswith("/") or value.startswith(".."):
            raise ValueError("dataset must be a relative dataset name from the catalog")
        return value


class QueryDatasetRequest(BaseModel):
    """Input model for querying a dataset with read-only SQL."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    dataset: str = Field(..., min_length=1, max_length=256)
    sql: str = Field(..., min_length=1, max_length=4000)
    limit: int = Field(default=25, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)
    response_format: ResponseFormat = ResponseFormat.MARKDOWN

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, value: str) -> str:
        if value.startswith("/") or value.startswith(".."):
            raise ValueError("dataset must be a relative dataset name from the catalog")
        return value


class DatasetSummary(BaseModel):
    """Dataset metadata for catalog listings."""

    name: str
    path: str
    file_format: str
    size_bytes: int
    modified_at: datetime


class PaginatedDatasetsResult(BaseModel):
    """Paginated list dataset result payload."""

    total_count: int
    count: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None
    datasets: list[DatasetSummary]


class DatasetColumn(BaseModel):
    """Schema column metadata."""

    name: str
    data_type: str
    nullable: str | None = None


class DatasetDescription(BaseModel):
    """Dataset description output payload."""

    dataset: DatasetSummary
    row_count: int
    columns: list[DatasetColumn]
    sample_rows: list[dict[str, object]]


class QueryResult(BaseModel):
    """SQL query result payload."""

    dataset: str
    sql: str
    total_count: int
    count: int
    limit: int
    offset: int
    has_more: bool
    next_offset: int | None
    columns: list[str]
    rows: list[dict[str, object]]


class HealthStatus(BaseModel):
    """Health check payload."""

    status: str
    server: str
    dataset_dir: str
    dataset_count: int
    checked_at: datetime
