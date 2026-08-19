"""Request schemas for API endpoints."""
from pydantic import BaseModel, field_validator


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty.")
        if len(v) > 2000:
            raise ValueError("Query is too long (max 2000 characters).")
        return v


class IngestRequest(BaseModel):
    """Optional body for /api/ingest. Defaults to data/raw."""
    data_dir: str = "./data/raw"
