"""The boundary between Creator Scout and wherever creator data comes from.

Providers are used in two ways:
- `list_creators`: bulk ingestion into our own database, where SQL filtering and vector search run.
- `get_recent_posts`: a paid, per-creator refresh, called at search time only for the shortlist,
  so the LLM scores fresh content without paying for fresh data on every candidate.

Provider data is validated on the way in: partner data is untrusted.
"""

from collections.abc import Iterable
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field


class PostRecord(BaseModel):
    id: str
    caption: str
    language: Literal["ar", "en", "fr", "mixed"]
    posted_at: datetime
    likes: int = Field(ge=0)
    comments: int = Field(ge=0)


class CreatorRecord(BaseModel):
    id: str
    platform: Literal["instagram", "tiktok", "youtube"]
    handle: str
    display_name: str
    bio: str
    country: str = Field(min_length=2, max_length=2)
    city: str
    languages: list[str]
    follower_count: int = Field(ge=0)
    engagement_rate: float = Field(ge=0)
    categories: list[str]
    posts: list[PostRecord]  # the snapshot available at ingestion time


class CreatorDataProvider(Protocol):
    name: str
    cost_per_call_usd: float  # cost of one get_recent_posts call

    def list_creators(self) -> Iterable[CreatorRecord]: ...

    def get_recent_posts(self, creator_id: str, limit: int = 10) -> list[PostRecord]: ...
