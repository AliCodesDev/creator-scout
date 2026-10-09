"""Serves the synthetic dataset in data/creators.json through the provider interface.

It does not simulate content drift: a refresh returns the same posts that were ingested.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from scout.config import REPO_ROOT
from scout.providers.base import CreatorRecord, PostRecord

DATASET_PATH = REPO_ROOT / "data" / "creators.json"


class SyntheticProvider:
    name = "synthetic"
    cost_per_call_usd = 0.002  # simulated, in the range of commercial creator-data APIs

    def __init__(self, path: Path = DATASET_PATH):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._creators = {c["id"]: CreatorRecord.model_validate(c) for c in raw}

    def list_creators(self) -> Iterable[CreatorRecord]:
        return self._creators.values()

    def get_recent_posts(self, creator_id: str, limit: int = 10) -> list[PostRecord]:
        posts = self._creators[creator_id].posts
        return sorted(posts, key=lambda p: p.posted_at, reverse=True)[:limit]
