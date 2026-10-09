"""Load creator data into Postgres through the CreatorDataProvider interface.

Drops and recreates the schema, then ingests everything the provider lists.

Usage (from api/): uv run python ../data/seed.py
"""

import psycopg

from scout.config import REPO_ROOT, settings
from scout.providers.base import CreatorDataProvider
from scout.providers.synthetic import SyntheticProvider

SCHEMA_PATH = REPO_ROOT / "api" / "scout" / "schema.sql"


def seed(provider: CreatorDataProvider) -> None:
    creators = list(provider.list_creators())
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text())
        cur.executemany(
            """
            INSERT INTO creators (id, platform, handle, display_name, bio, country, city,
                                  languages, follower_count, engagement_rate, categories, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    c.id,
                    c.platform,
                    c.handle,
                    c.display_name,
                    c.bio,
                    c.country,
                    c.city,
                    c.languages,
                    c.follower_count,
                    c.engagement_rate,
                    c.categories,
                    provider.name,
                )
                for c in creators
            ],
        )
        cur.executemany(
            """
            INSERT INTO posts (id, creator_id, caption, language, posted_at, likes, comments)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (p.id, c.id, p.caption, p.language, p.posted_at, p.likes, p.comments)
                for c in creators
                for p in c.posts
            ],
        )
    print(
        f"Seeded {len(creators)} creators and {sum(len(c.posts) for c in creators)} posts "
        f"from provider '{provider.name}'."
    )


if __name__ == "__main__":
    seed(SyntheticProvider())
