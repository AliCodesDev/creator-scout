-- Creator Scout schema. Applied by data/seed.py, which drops and recreates everything:
-- the data is synthetic and reseedable, so there are no migrations.

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS creators;

-- One row per platform account (a person on Instagram and TikTok is two rows).
CREATE TABLE creators (
    id              text PRIMARY KEY,                 -- 'c_0042'
    platform        text NOT NULL CHECK (platform IN ('instagram', 'tiktok', 'youtube')),
    handle          text NOT NULL,
    display_name    text NOT NULL,
    bio             text NOT NULL,                    -- untrusted input: may contain prompt injections
    country         text NOT NULL,                    -- ISO 3166-1 alpha-2
    city            text NOT NULL,
    languages       text[] NOT NULL,                  -- ISO 639-1 codes, e.g. {ar,en}
    follower_count  integer NOT NULL CHECK (follower_count >= 0),
    engagement_rate real NOT NULL CHECK (engagement_rate >= 0),
    categories      text[] NOT NULL,                  -- the provider's coarse labels; can be wrong
    source          text NOT NULL                     -- which CreatorDataProvider supplied the row
);

-- Posts are the unit of evidence: embeddings live here, and every fit reason cites a post id.
CREATE TABLE posts (
    id          text PRIMARY KEY,                     -- 'p_00193'
    creator_id  text NOT NULL REFERENCES creators (id) ON DELETE CASCADE,
    caption     text NOT NULL,
    language    text NOT NULL,                        -- ar | en | fr | mixed
    posted_at   timestamptz NOT NULL,
    likes       integer NOT NULL CHECK (likes >= 0),
    comments    integer NOT NULL CHECK (comments >= 0)
);

CREATE INDEX posts_creator_id_idx ON posts (creator_id);
