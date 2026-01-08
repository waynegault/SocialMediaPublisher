"""Backfill script to resolve LinkedIn URNs for existing mentions."""
import json
import time
from pathlib import Path
import requests

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from database import Database
from company_mention_enricher import CompanyMentionEnricher
from config import Config

DB = Path(__file__).resolve().parents[1] / "content_engine.db"

def run():
    db = Database(str(DB))
    enricher = CompanyMentionEnricher(db, None, None)  # type: ignore

    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, linkedin_mentions FROM stories WHERE linkedin_mentions IS NOT NULL"
        )
        rows = cursor.fetchall()

    updated = 0

    for row in rows:
        id_ = row[0]
        mentions_raw = row[1]
        try:
            mentions = json.loads(mentions_raw) if mentions_raw else []
        except Exception:
            mentions = []

        modified = False
        for m in mentions:
            if isinstance(m, dict) and (m.get("urn") is None):
                name = m.get("name")
                if not name:
                    continue
                urn = enricher._resolve_linkedin_urn(name)
                if urn:
                    m["urn"] = urn
                    modified = True
                    time.sleep(1)  # be polite

        if modified:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE stories SET linkedin_mentions = ? WHERE id = ?",
                    (json.dumps(mentions), id_),
                )
            updated += 1

    print(f"Resolved URNs for {updated} stories")


if __name__ == "__main__":
    run()