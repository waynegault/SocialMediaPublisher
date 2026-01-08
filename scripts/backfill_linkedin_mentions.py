import sqlite3
import json
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "content_engine.db"

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT id, company_mention_enrichment, linkedin_mentions FROM stories WHERE company_mention_enrichment IS NOT NULL ORDER BY id")
rows = cur.fetchall()
updated = 0

for r in rows:
    cm = r['company_mention_enrichment']
    lm = r['linkedin_mentions']
    is_empty = not lm or lm.strip() in ("", "[]", "NULL")
    if is_empty:
        # Extract company name from expected sentence
        name = cm.split(' is ', 1)[0].strip().strip('"') if cm else None
        if name:
            mentions = [{"name": name, "urn": None, "type": "organization"}]
            cur.execute('UPDATE stories SET linkedin_mentions = ? WHERE id = ?', (json.dumps(mentions), r['id']))
            updated += 1

conn.commit()
print(f'Backfilled {updated} rows')
conn.close()