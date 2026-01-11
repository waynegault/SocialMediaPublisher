import sqlite3

conn = sqlite3.connect("content_engine.db")
c = conn.cursor()

# Clear linkedin_mentions for re-test
c.execute("UPDATE stories SET linkedin_mentions = NULL")
conn.commit()
print("Cleared linkedin_mentions for re-test with incognito mode")

conn.close()
