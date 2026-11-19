import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
CrowdSense — SQLite Database Viewer Helper
Displays the schema and all saved analysis history records in a clean table format.
Usage:
    python view_db.py
"""

import sqlite3
import os

DB_PATH = "crowdsense.db"

def display_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_PATH}' not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Total records
    cursor.execute("SELECT COUNT(*) FROM analyses")
    total = cursor.fetchone()[0]

    print("=" * 95)
    print(f"  CROWDSENSE DATABASE VIEWER (Total Records: {total})")
    print("=" * 95)

    # 2. Table Columns
    cursor.execute("PRAGMA table_info(analyses)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"\n[Table Schema: 'analyses']")
    print("Columns: " + ", ".join(columns))

    # 3. Formatted Records
    query = """
        SELECT id, timestamp, filename, crowd_count, density_level, risk_level,
               estimation_reliability, processing_time_ms
        FROM analyses
        ORDER BY id DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        print("\nNo analysis records found in database.")
        conn.close()
        return

    print("\n" + "-" * 95)
    header = f"{'ID':<4} | {'Timestamp':<19} | {'Filename':<22} | {'Count':<6} | {'Level':<10} | {'Reliability':<11} | {'Time (ms)':<9}"
    print(header)
    print("-" * 95)

    for r in rows:
        r_id, ts, fname, count, level, risk, rel, ptime = r
        clean_ts = ts[:19].replace("T", " ") if ts else "N/A"
        clean_fname = fname[:20] + ".." if len(fname) > 22 else fname
        print(f"{r_id:<4} | {clean_ts:<19} | {clean_fname:<22} | {count:<6} | {level:<10} | {rel:>9}% | {ptime:>8.1f}ms")

    print("-" * 95)
    conn.close()

if __name__ == "__main__":
    display_database()

