"""
Database Migration Script
==========================
Migrates the attendance.db from old schema (employee_name based)
to new schema (emp_no based with AMS push tracking).

Steps:
  1. Back up existing attendance.db -> attendance.db.backup_<timestamp>
  2. Read employees.csv
  3. Drop old tables, create new schema
  4. Populate `employees` table from CSV
  5. Print summary

Usage:
    python migrate_db.py

After running, verify with:
    Open attendance.db in DB Browser -> Browse Data -> employees table
"""

import os
import csv
import sqlite3
import shutil
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "attendance.db")
CSV_PATH = os.path.join(_HERE, "employees.csv")


def backup_existing_db():
    """Make a timestamped backup of the existing DB before migrating."""
    if not os.path.exists(DB_PATH):
        print("[OK] No existing attendance.db — fresh start.")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DB_PATH}.backup_{stamp}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"[OK] Backup created -> {backup_path}")


def load_csv():
    """Read employees.csv and return list of dicts.

    Accepts the actual Excel column names:
        S. No.         (ignored — row counter)
        Employee ID    -> emp_no  (kept as TEXT to preserve leading zeros like "061")
        Employee Name  -> emp_name
        FOLDER_NAME    -> folder_name  (must match a folder in Dataset_Clicked/)

    Rows with empty FOLDER_NAME are skipped (employee has no face data yet).
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"employees.csv not found at {CSV_PATH}\n"
            f"Save your Excel as CSV at this path."
        )

    rows = []
    skipped_no_folder = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            # Strip whitespace from keys + values
            clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}

            # Validate required columns exist
            for col in ("Employee ID", "Employee Name", "FOLDER_NAME"):
                if col not in clean:
                    raise ValueError(
                        f"CSV missing required column: {col}\n"
                        f"Found columns: {list(clean.keys())}"
                    )

            emp_no      = clean["Employee ID"]
            emp_name    = clean["Employee Name"]
            folder_name = clean["FOLDER_NAME"]

            if not emp_no or not emp_name:
                continue   # skip empty rows

            if not folder_name:
                skipped_no_folder.append((emp_no, emp_name))
                continue   # employee has no gallery folder yet

            rows.append({
                "emp_no":      emp_no,
                "emp_name":    emp_name,
                "folder_name": folder_name,
            })

    if skipped_no_folder:
        print(f"[INFO] Skipped {len(skipped_no_folder)} employees with no FOLDER_NAME:")
        for emp_no, emp_name in skipped_no_folder:
            print(f"   {emp_no:>5}  |  {emp_name}")
        print()

    return rows


def create_schema(conn):
    """Drop old tables and create new schema."""
    conn.executescript("""
    DROP TABLE IF EXISTS attendance;
    DROP TABLE IF EXISTS sightings;
    DROP TABLE IF EXISTS employees;

    CREATE TABLE employees (
        emp_no       TEXT PRIMARY KEY,
        emp_name     TEXT NOT NULL,
        folder_name  TEXT NOT NULL UNIQUE
    );

    CREATE TABLE attendance (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_no          TEXT NOT NULL,
        date            TEXT NOT NULL,
        first_seen      TEXT NOT NULL,
        last_seen       TEXT NOT NULL,
        confidence      INTEGER DEFAULT 0,
        total_sightings INTEGER DEFAULT 1,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        UNIQUE(emp_no, date),
        FOREIGN KEY(emp_no) REFERENCES employees(emp_no)
    );

    CREATE TABLE sightings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_no          TEXT NOT NULL,
        timestamp       TEXT NOT NULL,
        confidence      INTEGER DEFAULT 0,
        ams_sent        INTEGER DEFAULT 0,
        ams_processed   INTEGER DEFAULT 0,
        ams_attempts    INTEGER DEFAULT 0,
        ams_last_error  TEXT,
        created_at      TEXT NOT NULL,
        FOREIGN KEY(emp_no) REFERENCES employees(emp_no)
    );

    CREATE INDEX idx_att_date ON attendance(date);
    CREATE INDEX idx_att_emp ON attendance(emp_no);
    CREATE INDEX idx_sight_ts ON sightings(timestamp);
    CREATE INDEX idx_sight_pending ON sightings(ams_sent, ams_processed);
    CREATE INDEX idx_emp_folder ON employees(folder_name);
    """)
    conn.commit()


def insert_employees(conn, rows):
    """Insert employees into the table."""
    inserted = 0
    skipped = []
    for row in rows:
        try:
            conn.execute(
                "INSERT INTO employees (emp_no, emp_name, folder_name) VALUES (?, ?, ?)",
                (row["emp_no"], row["emp_name"], row["folder_name"])
            )
            inserted += 1
        except sqlite3.IntegrityError as e:
            skipped.append((row, str(e)))
    conn.commit()
    return inserted, skipped


def verify_folders(rows):
    """Cross-check FOLDER_NAME column against actual Dataset_Clicked folders."""
    dataset_dir = os.path.join(_HERE, "..", "Data_Augmentation", "Dataset_Clicked")
    if not os.path.exists(dataset_dir):
        print(f"[WARN] {dataset_dir} not found — skipping folder validation")
        return

    existing = set(os.listdir(dataset_dir))
    csv_folders = {row["folder_name"] for row in rows}

    missing_in_disk = csv_folders - existing
    missing_in_csv  = existing - csv_folders

    if missing_in_disk:
        print("\n[WARN] Folders listed in CSV but NOT found on disk:")
        for f in sorted(missing_in_disk):
            print(f"   - {f!r}")

    if missing_in_csv:
        print("\n[INFO] Folders on disk but NOT in CSV (these won't be recognised):")
        for f in sorted(missing_in_csv):
            print(f"   - {f!r}")

    if not missing_in_disk and not missing_in_csv:
        print("\n[OK] All CSV folders match disk folders exactly.")


def main():
    print("=" * 60)
    print("  ATTENDANCE DB — MIGRATION")
    print("=" * 60)
    print()

    # 1. Backup
    print("[1/5] Backing up existing DB...")
    backup_existing_db()
    print()

    # 2. Load CSV
    print(f"[2/5] Reading {CSV_PATH}...")
    rows = load_csv()
    print(f"[OK] Loaded {len(rows)} employees from CSV.")
    for r in rows:
        print(f"   {r['emp_no']:>5}  |  {r['emp_name']:<30} |  folder={r['folder_name']!r}")
    print()

    # 3. Verify folder names match disk
    print("[3/5] Verifying FOLDER_NAME against Dataset_Clicked/...")
    verify_folders(rows)
    print()

    # 4. Create new schema
    print("[4/5] Creating new schema (dropping old tables)...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    create_schema(conn)
    print("[OK] Schema created: employees, attendance, sightings")
    print()

    # 5. Insert employees
    print("[5/5] Inserting employees...")
    inserted, skipped = insert_employees(conn, rows)
    print(f"[OK] Inserted {inserted} employees.")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} rows:")
        for row, err in skipped:
            print(f"   {row}  -> {err}")
    print()

    # Summary
    cur = conn.execute("SELECT COUNT(*) FROM employees")
    total = cur.fetchone()[0]
    conn.close()

    print("=" * 60)
    print(f"  DONE — {total} employees in database")
    print("=" * 60)
    print()
    print("Next: Open attendance.db in DB Browser -> Browse Data -> employees")


if __name__ == "__main__":
    main()
