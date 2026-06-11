"""
Sync Employees from CSV
========================
Safe, incremental sync of employees.csv into the `employees` table.

Does NOT touch attendance/sightings tables — your historical data is safe.

Workflow:
  1. Add a row to IT Staff.xlsx
  2. Save as CSV (overwrites employees.csv)
  3. Create the matching folder in Data_Augmentation/Dataset_Clicked/
  4. Run: python sync_employees.py
  5. Re-train the gallery so the new employee is recognised:
        .\stage3_adaface_train.bat
        (or .\stage3_train.bat for ArcFace)

What this script does:
  - INSERT new employees (emp_no not yet in DB)
  - UPDATE existing employees if emp_name or folder_name changed in CSV
  - REPORTS employees in DB but missing from CSV (does NOT auto-delete)

Usage:
    python sync_employees.py
"""

import os
import csv
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "attendance.db")
CSV_PATH = os.path.join(_HERE, "employees.csv")


def load_csv():
    rows = []
    skipped = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            emp_no      = clean.get("Employee ID", "")
            emp_name    = clean.get("Employee Name", "")
            folder_name = clean.get("FOLDER_NAME", "")
            if not emp_no or not emp_name:
                continue
            if not folder_name:
                skipped.append((emp_no, emp_name))
                continue
            rows.append({
                "emp_no":      emp_no,
                "emp_name":    emp_name,
                "folder_name": folder_name,
            })
    return rows, skipped


def verify_folders(rows):
    dataset_dir = os.path.join(_HERE, "..", "Data_Augmentation", "Dataset_Clicked")
    if not os.path.exists(dataset_dir):
        print(f"[WARN] {dataset_dir} not found")
        return
    existing = set(os.listdir(dataset_dir))
    missing = [(r["emp_no"], r["emp_name"], r["folder_name"])
               for r in rows if r["folder_name"] not in existing]
    if missing:
        print("\n[WARN] FOLDER_NAME entries in CSV with no matching folder on disk:")
        for emp_no, emp_name, folder in missing:
            print(f"   {emp_no:>5}  |  {emp_name:<30}  ->  {folder!r}")
        print("   These employees will NOT be recognised until you create the folder + photos.")


def sync(conn, rows):
    added = []
    updated = []
    unchanged = 0

    # Load existing employees
    existing = {r["emp_no"]: r for r in conn.execute(
        "SELECT emp_no, emp_name, folder_name FROM employees"
    ).fetchall()}

    csv_emp_nos = {r["emp_no"] for r in rows}

    for row in rows:
        emp_no = row["emp_no"]
        if emp_no not in existing:
            conn.execute(
                "INSERT INTO employees (emp_no, emp_name, folder_name) VALUES (?, ?, ?)",
                (emp_no, row["emp_name"], row["folder_name"])
            )
            added.append((emp_no, row["emp_name"]))
        else:
            old = existing[emp_no]
            if (old["emp_name"] != row["emp_name"] or
                    old["folder_name"] != row["folder_name"]):
                conn.execute(
                    "UPDATE employees SET emp_name = ?, folder_name = ? WHERE emp_no = ?",
                    (row["emp_name"], row["folder_name"], emp_no)
                )
                updated.append((emp_no, row["emp_name"]))
            else:
                unchanged += 1

    conn.commit()

    # Employees in DB but NOT in CSV — flag but don't auto-delete
    in_db_only = set(existing.keys()) - csv_emp_nos
    return added, updated, unchanged, in_db_only


def main():
    print("=" * 60)
    print("  EMPLOYEE SYNC — incremental")
    print("=" * 60)
    print()

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] {DB_PATH} not found. Run migrate_db.py first.")
        return

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] {CSV_PATH} not found.")
        return

    rows, skipped = load_csv()
    print(f"[OK] CSV has {len(rows)} employees with FOLDER_NAME.")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} employees with no FOLDER_NAME (no photos yet).")
    print()

    verify_folders(rows)
    print()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    added, updated, unchanged, in_db_only = sync(conn, rows)
    conn.close()

    print("[RESULT]")
    print(f"   Added:     {len(added)}")
    for emp_no, name in added:
        print(f"      + {emp_no:>5}  {name}")
    print(f"   Updated:   {len(updated)}")
    for emp_no, name in updated:
        print(f"      ~ {emp_no:>5}  {name}")
    print(f"   Unchanged: {unchanged}")

    if in_db_only:
        print(f"\n[INFO] {len(in_db_only)} employees in DB but NOT in current CSV:")
        for emp_no in sorted(in_db_only):
            print(f"      - {emp_no}")
        print("   These are kept (attendance history preserved).")
        print("   To remove an employee, delete them manually in DB Browser.")

    print()
    print("=" * 60)
    print("  DONE")
    print("=" * 60)
    if added or updated:
        print()
        print("Next: Re-train the gallery to include the new/updated employees:")
        print("   .\\stage3_adaface_train.bat    (AdaFace)")
        print("   .\\stage3_train.bat            (ArcFace)")


if __name__ == "__main__":
    main()
