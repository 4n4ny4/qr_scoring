import csv
import os
import random
from collections import defaultdict

RANDOM_SEED = 42
TRAIN_FRACTION = 0.8

def main():
    random.seed(RANDOM_SEED)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_dir, "data")
    print(f"Resolved project_dir={project_dir}")
    print(f"Resolved data_dir={data_dir}")
    
    input_csv = os.path.join(data_dir, "haystack_plan.csv")
    train_csv = os.path.join(data_dir, "train_plan.csv")
    test_csv = os.path.join(data_dir, "test_plan.csv")

    print(f"Reading {input_csv} ...")
    
    # 1. Read all rows and group by filename
    rows_by_filename = defaultdict(list)
    fieldnames = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows_by_filename[row["filename"]].append(row)

    unique_filenames = list(rows_by_filename.keys())
    print(f"Found {len(unique_filenames)} unique documents (filenames)")

    # 2. Shuffle and split filenames 80/20
    random.shuffle(unique_filenames)
    split_idx = int(len(unique_filenames) * TRAIN_FRACTION)
    
    train_filenames = set(unique_filenames[:split_idx])
    test_filenames = set(unique_filenames[split_idx:])

    print(f"Split size: {len(train_filenames)} train documents, {len(test_filenames)} test documents")

    # 3. Write train and test rows
    train_rows = []
    test_rows = []

    for filename, rows in rows_by_filename.items():
        if filename in train_filenames:
            train_rows.extend(rows)
        else:
            test_rows.extend(rows)

    print(f"Writing {len(train_rows)} rows to {train_csv}")
    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    print(f"Writing {len(test_rows)} rows to {test_csv}")
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print("Data splitting complete. No overlapping filenames between train and test.")

if __name__ == "__main__":
    main()
