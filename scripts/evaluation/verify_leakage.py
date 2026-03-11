import json
import os
import glob

TASKS = [
    "registrant_name",
    "headquarters_city",
    "headquarters_state",
    "incorporation_state",
    "incorporation_year",
    "employees_count_total",
    "ceo_lastname",
    "holder_record_amount",
]


def extract_filename_from_idx(idx: str) -> str:
    # idx format: <filename>_<task>_<row_index> where filename can contain underscores.
    # Strip task + row suffix robustly by checking known task suffixes.
    if not isinstance(idx, str):
        return ""
    for task in TASKS:
        marker = f"_{task}_"
        pos = idx.rfind(marker)
        if pos != -1:
            return idx[:pos]
    return ""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # scripts/evaluation -> scripts -> root
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_dir, "data")
    
    # Load detection (train) instances
    combined_detection_path = os.path.join(data_dir, "long_context_detection_optionA", "combined_detection.json")
    with open(combined_detection_path) as f:
        train_data = json.load(f)
        
    train_filenames = set()
    for inst in train_data:
        filename = extract_filename_from_idx(inst.get("idx", ""))
        if not filename:
            raise ValueError(f"Unable to parse train idx: {inst.get('idx')}")
        train_filenames.add(filename)
        
    print(f"Found {len(train_filenames)} unique filenames in train dataset (detection)")
    
    # Load niah (test) instances
    test_filenames = set()
    test_files = glob.glob(os.path.join(data_dir, "niah_input", "*_test.json"))
    for file in test_files:
        with open(file) as f:
            test_data = json.load(f)
            for inst in test_data:
                filename = extract_filename_from_idx(inst.get("idx", ""))
                if not filename:
                    raise ValueError(f"Unable to parse test idx: {inst.get('idx')}")
                test_filenames.add(filename)
                
    print(f"Found {len(test_filenames)} unique filenames in test dataset (ablation niah)")
    
    # Assert
    intersection = train_filenames.intersection(test_filenames)
    if intersection:
        print(f"FAILED: Found {len(intersection)} leaking filenames!")
        print(list(intersection)[:5])
        exit(1)
        
    print("SUCCESS: Zero data leakage confirmed. Datasets are strictly separated by SEC filing.")

if __name__ == "__main__":
    main()
