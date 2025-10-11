import os
import json
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import hashlib
from tqdm import tqdm
from datetime import datetime
from scipy.stats import ttest_ind

# PARAMETERS
INPUT_DIR = "/net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files/" #Folder containing the json files
OUTPUT_CSV = "./output/split_sample/splits_v1.csv"
REPORT_MD = "./output/split_sample/split_report.md"
SEED = 42 # Fixed seed for reproducibility
VAL_SIZE = 0.3 # 30% validation split

# Step 1. Load patient microdata
records = []
print("List started")

for file in tqdm(os.listdir(INPUT_DIR), desc="Loading JSON files", unit="file"):
    if file.endswith(".json"):
        with open(os.path.join(INPUT_DIR, file), "r") as f:
            data = json.load(f)
            patient_id = data["patient_id"]
            pcr = data["primary_lesion"]["pcr"]
            subtype = data["primary_lesion"]["tumor_subtype"]
            records.append({"patient_id": patient_id, "pcr": pcr, "subtype": subtype})

df = pd.DataFrame(records)
print(f"Loaded {len(df)} patients")

# Step 2. Create stratification key 
# Combine pCR and subtype to ensure approximate balance across both
df["strat_key"] = df["pcr"].astype(str) + "_" + df["subtype"].astype(str)

# Step 3. Perform a stratified split
splitter = StratifiedShuffleSplit(
    n_splits=1, test_size=VAL_SIZE, random_state=SEED
)

train_idx, val_idx = next(splitter.split(df, df["strat_key"]))
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"

# Step 4. Save split file
# Include seed info in CSV header as comment
with open(OUTPUT_CSV, "w") as f:
    f.write(f"# Split generated with SEED={SEED}\n")
df[["patient_id", "split"]].to_csv(OUTPUT_CSV, mode="a", index=False)

# Step 5. Check balance summary
summary = (
    df.groupby(["split", "pcr", "subtype"])
    .size()
    .unstack(fill_value=0)
)
print("\nBalance summary by split, pCR, and subtype:")
print(summary)
print(f"\nSaved to {OUTPUT_CSV}")

# Step 6. Sanity checks / summary statistics
summary_counts = df.groupby("split").size()
summary_pcr = df.groupby("split")["pcr"].mean()
summary_subtype = df.groupby(["split", "subtype"]).size().unstack(fill_value=0)

print("\n=== Sanity Check: Split Summary ===")
print("Patients per split:")
print(summary_counts)
print("\npCR rate per split:")
print(summary_pcr)
print("\nSubtype distribution per split:")
print(summary_subtype)

# Step 7. Write Markdown report
with open(REPORT_MD, "w") as f:
    f.write(f"# MAMA-MIA Split Report\n")
    f.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with SEED={SEED}_\n\n")
    f.write("## Split Overview\n")
    f.write(f"- Total patients: {len(df)}\n")
    f.write(f"- Train: {summary_counts['train']} ({(summary_counts['train']/len(df))*100:.1f}%)\n")
    f.write(f"- Validation: {summary_counts['val']} ({(summary_counts['val']/len(df))*100:.1f}%)\n\n")

    f.write("## pCR Rate per Split\n")
    f.write(summary_pcr.to_markdown() + "\n\n")

    f.write("## Subtype Distribution per Split\n")
    f.write(summary_subtype.to_markdown() + "\n\n")
print(f"Wrote report → {REPORT_MD}")