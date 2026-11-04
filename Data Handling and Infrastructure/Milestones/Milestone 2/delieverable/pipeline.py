import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

REQUIRED_FIELDS = ("ID", "Title", "Category", "Attack Type", "Scenario Description", "Attack Steps")
MITRE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def clean_row_dict(raw_row):
    """Strip spaces from column headers and normalize empty/NA strings."""
    cleaned = {}
    for k, v in raw_row.items():
        if k is None:
            continue
        key = str(k).strip()
        if key == "":
            key = "Unnamed: 15"
        val = "" if v is None else str(v).strip()
        if val.lower() in {"nan", "none", "n/a", "null"}:
            val = ""
        cleaned[key] = val
    return cleaned


def validate_row(row, seen_keys=None):
    """
    Validate a single CSV row against the data contract (Lab 5 & Lab 6).
    Returns (is_valid, reason, payload).
    """
    r = clean_row_dict(row)

    # 1. Check required fields are present
    for field in REQUIRED_FIELDS:
        if not r.get(field, ""):
            return False, f"missing:{field}", {"reason": f"missing:{field}", "raw": r}

    # 2. Check ID is a valid positive integer
    try:
        row_id = int(r["ID"])
        if row_id <= 0:
            raise ValueError()
    except ValueError:
        return False, "bad_number:ID", {"reason": "bad_number:ID", "raw": r}

    # 3. Check Scenario Description has a reasonable minimum length
    if len(r["Scenario Description"]) < 15:
        return False, "short_text:Scenario Description", {"reason": "short_text:Scenario Description", "raw": r}

    # 4. Check for truncated rows or shifted columns (Unnamed: 15 not empty or MITRE ID in Impact)
    if not r.get("Target Type") and not r.get("Vulnerability") and not r.get("MITRE Technique"):
        return False, "truncated_row", {"reason": "truncated_row", "raw": r}

    if r.get("Unnamed: 15", ""):
        return False, "column_shift:Unnamed_15", {"reason": "column_shift:Unnamed_15", "raw": r}

    if (
        MITRE_RE.search(r.get("Impact", ""))
        or MITRE_RE.search(r.get("Detection Method", ""))
        or MITRE_RE.search(r.get("Solution", ""))
    ):
        return False, "column_shift:pipe_delimiter", {"reason": "column_shift:pipe_delimiter", "raw": r}

    # 5. Check for duplicate (Title, Scenario Description)
    if seen_keys is not None:
        key = (r["Title"], r["Scenario Description"])
        if key in seen_keys:
            return False, "duplicate_scenario", {"reason": "duplicate_scenario", "raw": r}
        seen_keys.add(key)

    return True, "ok", r


def transform_good_row(r):
    """Clean category labels and compute basic numeric features for a valid row."""
    cat = r["Category"].strip()
    if cat == "Network security":
        cat = "Network Security"

    tools_str = r.get("Tools Used", "")
    tool_count = 0 if not tools_str else len([t for t in re.split(r"[,;/]", tools_str) if t.strip()])
    scen_text = r["Scenario Description"]
    steps_text = r["Attack Steps"]

    return {
        "ID": int(r["ID"]),
        "Title": r["Title"],
        "Category": cat,
        "Attack Type": r["Attack Type"],
        "Scenario Description": scen_text,
        "Tools Used": tools_str if tools_str else "Unknown",
        "Attack Steps": steps_text,
        "Target Type": r.get("Target Type") or "Unknown",
        "Vulnerability": r.get("Vulnerability") or "Unknown",
        "MITRE Technique": r.get("MITRE Technique") or "Unmapped",
        "Source": r.get("Source") or "Unknown",
        "scenario_len": len(scen_text),
        "scenario_word_count": len(scen_text.split()),
        "steps_word_count": len(steps_text.split()),
        "tool_count": tool_count,
    }


def build_preprocessor():
    """Leak-free ColumnTransformer fitted on training data only (Lecture 5)."""
    num_cols = ["scenario_len", "scenario_word_count", "steps_word_count", "tool_count"]
    cat_cols = ["Target Type"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
        ("txt", TfidfVectorizer(max_features=3000, stop_words="english"), "Scenario Description"),
    ])


def run_pipeline(input_csv, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    seen_keys = set()
    good_rows = []
    bad_rows = []

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counters["rows_read"] += 1
            is_ok, reason, payload = validate_row(row, seen_keys=seen_keys)
            if is_ok:
                counters["good_rows"] += 1
                good_rows.append(transform_good_row(payload))
            else:
                counters["bad_rows"] += 1
                counters[reason] += 1
                bad_rows.append(payload)

    assert counters["rows_read"] == counters["good_rows"] + counters["bad_rows"]

    # Write quarantined bad rows to JSONL (Lab 6)
    bad_path = out_dir / "bad_rows.jsonl"
    with open(bad_path, "w", encoding="utf-8") as bf:
        for b in bad_rows:
            bf.write(json.dumps(b) + "\n")

    df_clean = pd.DataFrame(good_rows)
    if len(df_clean) > 0:
        # Group rare categories (<10 rows) into 'Other Specialised Threats' like in Milestone 1
        vc = df_clean["Category"].value_counts()
        rare = vc[vc < 10].index
        df_clean["Target_Category"] = df_clean["Category"].replace(rare, "Other Specialised Threats")

        # Save cleaned Parquet
        df_clean.to_parquet(out_dir / "attack_features_v2.parquet", engine="pyarrow", compression="snappy", index=False)

        # Save category summary CSV
        summary = (
            df_clean.groupby("Target_Category", as_index=False)
            .agg(
                count=("ID", "count"),
                mean_words=("scenario_word_count", "mean"),
                median_words=("scenario_word_count", "median"),
                mean_tools=("tool_count", "mean"),
            )
            .sort_values("count", ascending=False)
        )
        summary.to_csv(out_dir / "category_summary.csv", index=False)

        # Stratified split (60% train, 15% dev, 15% test, 10% future holdout) if dataset is large enough
        if len(df_clean) >= 100 and df_clean["Target_Category"].value_counts().min() >= 5:
            df_pool, df_future = train_test_split(
                df_clean, test_size=0.10, stratify=df_clean["Target_Category"], random_state=42
            )
            df_train, df_temp = train_test_split(
                df_pool, test_size=(1 / 3), stratify=df_pool["Target_Category"], random_state=42
            )
            df_dev, df_test = train_test_split(
                df_temp, test_size=0.50, stratify=df_temp["Target_Category"], random_state=42
            )
            prep = build_preprocessor()
            X_train = prep.fit_transform(df_train)
            X_dev = prep.transform(df_dev)
            X_test = prep.transform(df_test)
            counters["train_rows"] = X_train.shape[0]
            counters["dev_rows"] = X_dev.shape[0]
            counters["test_rows"] = X_test.shape[0]
            counters["future_rows"] = len(df_future)

    print(f"Pipeline finished -> rows_read: {counters['rows_read']}, good_rows: {counters['good_rows']}, bad_rows: {counters['bad_rows']}")
    return dict(counters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack dataset validation and cleaning pipeline")
    parser.add_argument("--input", required=True, help="Path to raw Attack_Dataset.csv")
    parser.add_argument("--out", default="outputs", help="Output directory")
    args = parser.parse_args()
    run_pipeline(args.input, args.out)
