# Milestone 2 - Data Understanding and Pipeline Setup

**Module:** Data Handling and Infrastructure for AI  
**Repo:** https://github.com/OisinHick/AI-Hdip  

## Overview
This folder contains my work for Milestone 2, building on the dataset and storage setup from Milestone 1 (`Attack_Dataset.csv` from Kaggle: `tannubarot/cybersecurity-attack-and-defence-dataset`).

The main goal in this milestone is to run Exploratory Data Analysis (EDA) on the attack dataset, check for data quality issues (missing values, duplicates, shifted columns, and skewed text lengths), and set up a repeatable pipeline (`pipeline.py`) that quarantines bad rows instead of crashing or silently dropping them.

## Files in this folder
- `Milestone_2_Report.pdf` - My written report for Milestone 2 covering the rubric sections.
- `milestoneTwo.ipynb` - Jupyter notebook with the EDA plots, statistical checks (Lectures 4 & 5), and preprocessing pipeline.
- `pipeline.py` - Python script that validates raw CSV rows, writes quarantined rows to `outputs/bad_rows.jsonl`, and saves the cleaned dataset to Parquet (`outputs/attack_features_v2.parquet`) and `outputs/category_summary.csv`.
- `test_pipeline.py` - Unit tests (`pytest`) for the row validation and quarantine rules (based on Lab 6).
- `requirements.txt` - Python libraries needed to run the notebook and script.

## Setup and Running
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the unit tests:
```bash
pytest
```

3. Run the pipeline script (expects `Attack_Dataset.csv` downloaded from Kaggle as shown in the notebook):
```bash
python pipeline.py --input data/raw/Attack_Dataset.csv --out outputs
```
