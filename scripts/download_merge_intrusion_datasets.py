"""
Download, clean, and combine public IDS datasets into a single merged CSV.

Datasets covered:
- NSL-KDD
- UNSW-NB15
- CICIDS2017

Notes:
- These datasets do not share the same feature schema.
- This script performs a schema-union merge and adds:
  - dataset_source
  - label_text
  - label_binary
- The merged CSV is useful for exploration, meta-learning, and benchmarking.
  For production model training, prefer training per-dataset or on a carefully
  harmonized common feature space.

Example:
    python scripts/download_merge_intrusion_datasets.py

    python scripts/download_merge_intrusion_datasets.py --max-rows-per-file 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_name",
    "difficulty",
]


DATASET_URLS = {
    "nsl_kdd_train": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    "nsl_kdd_test": "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    "unsw_nb15": "https://raw.githubusercontent.com/abhinav-bhardwaj/IoT-Network-Intrusion-Detection-System-UNSW-NB15/master/datasets/UNSW_NB15.csv",
    "cicids2017_monday": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Monday-WorkingHours.pcap_ISCX.csv",
    "cicids2017_tuesday": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Tuesday-WorkingHours.pcap_ISCX.csv",
    "cicids2017_wednesday": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Wednesday-workingHours.pcap_ISCX.csv",
    "cicids2017_thursday_morning_web": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "cicids2017_thursday_afternoon_infiltration": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "cicids2017_friday_morning": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "cicids2017_friday_afternoon_portscan": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "cicids2017_friday_afternoon_ddos": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if df[col].dtype == object:
            continue
        median = df[col].median()
        if pd.isna(median):
            median = 0
        df[col] = df[col].fillna(median)

    for col in object_cols:
        df[col] = df[col].fillna("unknown")

    return df


def maybe_trim(df: pd.DataFrame, max_rows_per_file: int | None) -> pd.DataFrame:
    if max_rows_per_file is None or len(df) <= max_rows_per_file:
        return df
    return df.sample(n=max_rows_per_file, random_state=42)


def load_nsl(urls: Iterable[str], max_rows_per_file: int | None) -> pd.DataFrame:
    parts = []
    for url in urls:
        df = pd.read_csv(url, header=None, names=NSL_COLUMNS)
        df["dataset_part"] = Path(url).name
        parts.append(maybe_trim(df, max_rows_per_file))
    df = pd.concat(parts, ignore_index=True, sort=False)
    df["dataset_source"] = "NSL-KDD"
    df["label_text"] = df["attack_name"].astype(str)
    df["label_binary"] = np.where(
        df["attack_name"].astype(str).str.lower() == "normal", 0, 1
    )
    return clean_frame(df)


def load_unsw(url: str, max_rows_per_file: int | None) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = maybe_trim(df, max_rows_per_file)
    df["dataset_source"] = "UNSW-NB15"
    label_series = (
        df["attack_cat"].astype(str)
        if "attack_cat" in df.columns
        else df["label"].astype(str)
    )
    df["label_text"] = label_series
    df["label_binary"] = np.where(label_series.str.lower().isin(["normal", "0"]), 0, 1)
    return clean_frame(df)


def load_cic(url_map: dict[str, str], max_rows_per_file: int | None) -> pd.DataFrame:
    parts = []
    for name, url in url_map.items():
        if not name.startswith("cicids2017_"):
            continue
        df = pd.read_csv(url, low_memory=False)
        df = maybe_trim(df, max_rows_per_file)
        df["dataset_part"] = name
        parts.append(df)

    df = pd.concat(parts, ignore_index=True, sort=False)
    df["dataset_source"] = "CICIDS2017"
    label_col = "Label" if "Label" in df.columns else "label"
    labels = df[label_col].astype(str)
    df["label_text"] = labels
    df["label_binary"] = np.where(labels.str.upper() == "BENIGN", 0, 1)
    return clean_frame(df)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and merge intrusion detection datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where cleaned and merged CSV files will be written.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row cap per source file for faster experimentation.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Loading NSL-KDD...")
    nsl_df = load_nsl(
        [DATASET_URLS["nsl_kdd_train"], DATASET_URLS["nsl_kdd_test"]],
        max_rows_per_file=args.max_rows_per_file,
    )
    write_csv(nsl_df, output_dir / "nsl_kdd_cleaned.csv")
    print(
        f"Saved NSL-KDD cleaned CSV: {output_dir / 'nsl_kdd_cleaned.csv'} ({len(nsl_df):,} rows)"
    )

    print("Loading UNSW-NB15...")
    unsw_df = load_unsw(
        DATASET_URLS["unsw_nb15"], max_rows_per_file=args.max_rows_per_file
    )
    write_csv(unsw_df, output_dir / "unsw_nb15_cleaned.csv")
    print(
        f"Saved UNSW-NB15 cleaned CSV: {output_dir / 'unsw_nb15_cleaned.csv'} ({len(unsw_df):,} rows)"
    )

    print("Loading CICIDS2017...")
    cic_df = load_cic(DATASET_URLS, max_rows_per_file=args.max_rows_per_file)
    write_csv(cic_df, output_dir / "cicids2017_cleaned.csv")
    print(
        f"Saved CICIDS2017 cleaned CSV: {output_dir / 'cicids2017_cleaned.csv'} ({len(cic_df):,} rows)"
    )

    print("Merging all datasets with schema union...")
    merged_df = pd.concat([nsl_df, unsw_df, cic_df], ignore_index=True, sort=False)
    merged_df = clean_frame(merged_df)
    merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

    merged_path = output_dir / "intrusion_datasets_merged.csv"
    write_csv(merged_df, merged_path)
    print(
        f"Saved merged CSV: {merged_path} ({len(merged_df):,} rows, {len(merged_df.columns):,} columns)"
    )


if __name__ == "__main__":
    main()
