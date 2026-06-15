"""
Data Loading & Cleaning Module
==============================
Handles loading, merging, and cleaning Rossmann CSV files.
All operations are logged using the project logger.
"""

import os
import pandas as pd
from .logger_config import get_logger

logger = get_logger("data_loader")


def load_raw_data(data_dir="data"):
    """Load the three raw CSV files from disk."""
    logger.info(f"Loading raw CSV files from '{data_dir}/'")

    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "store": os.path.join(data_dir, "store.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }

    for name, path in paths.items():
        if not os.path.isfile(path):
            logger.error(f"Required file not found: {path}")
            raise FileNotFoundError(f"Missing: {path}")

    train = pd.read_csv(paths["train"], low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(paths["store"], low_memory=False)
    test  = pd.read_csv(paths["test"],  low_memory=False, parse_dates=["Date"])

    logger.info(f"  train: {train.shape[0]:>10,} rows x {train.shape[1]} cols")
    logger.info(f"  store: {store.shape[0]:>10,} rows x {store.shape[1]} cols")
    logger.info(f"  test : {test.shape[0]:>10,} rows x {test.shape[1]} cols")
    logger.info(f"  date range: {train['Date'].min().date()} -> {train['Date'].max().date()}")

    return train, store, test


def merge_store_info(train, store, test):
    """Merge store details into train and test using Store as the key."""
    logger.info("Merging store details (VLOOKUP-style join on Store)...")
    train = train.merge(store, on="Store", how="left")
    test  = test.merge(store,  on="Store", how="left")
    logger.info(f"  train after merge: {train.shape}")
    logger.info(f"  test  after merge: {test.shape}")
    return train, test


def clean_data(df):
    """Fill all missing values with sensible defaults."""
    df = df.copy()
    median_dist = df["CompetitionDistance"].median()

    df["CompetitionDistance"]       = df["CompetitionDistance"].fillna(median_dist)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"]  = df["CompetitionOpenSinceYear"].fillna(0)
    df["Promo2SinceWeek"]           = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"]           = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"]             = df["PromoInterval"].fillna("")
    df["Open"]                      = df["Open"].fillna(1)
    df["StateHoliday"]              = df["StateHoliday"].astype(str).replace("0", "None")

    return df


def filter_open_stores(train):
    """Keep only rows where store was open and sold something."""
    train_open = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    removed    = len(train) - len(train_open)
    logger.info(f"  Open-store rows: {len(train_open):,} ({removed:,} closed-day rows removed)")
    return train_open


def load_and_prepare(data_dir="data"):
    """Convenience: run the full data loading pipeline."""
    train, store, test = load_raw_data(data_dir)
    train, test = merge_store_info(train, store, test)

    logger.info("Cleaning data (filling missing values)...")
    train = clean_data(train)
    test  = clean_data(test)

    train_open = filter_open_stores(train)
    logger.info(f"  Missing values remaining: {train.isnull().sum().sum()}")

    return train, test, store, train_open
