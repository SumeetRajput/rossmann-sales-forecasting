"""
Feature Engineering Module (Task 2.1)
=====================================
Custom sklearn-compatible transformer that converts raw Rossmann data
into a numeric feature matrix ready for ML.

Creates 17 new features:
  - Date features:        Year, Month, Day, DayOfWeek, WeekOfYear, Quarter
  - Position flags:       IsWeekend, IsMonthStart, IsMonthMid, IsMonthEnd,
                          IsYearStart, IsYearEnd
  - Holiday distances:    DaysToChristmas, DaysAfterNewYear, DaysToEaster
  - Promo logic:          Promo2Active
  - Competition:          CompetitionOpenMonths

Plugs into sklearn Pipeline via fit() and transform() interface.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for Rossmann feature engineering."""

    def __init__(self):
        self.le_store_type = LabelEncoder()
        self.le_assortment = LabelEncoder()
        self.le_state_hol  = LabelEncoder()

    def fit(self, X, y=None):
        """Learn category labels from training data only."""
        df = self._fill_missing(X.copy())
        self.le_store_type.fit(df["StoreType"].astype(str))
        self.le_assortment.fit(df["Assortment"].astype(str))
        self.le_state_hol.fit(df["StateHoliday"].astype(str))
        return self

    def transform(self, X, y=None):
        """Apply all engineering steps in sequence."""
        df = self._fill_missing(X.copy())
        df = self._date_features(df)
        df = self._encode_cats(df)
        df = self._promo2_flag(df)
        df = self._competition_months(df)
        df = self._drop_cols(df)
        return df

    @staticmethod
    def _fill_missing(df):
        """Fill remaining missing values with sensible defaults."""
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
            df["CompetitionDistance"].median()
        )
        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                    "Promo2SinceWeek", "Promo2SinceYear"]:
            df[col] = df[col].fillna(0)
        df["PromoInterval"] = df["PromoInterval"].fillna("")
        df["Open"]          = df["Open"].fillna(1)
        df["StateHoliday"]  = df["StateHoliday"].astype(str).replace("0", "None")
        return df

    @staticmethod
    def _date_features(df):
        """Extract 15 features from the Date column."""
        df["Year"]         = df["Date"].dt.year
        df["Month"]        = df["Date"].dt.month
        df["Day"]          = df["Date"].dt.day
        df["DayOfWeek"]    = df["Date"].dt.dayofweek
        df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
        df["Quarter"]      = df["Date"].dt.quarter
        df["IsWeekend"]    = (df["DayOfWeek"] >= 5).astype(int)
        df["IsMonthStart"] = (df["Day"] <= 5).astype(int)
        df["IsMonthMid"]   = ((df["Day"] > 5) & (df["Day"] <= 20)).astype(int)
        df["IsMonthEnd"]   = (df["Day"] > 20).astype(int)
        df["IsYearStart"]  = (df["WeekOfYear"] == 1).astype(int)
        df["IsYearEnd"]    = (df["WeekOfYear"] >= 52).astype(int)

        def days_to_xmas(dt):
            xmas = pd.Timestamp(dt.year, 12, 25)
            if dt > xmas:
                xmas = pd.Timestamp(dt.year + 1, 12, 25)
            return (xmas - dt).days

        def days_to_easter(dt):
            easter_dates = {
                2013: pd.Timestamp(2013, 3, 31),
                2014: pd.Timestamp(2014, 4, 20),
                2015: pd.Timestamp(2015, 4,  5),
                2016: pd.Timestamp(2016, 3, 27),
            }
            easter = easter_dates.get(dt.year, pd.Timestamp(dt.year, 4, 1))
            return abs((easter - dt).days)

        df["DaysToChristmas"]  = df["Date"].apply(days_to_xmas)
        df["DaysAfterNewYear"] = df["Date"].apply(
            lambda d: (d - pd.Timestamp(d.year, 1, 1)).days
        )
        df["DaysToEaster"]     = df["Date"].apply(days_to_easter)
        return df

    def _encode_cats(self, df):
        """Convert text categories to integers via LabelEncoder."""
        df["StoreType"]    = self.le_store_type.transform(df["StoreType"].astype(str))
        df["Assortment"]   = self.le_assortment.transform(df["Assortment"].astype(str))
        df["StateHoliday"] = self.le_state_hol.transform(df["StateHoliday"].astype(str))
        return df

    @staticmethod
    def _promo2_flag(df):
        """Flag whether Promo2 is currently active for this row's month."""
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }

        def is_active(row):
            if row["Promo2"] == 0 or row["PromoInterval"] == "":
                return 0
            months = [month_map.get(m.strip(), 0)
                      for m in str(row["PromoInterval"]).split(",")]
            return int(row["Month"] in months)

        df["Promo2Active"] = df.apply(is_active, axis=1)
        return df

    @staticmethod
    def _competition_months(df):
        """How many months the competitor has been open."""
        df["CompetitionOpenMonths"] = (
            (df["Year"]  - df["CompetitionOpenSinceYear"])  * 12 +
            (df["Month"] - df["CompetitionOpenSinceMonth"])
        ).clip(lower=0)
        return df

    @staticmethod
    def _drop_cols(df):
        """Remove columns no longer needed after engineering."""
        to_drop = ["Date", "PromoInterval", "CompetitionOpenSinceMonth",
                   "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]
        return df.drop(columns=[c for c in to_drop if c in df.columns])
