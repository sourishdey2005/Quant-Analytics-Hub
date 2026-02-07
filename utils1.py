import pandas as pd
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Symbol", "Date"])
    return df


# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
def engineer_features(df):
    df = df.copy()

    # Returns
    df["Return"] = df.groupby("Symbol")["Close"].pct_change()

    # Cumulative return
    df["CumReturn"] = df.groupby("Symbol")["Return"].cumsum()

    # Volatility (20D)
    df["Volatility"] = (
        df.groupby("Symbol")["Return"]
        .rolling(20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Normalized price
    df["NormPrice"] = (
        df["Close"] /
        df.groupby("Symbol")["Close"].transform("first") * 100
    )

    # Moving Averages ✅ FIX
    df["MA20"] = (
        df.groupby("Symbol")["Close"]
        .rolling(20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["MA50"] = (
        df.groupby("Symbol")["Close"]
        .rolling(50)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


# -------------------------------
# PEER METRICS
# -------------------------------
def add_peer_metrics(df):
    df = df.copy()

    market_return = df.groupby("Date")["Return"].mean()
    df = df.merge(
        market_return.rename("MarketReturn"),
        on="Date",
        how="left"
    )

    df["Alpha"] = df["Return"] - df["MarketReturn"]
    return df


# -------------------------------
# DECISION ENGINE
# -------------------------------
def decision_score(row):
    score = 0
    score += 0.4 * (row["Alpha"] if pd.notna(row["Alpha"]) else 0)
    score += 0.3 * (row["Return"] if pd.notna(row["Return"]) else 0)
    score -= 0.3 * (row["Volatility"] if pd.notna(row["Volatility"]) else 0)
    return score


def decision_label(score):
    if score > 0.02:
        return "BUY"
    elif score < -0.02:
        return "AVOID"
    return "HOLD"
