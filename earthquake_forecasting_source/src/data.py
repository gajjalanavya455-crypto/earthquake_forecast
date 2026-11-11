from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ["time", "latitude", "longitude", "depth", "mag"]

def load_catalogue(csv_path: str, min_magnitude: float = 0.0) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download a USGS CSV and place it at this path."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected USGS columns include: {REQUIRED}"
        )

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for col in ["latitude", "longitude", "depth", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED)
    df = df.drop_duplicates(subset=["time", "latitude", "longitude", "depth", "mag"])
    df = df[df["mag"] >= min_magnitude]
    df = df[(df["latitude"].between(-90, 90)) &
            (df["longitude"].between(-180, 180)) &
            (df["depth"] >= 0)]
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} usable events remain. Download a larger catalogue "
            "or lower --min-magnitude. At least 50 records are recommended."
        )
    return df

def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    delta = out["time"].diff().dt.total_seconds() / 3600.0
    out["inter_event_hours"] = delta.clip(lower=0).fillna(0)
    out["rolling_mean_mag"] = out["mag"].rolling(window=5, min_periods=1).mean()
    out["rolling_event_count"] = (
        out.set_index("time")["mag"].rolling("7D").count().to_numpy(dtype=float)
    )
    feature_cols = [
        "latitude", "longitude", "depth", "mag",
        "inter_event_hours", "rolling_mean_mag", "rolling_event_count"
    ]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    return out.reset_index(drop=True), feature_cols
