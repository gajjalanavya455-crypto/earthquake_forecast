import argparse
from src.pipeline import train, forecast_only

def build_parser():
    p = argparse.ArgumentParser(
        description="Earthquake magnitude forecasting with a PyTorch LSTM."
    )
    p.add_argument("--csv", default="data/raw/earthquakes.csv",
                   help="Path to a USGS earthquake catalogue CSV.")
    p.add_argument("--output", default=".", help="Project output directory.")
    p.add_argument("--min-magnitude", type=float, default=2.5,
                   help="Minimum event magnitude to retain.")
    p.add_argument("--window", type=int, default=20,
                   help="Number of previous events in each input sequence.")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    p.add_argument("--forecast-only", action="store_true",
                   help="Load the saved model and forecast using the latest CSV events.")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.forecast_only:
        forecast_only(args)
    else:
        train(args)
