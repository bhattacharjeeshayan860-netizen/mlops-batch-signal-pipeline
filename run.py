import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping/dict.")

    required = {"seed", "window", "version"}
    missing = required - set(cfg.keys())
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    if not isinstance(cfg["seed"], int):
        raise ValueError("Config 'seed' must be an integer.")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError("Config 'window' must be a positive integer.")
    if not isinstance(cfg["version"], str):
        raise ValueError("Config 'version' must be a string.")

    logger.info(f"Config loaded — seed={cfg['seed']}, window={cfg['window']}, version={cfg['version']}")
    return cfg


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(f"Required column 'close' not found. Columns present: {list(df.columns)}")

    logger.info(f"Dataset loaded — {len(df)} rows, columns: {list(df.columns)}")
    return df


def compute_signals(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean with window={window}")
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()

    # First window-1 rows will have NaN rolling_mean — exclude from signal computation
    valid_mask = df["rolling_mean"].notna()
    df["signal"] = np.nan
    df.loc[valid_mask, "signal"] = (df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]).astype(int)

    logger.info(f"Rolling mean computed — {valid_mask.sum()} valid rows, {(~valid_mask).sum()} warm-up rows excluded from signal")
    logger.info("Signal generation complete (signal=1 if close > rolling_mean, else 0)")
    return df


def main():
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline")
    parser.add_argument("--input",   required=True, help="Path to input data.csv")
    parser.add_argument("--config",  required=True, help="Path to config.yaml")
    parser.add_argument("--output",  required=True, help="Path to output metrics.json")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path to run.log")
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    start_time = time.time()
    logger.info("=== Job started ===")

    version = "unknown"

    try:
        # 1) Config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]

        np.random.seed(seed)
        logger.info(f"NumPy random seed set to {seed}")

        # 2) Dataset
        df = load_dataset(args.input, logger)

        # 3 & 4) Rolling mean + signal
        df = compute_signals(df, window, logger)

        # 5) Metrics
        rows_processed = len(df)
        valid_signals  = df["signal"].dropna()
        signal_rate    = round(float(valid_signals.mean()), 4)
        latency_ms     = int((time.time() - start_time) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          signal_rate,
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        write_metrics(args.output, metrics)

        logger.info(f"Metrics — rows_processed={rows_processed}, signal_rate={signal_rate}, latency_ms={latency_ms}")
        logger.info("=== Job completed successfully ===")

        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Job failed: {exc}", exc_info=True)
        logger.info("=== Job ended with error ===")

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
        }
        try:
            write_metrics(args.output, error_metrics)
        except Exception as write_err:
            logger.error(f"Could not write error metrics: {write_err}")

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
