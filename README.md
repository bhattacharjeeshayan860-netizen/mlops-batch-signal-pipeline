# MLOps Batch Signal Pipeline

A minimal MLOps-style batch job that computes a rolling-mean crossover signal on OHLCV data.

---

## Project Structure

```
.
├── run.py           # Main pipeline script
├── config.yaml      # Job configuration
├── data.csv         # Input OHLCV dataset (10,000 rows)
├── requirements.txt # Python dependencies
├── Dockerfile       # Container definition
├── metrics.json     # Sample output from a successful run
└── run.log          # Sample log from a successful run
```

---

## Local Run

### Prerequisites
- Python 3.9+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

---

## Docker Build & Run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline, prints the final metrics JSON to stdout, and exits with code `0` on success or non-zero on failure.

---

## config.yaml fields

| Field     | Type   | Description                     |
|-----------|--------|---------------------------------|
| `seed`    | int    | NumPy random seed (reproducibility) |
| `window`  | int    | Rolling mean window size        |
| `version` | string | Pipeline version tag            |

---

## Signal Logic

- Compute a rolling mean on `close` using `window` rows.
- First `window - 1` rows produce `NaN` rolling mean and are **excluded** from signal computation.
- `signal = 1` if `close > rolling_mean`, else `signal = 0`.

---

## Example metrics.json

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 32,
  "seed": 42,
  "status": "success"
}
```

---

## Error output format

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

Metrics JSON is written in **both** success and error cases.
