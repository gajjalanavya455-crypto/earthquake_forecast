# Earthquake Forecasting Using Deep Learning

BCA capstone source code using a PyTorch LSTM to estimate the magnitude of the next event in a time-ordered earthquake catalogue.

> **Important scientific limitation:** This program does not predict the exact date, time, location, or occurrence of an earthquake. It estimates a magnitude value for the next event represented in the catalogue sequence. It is an academic experiment, not an operational warning system.

## 1. Project contents

```text
earthquake_forecasting_source/
├── data/
│   └── raw/                  # Put the downloaded CSV here
├── models/                   # Saved model and scaler (created after training)
├── results/                  # Metrics, predictions and plots (created after training)
├── src/
│   ├── __init__.py
│   ├── data.py               # Data loading, cleaning and feature engineering
│   ├── model.py              # PyTorch LSTM model
│   └── pipeline.py           # Training, evaluation, baselines and forecasting
├── main.py                  # Main program
├── requirements.txt
└── README.md
```

## 2. Dataset download

Use the USGS Earthquake Catalog: https://earthquake.usgs.gov/earthquakes/search/

Select a region and date range with your project guide. Choose a magnitude threshold suitable for the selected region, then export the results as CSV. Save the file as:

`data/raw/earthquakes.csv`

The CSV must contain these columns:

- `time`
- `latitude`
- `longitude`
- `depth`
- `mag`

USGS CSV exports normally include these fields. Do not rename columns unless necessary; if you do, update `REQUIRED` in `src/data.py`.

For a BCA experiment, record the exact search region, start/end dates, magnitude threshold, download date and number of records in your report. Do not claim a region or record count that you did not actually use.

## 3. Run in Google Colab

1. Open https://colab.research.google.com/
2. Create a new notebook.
3. Upload this ZIP and extract it, or upload the project folder to Google Drive and mount Drive.
4. Change the notebook working directory to the extracted project folder.
5. Install dependencies:

```python
!pip install -r requirements.txt
```

6. Upload your USGS CSV to `data/raw/earthquakes.csv`.
7. Train the model:

```python
!python main.py --csv data/raw/earthquakes.csv --window 20 --epochs 100
```

If the catalogue is small, try a lower magnitude threshold or a longer date range. Do not simply lower the threshold without checking whether the catalogue is complete at that magnitude.

To force CPU:

```python
!python main.py --csv data/raw/earthquakes.csv --cpu
```

## 4. Run on Windows

Install Python 3.10 or later. Open Command Prompt inside the project folder:

```bat
python -m venv eq_env
eq_env\Scripts\activate
pip install -r requirements.txt
python main.py --csv data/raw/earthquakes.csv --window 20 --epochs 100
```

## 5. Output files

After a successful run, check:

- `models/best_lstm.pt` — best model checkpoint selected using validation loss.
- `models/feature_scaler.joblib` — scaler fitted on training-period data.
- `results/metrics.json` — MAE, RMSE and R² for LSTM and two baselines.
- `results/test_predictions.csv` — actual and forecast magnitudes on the test period.
- `results/loss_curve.png` — training and validation loss.
- `results/actual_vs_predicted.png` — test actual values and LSTM estimates.
- `results/prediction_scatter.png` — predicted versus actual values.
- `results/error_histogram.png` — distribution of forecast errors.
- `results/next_event_forecast.json` — estimate for the next catalogue event.

## 6. Forecast with the saved model

After training, place an updated catalogue at the same CSV path and run:

```bash
python main.py --csv data/raw/earthquakes.csv --forecast-only
```

The updated CSV must have the same required columns and compatible preprocessing.

## 7. Methodology implemented

1. Load the earthquake catalogue and parse timestamps.
2. Remove duplicate, missing and invalid records.
3. Filter by minimum magnitude.
4. Sort events chronologically.
5. Create inter-event time, rolling mean magnitude and rolling event count.
6. Split the sequence chronologically: 70% training, 15% validation and 15% testing.
7. Fit feature scaling on training rows only.
8. Create sliding windows of previous events.
9. Train an LSTM using MSE loss and Adam optimizer.
10. Save the best checkpoint using validation loss and early stopping.
11. Evaluate the test period with MAE, RMSE and R².
12. Compare against training-mean and last-event baselines.

## 8. Important notes for your final report

- The code uses an LSTM implementation. Your report discusses both LSTM and GRU as alternatives; do not write that both were experimentally compared unless you implement and run the GRU experiment too.
- The code forecasts the next catalogue-event magnitude. It does not forecast when or where an earthquake will occur.
- Results are not available until you run the program on your selected dataset. Never invent MAE, RMSE, R², plots, screenshots or training completion.
- The test metrics may be poor. That is a valid result and should be reported honestly.
- The dataset's region, date range, completeness threshold and record count must be filled in after downloading and cleaning the actual data.

## 9. Troubleshooting

**FileNotFoundError:** Check that the CSV is at `data/raw/earthquakes.csv`, or pass the correct path using `--csv`.

**Missing required columns:** Check the CSV header. Required names are `time`, `latitude`, `longitude`, `depth`, and `mag`.

**Not enough usable events:** Download a longer date range or choose a scientifically justified lower magnitude threshold. The program needs enough records for chronological train, validation and test windows.

**CUDA error:** Run with `--cpu`.

**Poor scores:** Compare the LSTM against both baselines. A neural network is not automatically better than a simple baseline.
