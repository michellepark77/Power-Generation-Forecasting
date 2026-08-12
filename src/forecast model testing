import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    return {
        "rmse": rmse,
        "mae": mae
    }

#define different rmse evaluating horizons
def evaluate_by_horizon(y_true, y_pred, horizons=[24, 48, 168]):
    results = {}

    for h in horizons:
        h = min(h, len(y_true))
        rmse = np.sqrt(
            np.mean(
                (y_pred[:h] - y_true[:h]) ** 2))
        results[f"{h}h_rmse"] = rmse
    return results

#analyze residuals
def analyze_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    std = np.std(residuals)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "std_residual": std
        }
#analyze residuals by lowest generation, mid generation, and highest generation

def evaluate_by_generation_segment(y_true, y_pred):

    residuals = y_true - y_pred

    p75 = np.percentile(y_true, 75)

    peaks = residuals[y_true >= p75]
    mids = residuals[(y_true >= 1.0) & (y_true < p75)]
    lows = residuals[y_true < 1.0]

    results = {}

    if len(peaks) > 0:
        results["peak_rmse"] = np.sqrt(np.mean(peaks ** 2))

    if len(mids) > 0:
        results["mid_rmse"] = np.sqrt(np.mean(mids ** 2))

    if len(lows) > 0:
        results["low_rmse"] = np.sqrt(np.mean(lows ** 2))

    return results
``

#finding largest errors for callibration
def find_largest_errors(
    timestamps,
    y_true,
    y_pred,
    top_n=20
):

    error_df = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y_true,
        "predicted": y_pred
    })

    error_df["error"] = (
        error_df["actual"]
        - error_df["predicted"]
    )

    error_df["absolute_error"] = (
        error_df["error"].abs()
    )

    return error_df.sort_values(
        "absolute_error",
        ascending=False
    ).head(top_n)

#plotting residuals
def plot_residuals(y_true, y_pred):

    residuals = y_true - y_pred

    plt.figure(figsize=(10, 5))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.7,
        edgecolor="black"
    )

    plt.axvline(
        np.mean(residuals),
        color="red",
        linestyle="--",
        label="Bias"
    )

    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.legend()
    plt.tight_layout()
    plt.show()
def create_prediction_table(timestamps, forecast_values, actual_df=None):
    table_df = pd.DataFrame({
        "timestamp": timestamps,
        "predicted_mw": forecast_values.round(3)
    })

    if actual_df is not None:
        table_df = table_df.merge(
            actual_df,
            on="timestamp",
            how="left"
        )

        table_df["error_mw"] = (
            table_df["predicted_mw"] - table_df["actual_mw"]
        ).round(3)

    return table_df


def plot_forecast(timestamps, forecast_values, actual_df=None):
    plt.figure(figsize=(14, 5))
    plt.plot(
        timestamps,
        forecast_values,
        label="Predicted",
        linewidth=1.5
    )

    if actual_df is not None:
        plt.plot(
            actual_df["timestamp"],
            actual_df["actual_mw"],
            label="Actual",
            linewidth=1.5
        )

    plt.xlabel("Time")
    plt.ylabel("Power Generation")
    plt.title("Predicted vs Actual Power Generation")
    plt.legend()
    plt.tight_layout()
    plt.show()
