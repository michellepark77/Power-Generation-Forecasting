import pandas as pd
import numpy as np


def create_forecast_weather_features(forecast_df):
    forecast_df["wind_speed_100m"] = (
        forecast_df["wind_speed_80m"] + forecast_df["wind_speed_120m"]
    ) / 2

    forecast_df["wind_direction_100m"] = (
        forecast_df["wind_direction_80m"] + forecast_df["wind_direction_120m"]
    ) / 2

    forecast_df["hour_sin"] = np.sin(
        2 * np.pi * forecast_df["timestamp"].dt.hour / 24
    )

    forecast_df["hour_cos"] = np.cos(
        2 * np.pi * forecast_df["timestamp"].dt.hour / 24
    )

    forecast_df["month_sin"] = np.sin(
        2 * np.pi * forecast_df["timestamp"].dt.month / 12
    )

    forecast_df["month_cos"] = np.cos(
        2 * np.pi * forecast_df["timestamp"].dt.month / 12
    )

    forecast_df["wind_dir_sin"] = np.sin(
        2 * np.pi * forecast_df["wind_direction_100m"] / 360
    )

    forecast_df["wind_dir_cos"] = np.cos(
        2 * np.pi * forecast_df["wind_direction_100m"] / 360
    )

    return forecast_df


def build_lag_feature(history_df, forecast_df, forecast_start, decoder_steps, lag_hours):
    lag_start = forecast_start - pd.Timedelta(hours=lag_hours)

    lag_mask = (
        (history_df["timestamp"] >= lag_start) &
        (history_df["timestamp"] < forecast_start)
    )

    historical_lag = history_df.loc[lag_mask, "wind_speed_100m"].values
    forecast_values = forecast_df["wind_speed_100m"].values[:decoder_steps]

    return np.concatenate([historical_lag, forecast_values])[:decoder_steps]


def generate_forecast(model, encoder_input, decoder_input, scaler_y):
    forecast_scaled = model.predict(
        [encoder_input, decoder_input],
        verbose=0
    )

    forecast = scaler_y.inverse_transform(
        forecast_scaled.reshape(-1, 1)
    ).flatten()

    forecast = forecast.clip(min=0)

    return forecast
