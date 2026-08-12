import numpy as np
import pandas as pd

#build future weather features
def create_forecast_weather_features(forecast_df):

    forecast_df = forecast_df.copy()

    forecast_df["wind_speed_100m"] = (
        forecast_df["wind_speed_80m"]
        + forecast_df["wind_speed_120m"]
    ) / 2

    forecast_df["wind_direction_100m"] = (
        forecast_df["wind_direction_80m"]
        + forecast_df["wind_direction_120m"]
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

def pad_forecast_weather(
    forecast_df,
    decoder_steps
):

    forecast_df = forecast_df.copy()

    if len(forecast_df) >= decoder_steps:
        return forecast_df.head(decoder_steps)

    shortfall = decoder_steps - len(forecast_df)

    last_row = forecast_df.iloc[[-1]].copy()

    pad_rows = pd.concat(
        [last_row] * shortfall,
        ignore_index=True
    )

    last_time = forecast_df["timestamp"].iloc[-1]

    pad_rows["timestamp"] = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=shortfall,
        freq="h"
    )

    padded_df = pd.concat(
        [forecast_df, pad_rows],
        ignore_index=True
    )

    return padded_df

#build lag features 
def build_lag_feature(
    history_df,
    forecast_df,
    forecast_start,
    decoder_steps,
    lag_hours
):

    lag_start = forecast_start - pd.Timedelta(
        hours=lag_hours
    )

    lag_mask = (
        (history_df["timestamp"] >= lag_start)
        &
        (history_df["timestamp"] < forecast_start)
    )

    historical_lag = history_df.loc[
        lag_mask,
        "wind_speed_100m"
    ].values

    forecast_values = forecast_df[
        "wind_speed_100m"
    ].values[:decoder_steps]

    lag_feature = np.concatenate(
        [
            historical_lag,
            forecast_values
        ]
    )[:decoder_steps]

    return lag_feature

#create decoder features
def build_decoder_features(
    forecast_df,
    lag_feature,
    decoder_steps
):

    decoder_features = np.column_stack([
        forecast_df["wind_speed_100m"].values[:decoder_steps],
        lag_feature[:decoder_steps],
        forecast_df["hour_sin"].values[:decoder_steps],
        forecast_df["hour_cos"].values[:decoder_steps],
        forecast_df["month_sin"].values[:decoder_steps],
        forecast_df["month_cos"].values[:decoder_steps],
        forecast_df["wind_dir_sin"].values[:decoder_steps],
        forecast_df["wind_dir_cos"].values[:decoder_steps],
    ])

    return decoder_features

#generate 7 day forecasts
def generate_forecast(
    model,
    encoder_input,
    decoder_input,
    scaler_y
):

    forecast_scaled = model.predict(
        [encoder_input, decoder_input],
        verbose=0
    )

    forecast = scaler_y.inverse_transform(
        forecast_scaled.reshape(-1, 1)
    ).flatten()

    forecast = np.clip(
        forecast,
        0,
        None
    )

    return forecast
