"""
Seq2Seq LSTM Training Pipeline

Builds and evaluates a Seq2Seq LSTM model for
168-hour wind power forecasting using historical
weather observations and forecast weather inputs.

Encoder: 3 days of history
Decoder: Forecast weather
Output: Future power generation forecast
"""

#imports

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

#forecasted weather

df_fw = spark.read.table("forecasted_weather_dataset")

df_fw = df_fw.drop(
    "unnecessary_inputs"
)

window_spec = (
    Window
    .partitionBy("site_prevailing_time")
    .orderBy(
        col("forecast_generation_timestamp").desc()
    )
)

df_fw_clean = (
    df_fw
    .withColumn(
        "row",
        row_number().over(window_spec)
    )
    .filter(col("row") == 1)
    .drop(
        "row",
        "forecast_generation_timestamp_UTC"
    )
)

#sequence generation

def create_sequences(
    feature_vals,
    target_vals,
    encoder_steps,
    decoder_steps
):

    X_seq = []
    y_seq = []
    decoder_feat_seq = []

    for i in range(
        encoder_steps,
        len(feature_vals) - decoder_steps
    ):

        X_seq.append(
            feature_vals[
                i - encoder_steps:i
            ]
        )

        y_seq.append(
            target_vals[
                i:i + decoder_steps
            ]
        )

        decoder_feat_seq.append(
            feature_vals[
                i:i + decoder_steps
            ]
        )

    return (
        np.array(X_seq),
        np.array(y_seq),
        np.array(decoder_feat_seq)
    )

#model architecture

def build_seq2seq_lstm(
    encoder_steps,
    decoder_steps,
    num_features,
    lstm_units
):

    encoder_inputs = Input(
        shape=(encoder_steps, num_features)
    )

    encoder_lstm = layers.LSTM(
        units=lstm_units,
        return_sequences=False,
        return_state=True
    )

    _, state_h, state_c = encoder_lstm(
        encoder_inputs
    )

    encoder_states = [
        state_h,
        state_c
    ]

    decoder_inputs = Input(
        shape=(decoder_steps, num_features)
    )

    decoder_lstm = layers.LSTM(
        units=lstm_units,
        return_sequences=True
    )

    decoder_outputs = decoder_lstm(
        decoder_inputs,
        initial_state=encoder_states
    )

    decoder_dense = layers.Dense(1)

    dense_out = decoder_dense(
        decoder_outputs
    )

    output = layers.Reshape(
        (decoder_steps,)
    )(dense_out)

    model = Model(
        inputs=[
            encoder_inputs,
            decoder_inputs
        ],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

#set up cross validation

timeseries_lstm = TimeSeriesSplit(
    n_splits=3,
    gap=168,
    test_size=500,
    max_train_size=8000
)

# Use previous 72 hours to forecast next 168 hours
encoder_steps = 72
decoder_steps = 168

# Hyperparameters
lstm_units = 32
batch_size = 128
epochs = 20

target = "Power_with_Curtailment"

#different lag combinations I used

lag_combos = {
    ...
}

#differnt weeks I trained on

eval_weeks = {
    ...
}

#ensuring that evaluation weeks are sufficient for model training size

for name, (start, end) in eval_weeks.items():

    encoder_start = (
        pd.Timestamp(start)
        - pd.Timedelta(hours=encoder_steps)
    )

    available = pdf[
        (
            pdf["site_prevailing_time"]
            >= str(encoder_start)
        )
        &
        (
            pdf["site_prevailing_time"]
            < end
        )
    ]

    print(
        f"{name}: {len(available)} rows available"
    )

#train and evaluate function

def train_and_evaluate(
    lag_name,
    predictive_cols,
    eval_name,
    eval_start,
    eval_end
):

    """
    Paste your full train_and_evaluate()
    function here
    """

#parallel compute
available_cols = set(
    pdf.columns
)

valid_combos = {
    k: v
    for k, v in lag_combos.items()
    if all(
        c in available_cols
        for c in v
    )
}

futures = []

with ThreadPoolExecutor(
    max_workers=3
) as executor:

    for eval_name, (
        eval_start,
        eval_end
    ) in eval_weeks.items():

        for lag_name, predictive_cols in valid_combos.items():

            future = executor.submit(
                train_and_evaluate,
                lag_name,
                predictive_cols,
                eval_name,
                eval_start,
                eval_end
            )

            futures.append(future)

results = [
    f.result()
    for f in futures
]

#result summary for each lag combo and time evaluation grouping

print(
    f"{'Lag Combo':<18}"
    f"{'Eval Week':<12}"
    f"{'CV RMSE':>8}"
    f"{'Eval RMSE':>10}"
)

for r in results:

    print(
        f"{r['lag_name']:<18}"
        f"{r['eval_week']:<12}"
        f"{r['cv_rmse']:>8.3f}"
        f"{r['eval_rmse']:>10.3f}"
    )
