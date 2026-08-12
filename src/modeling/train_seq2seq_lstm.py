#implemented encoder-decoder LSTM with different lag combinations for training the model

from sklearn.model_selection import TimeSeriesSplit

#time series split for lstm testing
timeseries_lstm = TimeSeriesSplit(
    n_splits = 3,
    gap = 168,
    test_size = 500,
    max_train_size = 8000,
)

encoder_steps = 72
decoder_steps = 168
lstm_units = 32
batch_size = 128

target = "Power_with_Curtailment"
lag_combos = {
"double_lag_12" :[
    "wind_speed_100m",
    "wind_speed_100m_lag_12",
    "hour_sin", 
    "hour_cos",
    "month_sin", 
    "month_cos",
    "wind_dir_sin",
    "wind_dir_cos"],
"double_lag_24": [
    "wind_speed_100m",
    "wind_speed_100m_lag_24",
    "hour_sin", 
    "hour_cos",
    "month_sin", 
    "month_cos",
    "wind_dir_sin",
    "wind_dir_cos"],
"double_lag_168":[
    "wind_speed_100m",
    "wind_speed_100m_lag_168",
    "hour_sin", 
    "hour_cos",
    "month_sin", 
    "month_cos",
    "wind_dir_sin",
    "wind_dir_cos"],
"lag_medium_1": [
    "wind_speed_100m",
    "wind_speed_100m_lag_24",
    "wind_speed_100m_lag_168",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "wind_dir_sin",
    "wind_dir_cos"]
}

#what time frames the model will train and evaluate
eval_weeks = {
    "dec_23_29": ("2025-12-23", "2025-12-30"),
    "aug_16_22": ("2024-08-14", "2024-08-23"),
    "jul_15_22": ("2026-07-15", "2026-07-22")
}

for name, (start, end) in eval_weeks.items():
    encoder_start = pd.Timestamp(start) - pd.Timedelta(hours=encoder_steps)
    available = pdf[(pdf["site_prevailing_time"] >= str(encoder_start)) & (pdf["site_prevailing_time"] < end)]
    print(f"  {name}: {len(available)} rows available (need {encoder_steps + decoder_steps})")

#parallel compute function
def train_and_evaluate(lag_name, predictive_cols, eval_name, eval_start, eval_end):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    feature_vals = scaler_X.fit_transform(pdf[predictive_cols].values)
    target_vals  = scaler_y.fit_transform(pdf[[target]].values).flatten()

    X_seq            = []
    y_seq            = []
    decoder_feat_seq = []

    for i in range(encoder_steps, len(feature_vals) - decoder_steps): #iterates through all the features at each timestamp
        X_seq.append(feature_vals[i - encoder_steps: i])
        y_seq.append(target_vals[i: i + decoder_steps])
        decoder_feat_seq.append(feature_vals[i : i + decoder_steps])

    X_seq            = np.array(X_seq)
    y_seq            = np.array(y_seq)
    decoder_feat_seq = np.array(decoder_feat_seq)

    # --- lstm model
    decoder_shape = len(predictive_cols)

    rmse_values = []
    mse_values = []
    mae_values = []

    for fold, (train_idx, test_idx) in enumerate(timeseries_lstm.split(X_seq), 1):
        encoder_inputs = Input(shape=(encoder_steps, decoder_shape))

        encoder_lstm = layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            return_state=True
        )

        _, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(decoder_steps, decoder_shape))

        decoder_lstm = layers.LSTM(
            units=lstm_units,
            return_sequences=True
        )

        decoder_outputs = decoder_lstm(
            decoder_inputs,
            initial_state=encoder_states
        )

        decoder_dense = layers.Dense(1)

        dense_out = decoder_dense(decoder_outputs)

        output = layers.Reshape((decoder_steps,))(dense_out)

        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=output
        )

        model.compile(optimizer="adam", loss="mse")

        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_seq[train_idx], y_seq[test_idx]
        dec_train = decoder_feat_seq[train_idx]
        dec_test  = decoder_feat_seq[test_idx]

        model.fit(
            [X_train, dec_train], y_train,
            epochs=20, batch_size=batch_size, validation_split=0.2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
            verbose=0
        )

        y_pred = model.predict([X_test, dec_test], verbose=0)
        y_pred_mw = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        y_test_mw = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        rmse = np.sqrt(np.mean((y_pred_mw - y_test_mw) ** 2))
        mse = np.mean((y_pred_mw - y_test_mw) ** 2)
        mae = np.mean(np.abs(y_test_mw - y_pred_mw))
        rmse_values.append(rmse)
        mse_values.append(mse)
        mae_values.append(mae)

    print(f"[{lag_name} | {eval_name}] CV — Avg RMSE: {np.mean(rmse_values):.3f} | MSE: {np.mean(mse_values):.3f} | MAE: {np.mean(mae_values):.3f}")

    # input timeframes here
    eval_start_ts = pd.Timestamp(eval_start)
    encoder_start_ts = eval_start_ts - pd.Timedelta(hours=encoder_steps)

    enc_mask = (pdf["site_prevailing_time"] >= encoder_start_ts) & (pdf["site_prevailing_time"] < eval_start_ts)
    dec_mask = (pdf["site_prevailing_time"] >= eval_start_ts) & (pdf["site_prevailing_time"] < eval_end)

    enc_data = pdf.loc[enc_mask, predictive_cols].values[-encoder_steps:]
    dec_data = pdf.loc[dec_mask, predictive_cols].values[:decoder_steps]
    actual = pdf.loc[dec_mask, target].values[:decoder_steps]

    assert len(enc_data) == encoder_steps, f"Need {encoder_steps} encoder rows, got {len(enc_data)}"
    assert len(dec_data) == decoder_steps, f"Need {decoder_steps} decoder rows, got {len(dec_data)}"

    # Scale the encoder and decoder
    enc_scaled = scaler_X.transform(enc_data)[np.newaxis]
    dec_scaled = scaler_X.transform(dec_data)[np.newaxis]

    # Predict
    forecast_scaled = model.predict([enc_scaled, dec_scaled], verbose=0)
    forecast_mw = scaler_y.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    # Evaluation metrics for target week
    eval_rmse = np.sqrt(np.mean((forecast_mw - actual) ** 2))
    eval_mse = np.mean((forecast_mw - actual) ** 2)
    eval_mae = np.mean(np.abs(forecast_mw - actual))
    print(f"[{lag_name} | {eval_name}] Eval — RMSE: {eval_rmse:.3f} | MSE: {eval_mse:.3f} | MAE: {eval_mae:.3f}")

    return {
        "lag_name": lag_name,
        "eval_week": eval_name,
        "cv_rmse": np.mean(rmse_values),
        "cv_mse": np.mean(mse_values),
        "cv_mae": np.mean(mae_values),
        "eval_rmse": eval_rmse,
        "eval_mse": eval_mse,
        "eval_mae": eval_mae,
        "forecast": forecast_mw,
        "actual": actual
    }
    
#train model through all iterations of time frames and lag combinations, outputting error metrics
available_cols = set(pdf.columns)
valid_combos = {k: v for k, v in lag_combos.items() if all(c in available_cols for c in v)}
skipped = set(lag_combos.keys()) - set(valid_combos.keys())
if skipped:
    print(f"Skipping combos with missing features: {skipped}")

futures = []

with ThreadPoolExecutor(max_workers=3) as executor:
    for eval_name, (eval_start, eval_end) in eval_weeks.items():
        for lag_name, predictive_cols in valid_combos.items():
            future = executor.submit(
                train_and_evaluate, lag_name, predictive_cols, eval_name, eval_start, eval_end
            )
            futures.append(future)

results = [f.result() for f in futures]

# Summary table
print(f"{'Lag Combo':<18} {'Eval Week':<12} {'CV RMSE':>8} {'Eval RMSE':>10} {'Eval MSE':>9} {'Eval MAE':>9}")
print("-" * 70)
for r in results:
    print(f"{r['lag_name']:<18} {r['eval_week']:<12} {r['cv_rmse']:>8.3f} {r['eval_rmse']:>10.3f} {r['eval_mse']:>9.3f} {r['eval_mae']:>9.3f}")
