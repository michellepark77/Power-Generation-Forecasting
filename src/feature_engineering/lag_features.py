#remove non-model features
df_drop = ["Actual_Power_MW_Reading_10_Minute",
           "Curtailment_Active_Flag_10_Minute",
           "Curtailment_Power_MW_Reading_10_Minute",
           "Wind_Speed_Average_Meter_Sec_Reading_10_Minute",
           "Site Meter Power", 
           "iso", 
           "site_name", 
           "cloud_cover", 
           "cloud_cover_low", 
           "cloud_cover_mid", 
           "cloud_cover_high", 
           "Power_without_Curtailment", 
           "historical_time"]
#making wind direction readable for the model
df_joined = df_joined.withColumn(
    "wind_dir_sin", F.sin(2 * np.pi * col("wind_direction_100m") / 360)
).withColumn(
    "wind_dir_cos", F.cos(2 * np.pi * col("wind_direction_100m") / 360)
)

df_model = df_joined.drop(*df_drop)
df_lag = df_model

#create the lag features
df_lag = df_lag.orderBy("site_prevailing_time")

lags = [12, 24, 168]
lag_cols = ["wind_speed_100m"]
window_spec = Window.orderBy("site_prevailing_time")

lag_exprs = {
    f"{lag_col}_lag_{lag_value}": lag(col(lag_col), lag_value).over(window_spec)
    for lag_value in lags
    for lag_col in lag_cols
}
df_lag = df_lag.withColumns(lag_exprs)
display(df_lag)

#review generated lag features
lag_preview_cols = ["site_prevailing_time"] + lag_cols + [c for c in df_lag.columns if "_lag_" in c]
display(df_lag.select(lag_preview_cols))

#upfilling null lag values
clean_lag = df_lag.na.fill(value=2.6988, subset=[
    "wind_speed_100m_lag_12",
    "wind_speed_100m_lag_24",
    "wind_speed_100m_lag_168"
])
