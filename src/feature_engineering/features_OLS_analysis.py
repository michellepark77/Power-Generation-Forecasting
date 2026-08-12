#examining if lags are statistically significant and adding temporal variables for hour and month

lag_feature_cols = [
    "wind_speed_100m",
    "wind_speed_100m_lag_12",
    "wind_speed_100m_lag_24",
    "wind_speed_100m_lag_168"
]
target = "Power_with_Curtailment"

pdf = (
    clean_lag
    .select(["site_prevailing_time", target] + lag_feature_cols + ["wind_dir_sin", "wind_dir_cos"])
    .dropna()
    .toPandas()
    .sort_values("site_prevailing_time")
    .reset_index(drop=True)
)

X = pdf[lag_feature_cols]
y = pdf[target]

X = sm.add_constant(X)
linear_regress = sm.OLS(y, X).fit()

print(linear_regress.summary())
print(X)
