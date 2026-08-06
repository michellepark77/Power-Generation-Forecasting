# Load data
df_place1 = spark.read.table(...)
df_historical_weather= spark.read.table(...)

# Drop columns
df_place1 = df_place1.drop(...)
df_historical_weather = df_historical_weather.drop(...)
df_hw = df_historical_weather.withColumnRenamed("site_prevailing_time", "historical_time")
 
cols_check_place1 = [
    "Actual_Power_MW_Reading_10_Minute",
    "Curtailment_Active_Flag_10_Minute",
    "Curtailment_Power_MW_Reading_10_Minute",
    "Wind_Speed_Average_Meter_Sec_Reading_10_Minute"
]
# Null checks
df_place1.select([
  (count(when(col(c).isNull(), c)) / count("*") * 100).alias(c)
  for c in cols_check_place1
]).show()
 
# Remove nulls
df_clean = df_place1.na.drop(subset=cols_check_place1)

# Join place1 data with historical weather data
df_joined = df_clean.join(df_historical_weather, df_clean["site_prevailing_time"] == df_historical_weather["site_prevailing_time"], "inner")
df_joined.describe()
display(df_joined)
