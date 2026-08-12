# Find maximum wind speed
max_ws = ws_time["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"].idxmax()
max_where = ws_time.loc[max_ws, "Wind_Speed_Average_Meter_Sec_Reading_10_Minute"]
max_when = ws_time.loc[max_ws, "site_prevailing_time"]

print(f"The max wind speed {max_where:.3f} m/s occurred on {max_when}")

# Find minimum wind speed
min_ws = ws_time["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"].idxmin()
min_where = ws_time.loc[min_ws, "Wind_Speed_Average_Meter_Sec_Reading_10_Minute"]
min_when = ws_time.loc[min_ws, "site_prevailing_time"]

print(f"The min wind speed {min_where:.3f} m/s occurred on {min_when}")

max_ws_week = (
    df_joined
    .select(
        "Wind_Speed_Average_Meter_Sec_Reading_10_Minute",
        df_nw["site_prevailing_time"]
    )
    .filter(col("site_prevailing_time").between(highest_wind_speed_week))
    .orderBy(col("site_prevailing_time"))
    .toPandas()
)

plt.figure(figsize=(14,4))
plt.plot(
    max_ws_week["site_prevailing_time"],
    max_ws_week["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"],
    linewidth=0.5
)
plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")
plt.title("Wind Speeds for the Week of Highest Wind Speed")
plt.tight_layout()
plt.show()

min_ws_week = (
    df_joined
    .select(
        "Wind_Speed_Average_Meter_Sec_Reading_10_Minute",
        df_nw["site_prevailing_time"]
    )
    .filter(col("site_prevailing_time").between(lowest_wind_speed_week))
    .orderBy(col("site_prevailing_time"))
    .toPandas()
)

plt.figure(figsize=(14,4))
plt.plot(
    min_ws_week["site_prevailing_time"],
    min_ws_week["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"],
    linewidth=0.5
)
plt.xlabel("Time")
plt.ylabel("Wind Speed (m/s)")
plt.title("Wind Speeds for the Week of Lowest Wind Speed")
plt.tight_layout()
plt.show()
