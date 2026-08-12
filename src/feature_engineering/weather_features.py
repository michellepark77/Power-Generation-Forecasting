import pyspark.sql.functions as F
from pyspark.sql.functions import col, when


# ============================================================
# Wind Speed Outlier Detection
# ============================================================

Q1 = ws_seasonal["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"].quantile(0.25)
Q3 = ws_seasonal["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

ws_outliers = ws_seasonal[
    (
        ws_seasonal["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"]
        < lower_bound
    )
    |
    (
        ws_seasonal["Wind_Speed_Average_Meter_Sec_Reading_10_Minute"]
        > upper_bound
    )
]


# ============================================================
# High Wind Outlier Flag
# ============================================================

df_joined = df_joined.withColumn(
    "is_high_outlier",
    when(
        col("Wind_Speed_Average_Meter_Sec_Reading_10_Minute")
        > upper_bound,
        1
    ).otherwise(0)
)

print(
    f"Total high outliers: "
    f"{df_joined.filter(col('is_high_outlier') == 1).count()}"
)


# ============================================================
# Wind Direction Binning
# ============================================================

sector_size = 360 / 16

df_joined = df_joined.withColumn(
    "Nearest Bin",
    (
        F.floor(
            (
                col("wind_direction_100m")
                + sector_size / 2
            ) / sector_size
        ) % 16
    ).cast("int")
)

directions = [
    "N", "NNE", "NE", "ENE",
    "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW",
    "W", "WNW", "NW", "NNW"
]

df_joined = df_joined.withColumn(
    "Which Direction",
    F.element_at(
        F.array(*[F.lit(d) for d in directions]),
        (col("Nearest Bin") + F.lit(1)).cast("int")
    )
)
