#adjusted power here from curtailment
#filter non-negative power readings
df_joined = df_joined.withColumn(
    "Actual_Power_MW_Reading_10_Minute",
    when(col("Actual_Power_MW_Reading_10_Minute") < 0, 0)
    .otherwise(col("Actual_Power_MW_Reading_10_Minute"))
)
 
#create adjusted power column, when curtailed = actual + curtailed power
df_joined = df_joined.withColumn(
    "Actual_Power_MW_Adjusted",
    when(
        col("Curtailment_Active_Flag_10_Minute") == 1,
        col("Actual_Power_MW_Reading_10_Minute") + 
        col("Curtailment_Power_MW_Reading_10_Minute")
    ).otherwise(col("Actual_Power_MW_Reading_10_Minute"))
)
display(df_joined)

#renaming power cols
df_joined = df_joined.withColumn(
    "Power_with_Curtailment",
    col("Actual_Power_MW_Adjusted")
)

df_joined = df_joined.withColumn(
    "Power_without_Curtailment",
     col("Actual_Power_MW_Reading_10_Minute")
)

summary = df_joined.agg(
    spark_sum("Power_with_Curtailment").alias("Total power with curtailment"),
    spark_sum("Power_without_Curtailment").alias("Total power without curtailment")
)
summary_row = summary.collect()[0]
power_lost_df = summary_row["Total power with curtailment"] - summary_row["Total power without curtailment"]
display(f"Total power lost: {power_lost_df: ,.2f} MW")
