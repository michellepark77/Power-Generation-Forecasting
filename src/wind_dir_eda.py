wd_dec = (
    df_joined
    .filter(
        col("site_prevailing_time").between(
            "2025-12-23",
            "2025-12-29"
        )
    )
    .groupBy("Which Direction")
    .agg(count("*").alias("n"))
    .toPandas()
)

wd_dec["Which Direction"] = pd.Categorical(
    wd_dec["Which Direction"],
    categories=directions,
    ordered=True
)

wd_dec = wd_dec.sort_values("Which Direction")

wd_aug = (
    df_joined
    .filter(
        col("site_prevailing_time").between(
            "2024-08-16",
            "2024-08-22"
        )
    )
    .groupBy("Which Direction")
    .agg(count("*").alias("n"))
    .toPandas()
)

*d_aug["Which Direction"] = pd.Cate*orical(
    wd_aug["Which Direction"],
    categories=directions,
   *ordered=True
)

wd_aug = wd_aug.so*t_values("Which Direction")
