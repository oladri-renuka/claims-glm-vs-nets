# Segmentation report (I5) and anomaly inspection (I6)

All numbers are measured on the held-out **test** split; clusterers were fitted on a
100K-policy sample of the training split. VAE/raw-feature metrics are means over 3 seeds.

![segmentation spread](figures/segmentation_spread.png)

## Method comparison at k = 6

| method       |   max_min_ratio |   max_min_std |   freq_cv |   deviance_reduction_pct |   silhouette |   ari |   min_exposure_share |
|:-------------|----------------:|--------------:|----------:|-------------------------:|-------------:|------:|---------------------:|
| glm_quantile |           4.793 |       nan     |     0.595 |                    4.534 |        0.460 | 1.000 |                0.110 |
| raw_kmeans   |           2.511 |         0.006 |     0.324 |                    1.556 |        0.122 | 0.998 |                0.093 |
| vae_gmm      |           2.340 |         0.179 |     0.236 |                    0.775 |        0.106 | 0.643 |                0.026 |
| vae_kmeans   |           2.302 |         0.072 |     0.313 |                    1.442 |        0.214 | 0.531 |                0.101 |

Full sweep over k = 4..10: [tables/segmentation.md](tables/segmentation.md).
`ari` is the mean pairwise adjusted Rand index across seeds (GLM quantiles are deterministic, so 1 by construction).
`deviance_reduction_pct` is the Poisson-deviance reduction from pricing each policy at its segment's observed rate.

## VAE k-means segments (k = 6, seed 0), sorted by observed frequency

|   segment |   policies |   exposure |   claims |   DrivAge |   BonusMalus |   VehAge |   VehPower |   Density |   diesel_share | top_area   | top_brand   | top_region                  |   exposure_share |   frequency |
|----------:|-----------:|-----------:|---------:|----------:|-------------:|---------:|-----------:|----------:|---------------:|:-----------|:------------|:----------------------------|-----------------:|------------:|
|         2 |      10941 |   7377.246 |      553 |    49.000 |       50.000 |   10.000 |      6.000 |   138.000 |          0.000 | C          | B1          | Centre                      |            0.206 |       0.075 |
|         0 |      15602 |   9593.541 |      735 |    46.000 |       50.000 |    7.000 |      6.000 |    85.000 |          0.888 | C          | B2          | Centre                      |            0.268 |       0.077 |
|         5 |      11988 |   5876.737 |      601 |    45.000 |       50.000 |    6.000 |      7.000 |  2642.000 |          0.823 | E          | B1          | Ile-de-France               |            0.164 |       0.102 |
|         3 |       8674 |   4680.707 |      483 |    47.000 |       50.000 |    5.000 |      5.000 |  3321.000 |          0.000 | E          | B1          | Ile-de-France               |            0.131 |       0.103 |
|         4 |      10855 |   4335.649 |      546 |    48.000 |       50.000 |    2.000 |      7.000 |   267.000 |          0.441 | C          | B12         | Provence-Alpes-Cotes-D'Azur |            0.121 |       0.126 |
|         1 |       9742 |   3982.857 |      682 |    28.000 |       90.000 |    8.000 |      6.000 |   496.000 |          0.483 | C          | B2          | Centre                      |            0.111 |       0.171 |

## GLM-frequency segments (k = 6) for comparison

|   segment |   policies |   exposure |   claims |   DrivAge |   BonusMalus |   VehAge |   VehPower |   Density |   diesel_share | top_area   | top_brand   | top_region    |   exposure_share |   frequency |
|----------:|-----------:|-----------:|---------:|----------:|-------------:|---------:|-----------:|----------:|---------------:|:-----------|:------------|:--------------|-----------------:|------------:|
|         0 |      11305 |   6980.828 |      377 |    39.000 |       50.000 |   12.000 |      6.000 |   115.000 |          0.553 | A          | B2          | Centre        |            0.195 |       0.054 |
|         1 |      11424 |   6760.367 |      449 |    46.000 |       50.000 |    8.000 |      6.000 |   214.000 |          0.524 | C          | B1          | Centre        |            0.189 |       0.066 |
|         2 |      11185 |   6536.098 |      506 |    49.000 |       50.000 |    6.000 |      6.000 |   298.000 |          0.513 | C          | B1          | Centre        |            0.182 |       0.077 |
|         3 |      11236 |   6233.479 |      598 |    49.000 |       50.000 |    5.000 |      6.000 |   578.500 |          0.462 | C          | B12         | Centre        |            0.174 |       0.096 |
|         4 |      11276 |   5380.052 |      646 |    41.000 |       72.000 |    6.000 |      6.000 |  1109.000 |          0.416 | E          | B12         | Rhone-Alpes   |            0.150 |       0.120 |
|         5 |      11376 |   3955.912 |     1024 |    41.000 |       80.000 |    0.000 |      6.000 |   775.000 |          0.470 | E          | B12         | Ile-de-France |            0.110 |       0.259 |

## Autoencoder anomalies: top 1% of test policies by reconstruction error

* Cut-off score 6.07; 1.00% of test policies exceed the training 99th-percentile threshold used by the API.
* Overlap of the top-1% sets across 3 autoencoder seeds (mean Jaccard): 0.25.
* Observed claim frequency: flagged 0.2042 vs rest 0.0995.

### Continuous features (medians)

|            |   rest_median |   top1pct_median |
|:-----------|--------------:|-----------------:|
| DrivAge    |        44.000 |           57.000 |
| VehAge     |         6.000 |            6.000 |
| VehPower   |         6.000 |            7.000 |
| BonusMalus |        50.000 |           80.000 |
| Density    |       393.000 |          473.000 |
| Exposure   |         0.490 |            0.440 |

### Over-represented categories among flagged policies (share and lift vs rest)

**VehBrand**

| VehBrand   |   False |   True |   lift |
|:-----------|--------:|-------:|-------:|
| B14        |   0.005 |  0.063 | 11.808 |
| B5         |   0.050 |  0.233 |  4.699 |
| B6         |   0.041 |  0.153 |  3.692 |
| B4         |   0.037 |  0.102 |  2.768 |

**Region**

| Region            |   False |   True |   lift |
|:------------------|--------:|-------:|-------:|
| Franche-Comte     |   0.002 |  0.021 | 11.161 |
| Champagne-Ardenne |   0.004 |  0.044 | 10.745 |
| Alsace            |   0.003 |  0.019 |  5.815 |
| Corse             |   0.006 |  0.028 |  4.368 |

**Area**

| Area   |   False |   True |   lift |
|:-------|--------:|-------:|-------:|
| F      |   0.026 |  0.046 |  1.776 |
| E      |   0.203 |  0.216 |  1.066 |
| D      |   0.224 |  0.230 |  1.024 |
| A      |   0.153 |  0.152 |  0.989 |

**VehGas**

| VehGas   |   False |   True |   lift |
|:---------|--------:|-------:|-------:|
| Regular  |   0.510 |  0.576 |  1.130 |
| Diesel   |   0.490 |  0.424 |  0.865 |

### Ten most unusual test policies

|      IDpol |   DrivAge |   VehAge |   VehPower |   BonusMalus | VehBrand   | VehGas   | Area   |   Density | Region           |   Exposure |   ClaimNb |   score |
|-----------:|----------:|---------:|-----------:|-------------:|:-----------|:---------|:-------|----------:|:-----------------|-----------:|----------:|--------:|
|   95917.00 |        80 |        5 |          4 |          115 | B2         | Regular  | C      |       155 | Centre           |       0.01 |         0 |   24.23 |
| 5107463.00 |        77 |        0 |          8 |          114 | B12        | Diesel   | A      |        19 | Rhone-Alpes      |       0.07 |         0 |   22.50 |
|  152108.00 |        48 |        3 |         15 |           50 | B5         | Regular  | E      |      3866 | Centre           |       1.00 |         0 |   20.93 |
| 2063253.00 |        83 |        1 |          6 |          112 | B10        | Diesel   | C      |       399 | Centre           |       0.72 |         2 |   20.26 |
|  100629.00 |        75 |        0 |         15 |           97 | B14        | Regular  | E      |      4348 | Aquitaine        |       0.54 |         1 |   19.61 |
| 1100630.00 |        76 |        1 |         15 |           97 | B14        | Regular  | E      |      4348 | Aquitaine        |       1.00 |         0 |   19.24 |
|  167281.00 |        55 |        8 |         15 |           50 | B5         | Regular  | E      |      3457 | Rhone-Alpes      |       0.89 |         0 |   19.06 |
| 2225541.00 |        40 |       14 |         15 |           50 | B5         | Regular  | A      |        17 | Rhone-Alpes      |       0.08 |         0 |   19.00 |
| 2287015.00 |        40 |       14 |         15 |           50 | B5         | Regular  | A      |        17 | Rhone-Alpes      |       0.39 |         0 |   19.00 |
| 1090082.00 |        81 |       12 |          5 |          101 | B2         | Diesel   | C      |       270 | Poitou-Charentes |       0.41 |         0 |   18.51 |
