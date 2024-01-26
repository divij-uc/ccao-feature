# Spatial feature engineering for CCAO

The goal of this exercise is to engineer a new spatial feature for the Cook County 
Assessor's Office. We are trying to create "neighborhoods" of homogenous sales 
price environments.

We use the following general method - 

- Use SHAP values (for each parcel-sale) from Cook County sales data
- Cluster using one of the methods
- Turn clusters into spatial boundaries
- Asssign a cluster to every parcel in the county 
- Use CCAO benchamrking model to create benchmarks

## Cluster methods

Since we are trying to create contiguous spatial regions, we need some way to 
force contiguity even though parcels themselves are not contiguous. We use two 
separate approaches to that effect -

### Census Block Group approach

This involves using the CBG boundaries as the neighborhood units. Since all CBGs 
are contiguous, once the parcel SHAPs are mapped to CBGs, we can use CBG contiguity 
instead of parcel contiguity. For the set of SHAP values spatially intersecting 
with the CBG, we assign the median to the CBG. 

We try two approaches, one using raw SHAPs (~95 columns), another after applying 
PCA (10 columns, ~0.974 explained variance ratio)

Next, these CBGs are clustered using each of the following methods

#### K-Means

using `sklearn.cluster.KMeans`. Creates a base-line for comparison. Generally allows 
us to see if spatial patterns exist at all.

#### Agglomerative Heirarchical Clustering

Using `sklearn.cluster.AgglomerativeClustering`, spatial contiguity graph through 
`pysal.lib.weights.Rook`.

#### Skater

Using `spopt.region.Skater`, spatial contiguity graph through `pysal.lib.weights.Rook`.


Post clustering, every parcel in the county is assigned to its actual Census Block 
Group.

### Cookie Cutter Approach

The other approach involved a very different entry point -

- We create a "cookie cutter" for Cook County using highways, railways, 
parks, rivers etc. The assumption being that regions disconnected by these are 
not logically one homogenous sales environment. 
- Within each of these regions, we take all the co-ordinates of parcels with SHAPs,
and create Voronoi polygons to create contiguity within each region.
- Using `spreg.skater_reg.Skater_reg` within each region, creating a spatial 
contiguity graph through `pysal.lib.weights.Rook`, we create clusters.
- Currenly, we assign the cluster to each parcel by intersecting with the voronoi.

## Testing

After this, we trigger a run of the CCAO's `report-model-benchmark`. We only run
the lightGBM model for comparison purposes.

## Results

|    | run_id                                                   | time                       |   rmse |     mae |    mape |      rsq |     cod |     prd |       prb |      mki |
|---:|:---------------------------------------------------------|:---------------------------|-------:|--------:|--------:|---------:|--------:|--------:|----------:|---------:|
|  5 | ccao-lightgbm-3.3.5-cpu-meta_nbhd_code                   | 2023-11-14 14:58:53.840358 | 130989 | 74123.8 | 26.9748 | 0.883281 | 27.6341 | 1.14004 | -0.224793 | 0.850884 |
|  4 | ccao-lightgbm-3.3.5-cpu-clusteragg_pca_100               | 2023-11-14 07:41:59.951715 | 131049 | 74081.7 | 27.0185 | 0.882763 | 27.6838 | 1.14044 | -0.226304 | 0.850245 |\n|  9 | ccao-lightgbm-3.3.5-cpu-voronoi_res_full_1969            | 2023-12-18 20:10:13.409875 | 131056 | 74307.6 | 27.1667 | 0.882968 | 27.8384 | 1.1413  | -0.227641 | 0.850052 |
|  7 | ccao-lightgbm-3.3.5-cpu-no_nbhd                          | 2024-01-18 15:50:50.439836 | 131212 | 74087.1 | 27.0625 | 0.884051 | 27.7432 | 1.14244 | -0.231406 | 0.848332 |\n|  3 | ccao-lightgbm-3.3.5-cpu-clusterskaterreg_pca_100         | 2023-11-14 08:08:56.320445 | 131237 | 74414.9 | 27.0045 | 0.882514 | 27.7209 | 1.13898 | -0.222828 | 0.853492 |
|  1 | ccao-lightgbm-3.3.5-cpu-no_lat_lon                       | 2024-01-18 08:55:23.238860 | 131274 | 74897.9 | 27.9106 | 0.882865 | 28.5737 | 1.1503  | -0.247906 | 0.839734 |\n|  2 | ccao-lightgbm-3.3.5-cpu-no_lat_lon_nbhd                  | 2024-01-18 08:28:34.973211 | 131318 | 74566.5 | 27.9264 | 0.884194 | 28.4876 | 1.153   | -0.253454 | 0.834738 |
| 10 | ccao-lightgbm-3.3.5-cpu-clusterskaterreg_pca_250         | 2023-11-14 14:51:52.165312 | 131496 | 74310.2 | 27.2497 | 0.881957 | 27.9009 | 1.14289 | -0.230853 | 0.847477 |\n|  8 | ccao-lightgbm-3.3.5-cpu-clusteragg_pca_250               | 2023-11-14 14:15:37.478917 | 131753 | 74311.7 | 27.2873 | 0.881603 | 27.9541 | 1.14367 | -0.233318 | 0.84718  |
| 11 | ccao-lightgbm-3.3.5-cpu-voronoi_res_1969_no_lat_lon_     | 2024-01-18 16:14:38.548463 | 132343 | 74896.7 | 28.1085 | 0.88079  | 28.6676 | 1.1533  | -0.253863 | 0.835946 |\n|  0 | ccao-lightgbm-3.3.5-cpu-voronoi_res_1969_no_lat_lon_prox | 2024-01-18 16:32:24.236351 | 135349 | 76277.4 | 27.6757 | 0.876846 | 28.4374 | 1.14842 | -0.242991 | 0.843838 |
|  6 | ccao-lightgbm-3.3.5-cpu-no_nbhd_lat_lon_prox             | 2024-01-18 16:49:46.994630 | 136007 | 76552.8 | 28.1514 | 0.876102 | 28.8709 | 1.15613 | -0.261807 | 0.833641 |

## Maps
