# %%
import requests
import urllib3
import dotenv
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pysal.lib import weights
from sklearn import cluster
from esda.moran import Moran

dotenv.load_dotenv()
parcel_uni_sales_cbg = gpd.read_parquet("data/parcel_uni_sales_cbg.parquet")

## https://darribas.org/gds_course/content/bG/lab_G.html
## https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html
# %%
km = cluster.KMeans(n_clusters=8, random_state=4242)
km_fit = km.fit(parcel_uni_sales_cbg.iloc[:, 13:-1].fillna(-1))
parcel_uni_sales_cbg.loc[:, "clus_labels"] = km_fit.labels_
# %%

weights_cbg = weights.Rook.from_dataframe(parcel_uni_sales_cbg)
agg = cluster.AgglomerativeClustering(n_clusters=8, connectivity=weights_cbg.sparse)
agg_fit = agg.fit(parcel_uni_sales_cbg.iloc[:, 13:-1].fillna(-1))
parcel_uni_sales_cbg.loc[:, "clus_labels"] = agg_fit.labels_

# %%
parcel_uni_sales_cbg.explore("clus_labels", categorical=True)

# %%

mi_results = [
    Moran(parcel_uni_sales_cbg[variable].fillna(-1), weights_cbg)
    for variable in parcel_uni_sales_cbg.columns[13:-1]
]
mi_results = [
    (variable, res.I, res.p_sim)
    for variable, res in zip(parcel_uni_sales_cbg.columns[13:-1], mi_results)
]
# Display on table
table = pd.DataFrame(
    mi_results, columns=["Variable", "Moran's I", "P-value"]
).set_index("Variable")
table


# %%
