# %%
import requests
import urllib3
import dotenv
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dotenv.load_dotenv()
# %%
il_cbg = gpd.read_file("data/shapefiles/cb_2022_17_bg_500k/cb_2022_17_bg_500k.shp")
cc_township = gpd.read_file(
    "https://gis.cookcountyil.gov/traditional/rest/services/politicalBoundary/MapServer/3/query?where=1%3D1&outFields=*&outSR=4326&f=json"
)


# %%
def map_to_cbg(gs):
    return_series = il_cbg.intersects(gs.geometry).replace(
        {True: gs.NAME, False: np.nan}
    )
    return return_series


cc_township = cc_township.set_index("NAME", drop=False)
cc_township_limited = cc_township.loc[
    cc_township.NAME.isin(["SOUTH", "NORTH", "WEST"]), :
]
cc_township_limited = cc_township_limited.to_crs(il_cbg.crs)
intersection_results = (
    cc_township_limited.apply(map_to_cbg, axis=1)
    .T.dropna(how="all")
    .bfill(axis=1)
    .iloc[:, 0]
    .rename("TOWNSHIP")
)

il_cbg_limited = pd.merge(
    il_cbg, intersection_results, left_index=True, right_index=True, how="right"
)
# %%
parcel_uni = pd.read_json(
    "https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json?$where=township_name%20in(%27North%20Chicago%27,%27South%20Chicago%27,%27West%20Chicago%27)&tax_year=2021&$limit=400000"
)
parcel_uni = gpd.GeoDataFrame(
    parcel_uni,
    geometry=gpd.GeoSeries.from_xy(x=parcel_uni.x_3435, y=parcel_uni.y_3435, crs=3435),
)
parcel_uni.pin = parcel_uni.pin.astype("str")

parcel_sales = pd.read_csv(
    "/Users/divijsinha/Documents/assessor-comp/_box_data/Cook County 2023/Assessor__Archived_05-11-2022__-_Sales.csv"
)

parcel_sales.loc[:, "recorded_date_dt"] = pd.to_datetime(
    parcel_sales.loc[:, "Recorded date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
)
parcel_sales_most_recent = (
    parcel_sales.sort_values(by=["PIN", "recorded_date_dt"], ascending=[True, False])
    .groupby("PIN")
    .first()
    .reset_index()
)
parcel_sales_most_recent.loc[:, "PIN"] = (
    parcel_sales_most_recent.PIN.str[0:2]
    + parcel_sales_most_recent.PIN.str[3:5]
    + parcel_sales_most_recent.PIN.str[6:9]
    + parcel_sales_most_recent.PIN.str[10:13]
    + parcel_sales_most_recent.PIN.str[14:18]
)

parcel_uni_sales = pd.merge(
    parcel_uni, parcel_sales_most_recent, how="left", left_on="pin", right_on="PIN"
)
# %%
parcel_uni_sales.to_parquet("data/parcel_uni_sales.parquet")
# %%

parcel_uni_sales_cbg = (
    parcel_uni_sales.groupby("census_block_group_geoid")
    .agg(
        {
            "recorded_date_dt": lambda x: x.isna().sum(),
            "pin": "count",
            "Sale price": ["mean", "median", "std"],
            "env_ohare_noise_contour_no_buffer_bool": "mean",
            "env_airport_noise_dnl": "mean",
            "access_cmap_walk_nta_score": "mean",
            "access_cmap_walk_total_score": "mean",
            "env_flood_fs_factor": "mean",
        }
    )
    .reset_index()
)

new_cols = parcel_uni_sales_cbg.columns.droplevel(1).to_list()
new_cols = [
    "GEOID",
    "total sales",
    "total pins",
    "mean sales price",
    "median sales price",
    "std sales price",
] + new_cols[6:]
parcel_uni_sales_cbg.columns = new_cols

parcel_uni_sales_cbg.loc[:, "GEOID"] = parcel_uni_sales_cbg.GEOID.astype("str").str[:-2]
parcel_uni_sales_cbg = pd.merge(
    il_cbg_limited,
    parcel_uni_sales_cbg,
    how="inner",
    on="GEOID",
)
parcel_uni_sales_cbg.to_parquet("data/parcel_uni_sales_cbg.parquet")
