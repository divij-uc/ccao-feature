import geopandas as gpd
import pandas as pd
import numpy as np
import logging

il_cbg = gpd.read_file("data/raw/shapefiles/cb_2022_17_bg_500k/cb_2022_17_bg_500k.shp")
cc_township = gpd.read_parquet("data/raw/shapefiles/cc_townships.parquet")
parcel_uni = gpd.read_parquet("data/raw/shapefiles/parcel_uni.parquet")
parcel_sales = pd.read_parquet("data/raw/tabular/parcel_sales.parquet")


def map_feature_to_cbg(gs):
    return_series = il_cbg.intersects(gs.geometry).replace(
        {True: gs.NAME, False: np.nan}
    )
    return return_series


def map_township_to_cbg(township_list, cbg_df=il_cbg):
    township_list = township_list.to_crs(cbg_df.crs)
    intersection_results = (
        township_list.apply(map_feature_to_cbg, axis=1)
        .T.dropna(how="all")
        .bfill(axis=1)  ## some that intersect with multiple get the first column value
        .iloc[:, 0]
        .rename("TOWNSHIP")
    )

    cbg_df = pd.merge(
        cbg_df, intersection_results, left_index=True, right_index=True, how="right"
    )
    return cbg_df


def get_latest_parcel_sales(parcel_sales):
    latest_parcel_sales = (
        parcel_sales.sort_values(by=["pin", "sale_date"], ascending=[True, False])
        .groupby("pin")
        .first()
        .reset_index()
    )
    return latest_parcel_sales


def create_cbg_parcel_metrics(parcel_uni_sales):
    cbg_parcel = (
        parcel_uni_sales.groupby("census_block_group_geoid")
        .agg(
            {
                "sale_date": [
                    lambda x: x.isna().sum(),
                    lambda x: (x.dt.year > 2019).sum(),
                ],
                "pin": "count",
                "sale_price": [
                    "mean",
                    "median",
                    "std",
                    lambda x: np.power(10, np.mean(np.log10(x.dropna()))),
                ],
                "env_airport_noise_dnl": "mean",
                "access_cmap_walk_nta_score": "mean",
                "access_cmap_walk_total_score": "mean",
                "env_flood_fs_factor": "mean",
            }
        )
        .reset_index()
    )

    new_cols = cbg_parcel.columns.droplevel(1).to_list()
    new_cols = [
        "GEOID",
        "total sales",
        "total sales post 2019",
        "total pins",
        "mean sales price",
        "median sales price",
        "std sales price",
        "gmean sales price",
    ] + new_cols[8:]
    cbg_parcel.columns = new_cols
    cbg_parcel.loc[:, "GEOID"] = cbg_parcel.GEOID.astype("str").str[:-2]
    cbg_parcel.loc[:, "ratio sales to pins"] = (
        cbg_parcel.loc[:, "total sales"] / cbg_parcel.loc[:, "total pins"]
    )

    return cbg_parcel


if __name__ == "__main__":
    cc_township = cc_township.set_index("NAME", drop=False)
    cc_township_limited = cc_township.loc[
        cc_township.NAME.isin(["SOUTH", "NORTH", "WEST"]), :
    ]
    cbg_township = map_township_to_cbg(cc_township_limited)
    latest_parcel_sales = get_latest_parcel_sales(parcel_sales)
    parcel_uni_sales = pd.merge(parcel_uni, latest_parcel_sales, how="left", on="pin")
    cbg_parcel = create_cbg_parcel_metrics(parcel_uni_sales)
    cbg_parcel_twn = pd.merge(cbg_township, cbg_parcel, "inner", "GEOID")
    cbg_parcel_twn.to_parquet("data/in_process/shapefiles/cbg_parcel_twn.parquet")
