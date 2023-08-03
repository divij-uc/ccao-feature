import geopandas as gpd
import pandas as pd
from config import TAX_YEAR, TOWNSHIP_LIST
import logging


def dl_cc_townships(file_loc="data/raw/shapefiles/cc_townships.parquet"):
    cc_township = gpd.read_file(
        "https://gis.cookcountyil.gov/traditional/rest/services/politicalBoundary/MapServer/3/query?where=1%3D1&outFields=*&outSR=4326&f=json"
    )
    cc_township.to_parquet(file_loc)


def dl_parcel_uni(
    township_list=TOWNSHIP_LIST,
    tax_year=TAX_YEAR,
    file_loc="data/raw/shapefiles/parcel_uni.parquet",
):
    parcel_uni = pd.read_json(
        f"https://datacatalog.cookcountyil.gov/resource/nj4t-kc8j.json?$where=township_name%20in(%27{'%27,%27'.join(township_list).replace(' ', '%20')}%27)&tax_year={tax_year}&$limit=400000"
    )
    parcel_uni = gpd.GeoDataFrame(
        parcel_uni,
        geometry=gpd.GeoSeries.from_xy(
            x=parcel_uni.x_3435, y=parcel_uni.y_3435, crs=3435
        ),
    )
    parcel_uni.loc[:, "pin"] = parcel_uni.pin.astype("str")
    parcel_uni.to_parquet(file_loc)


def dl_parcel_sales(file_loc="data/raw/tabular/parcel_sales.parquet"):
    parcel_sales = pd.read_csv(
        "/Users/divijsinha/Documents/assessor-comp/_box_data/Cook County 2023/Assessor__Archived_05-11-2022__-_Sales.csv"
    )
    parcel_sales.loc[:, "PIN14"] = (
        parcel_sales.PIN.str[0:2]
        + parcel_sales.PIN.str[3:5]
        + parcel_sales.PIN.str[6:9]
        + parcel_sales.PIN.str[10:13]
        + parcel_sales.PIN.str[14:18]
    )
    parcel_sales.loc[:, "recorded_date_dt"] = pd.to_datetime(
        parcel_sales.loc[:, "Recorded date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    parcel_sales.to_parquet(file_loc)


if __name__ == "__main__":
    print("DOWNLOADING ALL (WILL OVERWRITE)")
    logging.debug("DOWNLOADING TOWNSHIPS")
    dl_cc_townships()
    logging.debug("DOWNLOADING PARCEL UNIVERSE")
    dl_parcel_uni()
    logging.debug("DOWNLOADING PARCEL SALES")
    dl_parcel_sales()
    print("FINISHED DOWNLOADING")
