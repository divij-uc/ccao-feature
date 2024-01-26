import pandas as pd
import geopandas as gpd
import shapely
from pysal.lib import weights


def add_sale_price(shaps_to_use, actual_sales):
    id_cols = ["meta_pin", "meta_card_num", "meta_year"]
    actual_sales.loc[:, ["meta_card_num"]] = actual_sales.meta_card_num.fillna(1)
    shaps_to_use.loc[:, ["meta_card_num"]] = shaps_to_use.meta_card_num.fillna(1)
    actual_sales_median = (
        actual_sales.groupby(id_cols, dropna=False)
        .agg({"meta_sale_price": "median"})
        .reset_index()
    )
    sm = pd.merge(
        shaps_to_use,
        actual_sales_median,
        how="outer",
        on=id_cols,
    )
    return sm


def localise_parcel_uni(shaps_to_use, parcel_uni):
    sl = (
        parcel_uni.loc[:, ["pin", "geometry"]]
        .merge(
            shaps_to_use,
            left_on="pin",
            right_on="meta_pin",
            how="inner",
            validate="1:m",
        )
        .drop(columns="pin")
    )
    return sl


def regionise_morph_regions(shaps_to_use, morph_regions):
    sr = gpd.sjoin(
        left_df=shaps_to_use,
        right_df=morph_regions,
        how="left",
        predicate="intersects",
    ).drop(columns="index_right")
    return sr


def create_voronoi(shaps_to_use, morph_regions):
    voronoi_shaps_final = None

    for morph_id, morph_region_group in shaps_to_use.groupby("morph_id"):
        morph_region_geometry = morph_regions.loc[
            morph_regions.morph_id == morph_id, "geometry"
        ]
        morph_region_geometry = morph_region_geometry.make_valid()
        morph_region_geometry = morph_region_geometry.iloc[0]
        voronoi_shaps_geoms = shapely.ops.voronoi_diagram(
            geom=shapely.MultiPoint(morph_region_group.geometry.to_list()),
            envelope=morph_region_geometry,
        )
        voronoi_shaps_geoseries = gpd.GeoSeries(
            voronoi_shaps_geoms.geoms, crs="EPSG:3435"
        )
        voronoi_shaps_geoseries = voronoi_shaps_geoseries.make_valid()
        voronoi_shaps_geoseries = voronoi_shaps_geoseries.intersection(
            morph_region_geometry
        )
        voronoi_shaps = gpd.GeoDataFrame(geometry=voronoi_shaps_geoseries)
        voronoi_shaps = gpd.sjoin(
            left_df=voronoi_shaps,
            right_df=morph_region_group,
            how="left",
            predicate="intersects",
        ).drop(columns="index_right")

        if voronoi_shaps_final is None:
            voronoi_shaps_final = voronoi_shaps
        else:
            voronoi_shaps_final = pd.concat([voronoi_shaps, voronoi_shaps_final])

    voronoi_shaps_final = voronoi_shaps_final.reset_index(drop=True)

    return voronoi_shaps_final


def create_connected_comp_labels(voronoi_shaps):
    voronoi_shaps_weights = weights.Rook.from_dataframe(
        voronoi_shaps, ids=voronoi_shaps.index.to_list()
    )

    voronoi_shaps.loc[:, ["comp_labels_"]] = voronoi_shaps_weights.component_labels
    for id, i in voronoi_shaps_weights.id2i.items():
        lhs = voronoi_shaps_weights.component_labels[i]
        rhs = voronoi_shaps.loc[id, "comp_labels_"]
        assert lhs == rhs

    return voronoi_shaps
