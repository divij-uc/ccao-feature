from spreg.skater_reg import Skater_reg
from sklearn import cluster, metrics
import spopt
from statsmodels.regression import linear_model
import numpy as np
import pandas as pd
import config
from matplotlib import pyplot as plt

spanning_forest_kwds = dict(
    dissimilarity=metrics.pairwise.euclidean_distances,
    affinity=None,
    reduction=np.sum,
    center=np.mean,
    verbose=1,
)


def save_res(res, res_file_name):
    res = res.loc[:, ["geometry", "clus_labels"]]
    l = res.clus_labels.unique().shape[0]
    even_space = np.linspace(0, 100, l)
    np.random.shuffle(even_space)
    res.loc[:, "clus_labels_cont"] = res.clus_labels.map(
        {k: v for k, v in zip(np.arange(l), even_space)}
    )
    res.to_file(config.OUTPUT_FILE, layer=res_file_name, driver="GPKG")


def cluster_kmeans(res_shap, layer_name=None):
    km = cluster.KMeans(
        n_clusters=config.KMEANS_NCLUSTERS,
        n_init=config.KMEANS_NINIT,
        max_iter=config.KMEANS_MAXITER,
        random_state=4242,
    )
    km_fit = km.fit(res_shap)
    res_shap.loc[:, "clus_labels"] = km_fit.labels_

    res = pd.merge(
        il_cbg,
        res_shap.reset_index(),
        how="right",
        left_on="GEOID",
        right_on="census_block_group_geoid",
        validate="1:1",
    )

    res_shap = res_shap.drop(columns="clus_labels")
    if layer_name:
        save_res(res, layer_name)
    return res


def cluster_agg(res_shap_geo, weights_res_shap, layer_name=None):
    agg = cluster.AgglomerativeClustering(
        n_clusters=config.AGG_NCLUSTERS,
        metric=config.AGG_METRIC,
        compute_full_tree=config.AGG_COMPUTEFULLTREE,
        linkage=config.AGG_LINKAGE,
        connectivity=weights_res_shap.sparse,
    )
    agg_fit = agg.fit(res_shap_geo.drop(columns="geometry"))
    res = res_shap_geo.copy(True)
    res.loc[:, "clus_labels"] = agg_fit.labels_

    if layer_name:
        save_res(res, layer_name)
    return res


def cluster_skater(res_shap_geo, weights_res_shap, layer_name):
    skater = spopt.region.Skater(
        res_shap_geo,
        weights_res_shap,
        attrs_name=res_shap_geo.columns[res_shap_geo.columns != "geometry"],
        n_clusters=config.SKATERREG_NCLUSTERS,
        floor=config.SKATERREG_FLOOR,
        trace=False,
        islands=config.SKATERREG_ISLANDS,
        spanning_forest_kwds=spanning_forest_kwds,
    )
    skater.solve()

    res = res_shap_geo.copy(True)
    res.loc[:, "clus_labels"] = skater.labels_

    if layer_name:
        save_res(res, layer_name)
    return res


def cluster_skater_reg(
    res_shap_geo,
    res_sale_y,
    weights_res_shap,
    layer_name,
    plot_ssr=False,
    nclusters=None,
):
    skater_reg = Skater_reg(
        dissimilarity=spanning_forest_kwds["dissimilarity"],
        reduction=spanning_forest_kwds["reduction"],
        center=spanning_forest_kwds["center"],
    )

    if nclusters is None:
        nclusters = config.SKATERREG_NCLUSTERS

    if "pin" in res_shap_geo.columns:
        x = res_shap_geo.drop(columns=["pin", "geometry"]).to_numpy()
    elif "meta_pin" in res_shap_geo.columns:
        x = res_shap_geo.drop(columns=["meta_pin", "geometry"]).to_numpy()
    else:
        x = res_shap_geo.drop(columns="geometry").to_numpy()

    y = res_sale_y.to_numpy()
    skater_reg_fitted = skater_reg.fit(
        nclusters,
        weights_res_shap,
        x,
        data_reg={
            "reg": linear_model.OLS,
            "y": y,
            "x": x,
        },
        model_family="statsmodels",
        verbose=3,
        # quorum=10
    )
    if plot_ssr:
        trace = [
            skater_reg_fitted._trace[i][1][2]
            for i in range(1, len(skater_reg_fitted._trace))
        ]
        fig, ax = plt.subplots()
        ax.plot(list(range(2, len(trace) + 2)), trace, "-o", color="black", linewidth=2)

        ax.set(xlabel="Number of clusters", ylabel="Total sum of squared residuals")
        ax.grid()

        plt.show()

    res = res_shap_geo.copy(True)
    res.loc[:, "clus_labels"] = skater_reg_fitted._trace[-1][0]

    if layer_name:
        save_res(res, layer_name)
    return res
