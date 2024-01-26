import numpy as np
from cluster_methods import cluster_skater_reg
from pysal.lib import weights
from config import N_CLUSTERS

voronoi_per_cluster = np.floor((421462 / N_CLUSTERS) / 50) * 50 / 2


def per_region_run(grp):
    cl, voronoi_group = grp
    voronoi_group = voronoi_group.drop(
        columns=[
            "meta_year",
            "meta_card_num",
            "meta_nbhd_code",
            "township_code",
            "morph_id",
            "label",
            "comp_labels_",
        ]
    )
    nclusters = np.ceil(voronoi_group.shape[0] / voronoi_per_cluster)
    print(f"{cl=} {voronoi_group.shape[0]=} {nclusters=}")

    voronoi_group_weights = weights.Rook.from_dataframe(voronoi_group)
    assert voronoi_group_weights.n_components == 1

    if nclusters == 1:
        voronoi_res.loc[:, ["clus_labels"]] = f"{cl}_0"
    else:
        voronoi_res = cluster_skater_reg(
            voronoi_group.drop(columns="meta_sale_price"),
            voronoi_group.loc[:, "meta_sale_price"],
            voronoi_group_weights,
            layer_name=None,
            plot_ssr=False,
            nclusters=nclusters,
        )
        voronoi_res.loc[:, ["clus_labels"]] = f"{cl}_" + voronoi_res.clus_labels.astype(
            "str"
        )
    return voronoi_res
