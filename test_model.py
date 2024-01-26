import yaml
import subprocess
import pandas as pd


def update_data(res, model_name, reg="cbg"):
    prefix = "report-model-benchmark-limited/input/"
    files = ["training_data.parquet", "assessment_data.parquet"]
    if reg == "cbg":
        global parcel_uni
        res = parcel_uni.merge(
            res, how="left", on="census_block_group_geoid", indicator=True
        )
        print(res._merge.value_counts(dropna=False, normalize=True))
    res = res.loc[:, ["meta_pin", "clus_labels"]]
    res.loc[:, "clus_labels"] = res.loc[:, "clus_labels"].fillna(-1)
    res.loc[:, "clus_labels"] = res.loc[:, "clus_labels"].astype("str")
    res = res.rename({"clus_labels": model_name}, axis=1)
    print(res.columns)
    for file in files:
        file_path = f"{prefix}/{file}"
        cur_file = pd.read_parquet(file_path)
        if model_name in cur_file.columns:
            cur_file = cur_file.drop(columns=model_name)
        upd_file = cur_file.merge(
            res, how="left", left_on="meta_pin", right_on="meta_pin", indicator=True
        )
        print(upd_file._merge.value_counts(dropna=False, normalize=True))
        upd_file = upd_file.drop(columns=["_merge"])
        upd_file.to_parquet(file_path)


def update_params(model_name):
    with open("report-model-benchmark-limited/params_new.yaml", "r") as yd:
        d = yaml.load(yd, Loader=yaml.Loader)
        try:
            with open("last_model_name.txt", "r") as lmn:
                last_model_name = lmn.readline()
        except:
            last_model_name = "meta_nbhd_code"
        d["model"]["predictor"]["all"].remove(last_model_name)
        d["model"]["predictor"]["categorical"].remove(last_model_name)
        d["model"]["predictor"]["hash_cat"].remove(last_model_name)

        d["model"]["predictor"]["all"].append(model_name)
        d["model"]["predictor"]["categorical"].append(model_name)
        d["model"]["predictor"]["hash_cat"].append(model_name)

    with open("report-model-benchmark-limited/params_new.yaml", "w") as yd:
        yaml.dump(d, yd)

    with open("last_model_name.txt", "w") as lmn:
        lmn.write(model_name)


def update_r(model_name):
    with open("report-model-benchmark-limited/test-models-limited.R", "r") as rf:
        rfl = rf.readlines()
    rfl[144] = f'    technique = "{model_name}",\n'
    with open("report-model-benchmark-limited/test-models-limited.R", "w") as rf:
        rf.writelines(rfl)


def run_r():
    subprocess.Popen(
        "Rscript --vanilla report-model-benchmark-limited/test-models-limited.R",
        shell=True,
    )


def test_model(res, model_name, cbg=True):
    update_data(res, model_name, reg=cbg)
    update_params(model_name)
    update_r(model_name)
    run_r()


def test_model_without_params(params_list, run_name):
    param_store = {}
    with open("report-model-benchmark-limited/params_new.yaml", "r") as yd:
        d = yaml.load(yd, Loader=yaml.Loader)
        for param in params_list:
            param_store[param] = []
            for predictor_type in ["all", "categorical", "hash_cat"]:
                if param in d["model"]["predictor"][predictor_type]:
                    d["model"]["predictor"][predictor_type].remove(param)
                    param_store[param].append(predictor_type)

    with open("report-model-benchmark-limited/params_new.yaml", "w") as yd:
        yaml.dump(d, yd)

    update_r(run_name)
    return param_store


def return_to_normal(params_list, param_store):
    with open("report-model-benchmark-limited/params_new.yaml", "r") as yd:
        d = yaml.load(yd, Loader=yaml.Loader)
        for param in params_list:
            for predictor_type in ["all", "categorical", "hash_cat"]:
                if predictor_type in param_store[param]:
                    d["model"]["predictor"][predictor_type].append(param)

    with open("report-model-benchmark-limited/params_new.yaml", "w") as yd:
        yaml.dump(d, yd)
