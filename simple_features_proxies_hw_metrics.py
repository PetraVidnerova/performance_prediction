import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from multiprocessing import Pool 


from utils import train_and_test



# TRAIN_SIZE = 1366
# DEVICE = "edgegpu" # eyeriss, edgegpu, fpga
# MODEL = "DTREE" # XGBRegressor, DTREE


def run_experiment(train_size, device, model, features_type, random_seed=42, repeat=10):

    print(f"Computing {train_size} {device} {model} {features_type}")
    
    rng = np.random.RandomState(random_seed)


    df_features = pd.read_csv("mydata/simple_features.csv", index_col=0)
    df_proxies = pd.read_csv("mydata/zero_cost_proxies.csv", index_col=0).drop(columns=["params", "flops"])
    #    df_macs = pd.read_csv("mydata/network_macs.csv", index_col=0)
    df_one_hot = pd.get_dummies(
        pd.read_csv("mydata/one_hot_features.csv", index_col=0).astype(object)
    ).astype(int)
    
    df_features_proxies = pd.merge(df_features, df_proxies, left_index=True, right_index=True)
    df_features_proxies_one_hot = pd.merge(df_features_proxies, df_one_hot, left_index=True, right_index=True)
    df_all = df_features_proxies_one_hot

    df_y = pd.read_csv("mydata/hw_energy.csv", index_col=0)


    df = pd.merge(df_all, df_y, left_index=True, right_index=True)

    # do not include macs for now
    if features_type == "only_features":
        features = list(df_features.columns)   # only features 
    elif features_type == "only_proxies":
        features = list(df_proxies.columns) + ["params", "flops"] # only proxies
    elif features_type == "only_one_hot":
        features = list(df_one_hot.columns)
    elif features_type == "features_one_hot":
        features = list(df_one_hot.columns) + list(df_features.columns)
    elif features_type == "features_one_hot_proxies":
        features = list(df_one_hot.columns) + list(df_features.columns) + list(df_proxies.columns)        
    elif features_type == "features_proxies":
        features = list(df_features.columns) + list(df_proxies.columns)  # features + proxies
    else:
        raise ValueError("unknown features type")
    
    # EXP="selected"
    # features = list(df_x.columns) + ["nwot", "synflow"] # selected

    
    energies = ["eyeriss_energy", "edgegpu_energy", "fpga_energy"]

    
    X = df[features]
    Y = df[energies]


    energy_scores = {}
    energy_kendals = {}
    energy_importances = []


    for run in range(repeat):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=rng)
        energy_score, energy_kendal, energy_feature_importances = train_and_test(X_train, Y_train, X_test, Y_test,
                                                                                 what=f"{device}_energy",
                                                                                 model_type=model,
                                                                                 random_state=rng)
   
        energy_scores[run] = energy_score
        energy_kendals[run] = energy_kendal
        energy_importances.append(energy_feature_importances)
        if run % 10 == 0:
            print(run)

    print("Done.")


    #energy_scores
    df_score = pd.DataFrame(energy_scores, index=["r2_score"]).T
    df_kendal = pd.DataFrame(energy_kendals, index=["kendaltau"]).T

    R2 = df_score.describe().loc["mean", "r2_score"]
    R2_std = df_score.describe().loc["std", "r2_score"]

    print(R2)

    kendal = df_kendal.describe().loc["mean", "kendaltau"]
    kendal_std = df_kendal.describe().loc["std", "kendaltau"]
    print(kendal)



    df_energy_importance = pd.DataFrame(energy_importances)
    importances = df_energy_importance.describe().loc[["mean", "std"]].T.sort_values(by="mean", ascending=False)
    importances.to_csv(f"{device}_{train_size}_{model}_{features_type}_feature_importances.csv")

    fig, ax = plt.subplots(figsize=(15,5))
    sns.barplot(df_energy_importance, ax=ax)
    ax.set_title(f"Feature importances - prediction {device} energy (test_size {train_size}, {model}, r2_score {R2})")
    plt.savefig(f"{device}_{train_size}_{model}_{features_type}_energy_prediction.png", bbox_inches="tight")
    #    plt.show()

    return {
        "train_size": train_size,
        "device": device,
        "model": model,
        "features_type": features_type,
        "random_seed" : random_seed,
        "repeat": repeat,
        "r2_mean": R2,
        "r2_std": R2_std,
        "kendal_mean": kendal,
        "kendal_std": kendal_std
    }
    

setup_list = [] 
for train_size in 11, 25, 55, 124, 276, 614, 1366, 3036, 6748, 15000:
    for device in "edgegpu", "eyeriss", "fpga":
        for features_type in "only_one_hot", "features_one_hot", "features_one_hot_proxies":
#        for features_type in "only_features", "only_proxies", "features_proxies":
            setup_list.append(
                (train_size, device, "XGBRegressor", features_type)
            )


print(setup_list)

def evaluate(setup):
    return run_experiment(*setup, repeat=100)

num_proc = 2

if num_proc > 1:
    pool = Pool(num_proc)
    result_list = pool.map(evaluate, setup_list)
else:
    result_list = map(evaluate, setup_list)

print(result_list)

df = pd.DataFrame(result_list)
df.to_csv("test_result.csv")


