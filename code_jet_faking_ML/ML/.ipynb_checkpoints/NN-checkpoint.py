import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uproot, wandb, os, logging, json, random
import awkward as ak
from copy import deepcopy
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import randint, uniform

# ML
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ML model
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# # Tensorflow
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/jlai/.local/share/mamba/envs/tf-gpu"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# wandb
import wandb
from wandb.integration.keras import WandbMetricsLogger

# Config
from bdt_config import sample_dict, zbi, getVarDict
from bdt_config import Vars, Vars2, Vars3, Vars_drop, Vars_plot
from bdt_config import variables, ntuple_name, ntuple_name_BDT

# Set up plot defaults
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 12.0,8.0  # Roughly 11 cm wde by 8 cm high
mpl.rcParams['font.size'] = 14.0 # Use 14 point font
sns.set(style="whitegrid")

# Check for gpu
# torch.cuda.is_available()

path = '/home/jlai/dark_photon/code/bdt_training_full_combined.csv'
df = pd.read_csv(path)

Vars2 = [
    "VertexBDTScore", "dmet", "dphi_jj", "dphi_met_phterm", 
    "dphi_met_jetterm", "met", "metsig", "balance", "ph_eta", 
    "ph_pt", "event",  "weights",  "label", "process",
    # "ph_phi", "metsigres", "jet_central_eta"
]
Vars3 = []
Vars_drop = ['event', 'weights', 'label', 'process'] 

# === Data Perparation ===
df_Vars2 = df[Vars2].copy()  # upright variables
df_Vars3 = df[Vars3].copy()  # inversed variables
df_Vars3.replace(-999, np.nan, inplace=True)
df_Vars3_inverted = 1 / df_Vars3.replace({0: np.nan})  # Avoid division by zero

df_ml_input = pd.concat([df_Vars2, df_Vars3_inverted], axis=1)
df_ml_input.replace(-999, np.nan, inplace=True)
print("Number of event with negative weights :", np.sum(df_ml_input.weights < 0))
df_ml_input["weights"] = df_ml_input["weights"].abs() # some of the weights are negative

# Renormalize weighting for signal and bkg
sum_weights_signal = sum(df_ml_input[df_ml_input.label == 1].weights)
sum_weights_bkg = sum(df_ml_input[df_ml_input.label == 0].weights)
df_ml_input2 = df_ml_input.copy()
mask_sig = df_ml_input2.label == 1
df_ml_input2.loc[mask_sig, 'weights'] = df_ml_input.loc[mask_sig, 'weights'] * (sum_weights_bkg / sum_weights_signal)
print('signal sum weight after:', sum(df_ml_input[df_ml_input.label == 1].weights), ' bkg sum weight: ', sum(df_ml_input[df_ml_input.label == 0].weights))
print('signal sum weight after:', sum(df_ml_input2[df_ml_input2.label == 1].weights), ' bkg sum weight: ', sum(df_ml_input2[df_ml_input2.label == 0].weights))

print("Number of nan in each variable: ")
print(df_ml_input.isna().sum())

X = df_ml_input.drop(Vars_drop, axis=1)
non_zero_nan = X.isna().sum()
non_zero_nan_loc = np.where(np.array(non_zero_nan) != 0)
non_zero_nan_keys = non_zero_nan.keys()[non_zero_nan_loc]
for key in non_zero_nan_keys:
    X[key+"_is_nan"] = X[key].isnull().astype(int)

y = df_ml_input['label']
weights = df_ml_input['weights']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # mean = 0, sigma = 1
X_scaled[np.isnan(X_scaled)] = 2.5  # setting the nan values to 2.5 sigma (still within the range)

feature_names = list(X.columns)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_scaled, y, weights, test_size=0.3, stratify=y, random_state=42
)

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'final_val_roc_auc',
        'goal': 'maximize'
    },
    'parameters': {
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'batch_size': {
            'values': [512, 1024, 2048]
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.25
        },
        'layer_1': {
            'values': [64, 128, 192, 256]
        },
        'layer_2': {
            'values': [128, 192, 256, 384, 512]
        },
        'layer_3': {
            'values': [512, 768, 1024, 1536, 2048]
        },
        'layer_4': {
            'values': [128, 256, 384, 512]
        },
        'layer_5': {
            'values': [32, 64, 96, 128]
        },
        'epochs': {
            'value': 50
        }
    }
}

def compute_significance(df, score_column="bdtscore", cut=0.01):
    sig = df[(df["label"] == 1) & (df[score_column] >= cut)]["weights"].sum()
    bkg = df[(df["label"] == 0) & (df[score_column] >= cut)]["weights"].sum()
    return sig / np.sqrt(bkg) if bkg > 0 else 0, sig/df[(df["label"] == 1)]["weights"].sum()

def train():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    wandb.init(project="tf-dnn-weighted-classifier")
    config = wandb.config

    model = Sequential()
    model.add(Dense(config.layer_1, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_2, activation='relu'))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_3, activation='relu'))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_4, activation='relu'))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.layer_5, activation='relu'))
    model.add(Dropout(config.dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
        ],
        weighted_metrics=[
            tf.keras.metrics.AUC(name="weighted_auc")
        ]
    )

    early_stop = EarlyStopping(
        monitor="val_weighted_auc",
        patience=5,
        mode="max",
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test, w_test),
        sample_weight=w_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stop, WandbMetricsLogger()],
        verbose=1
    )

    probs = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, probs, sample_weight=w_test)
    fpr, tpr, _ = roc_curve(y_test, probs, sample_weight=w_test)

    wandb.log({"final_val_roc_auc": auc})
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    wandb.log({"ROC Curve": wandb.Image(plt)})
    plt.close()

    X_scaled_array = np.asarray(X_scaled)
    nn_scores = model.predict(X_scaled_array, batch_size=config.batch_size).flatten()  # shape: (n_samples,)
    
    df_tmp = df_ml_input.copy()
    df_tmp['nn_scores'] = nn_scores

    best_significance, best_acceptance, best_cut = 0, 0, 0
    for cut in np.arange(0, 0.15, 0.01):
        sig, acc = compute_significance(df_tmp, score_column="nn_scores", cut=cut)
        if sig > best_significance:
            best_significance = sig
            best_acceptance = acc
            best_cut = cut

    wandb.log({
        "Max S/sqrt(B)": best_significance,
        "Acceptance": best_acceptance,
        "nnScore cut": best_cut
    })
    
    wandb.finish()

sweep_id = wandb.sweep(sweep_config, project="tf-dnn-weighted-classifier")
wandb.agent(sweep_id, function=train, count=100) 