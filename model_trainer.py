import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_SAVE_DIR = "trained_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def build_cnn(input_len):
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(input_len, 1)),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_feature_order(X_train_df):
    corr = X_train_df.corr().abs()
    dist = (1 - corr).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')
    order = dendrogram(Z, no_plot=True)['leaves']
    return order


def prepare_data(csv_path="parkinsons.csv"):
    df = pd.read_csv(csv_path)
    if 'name' in df.columns:
        df = df.drop(['name'], axis=1)

    X = df.drop(['status'], axis=1)
    y = df['status']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_order = get_feature_order(X_train)
    X_train_scaled_cnn = X_train_scaled[:, feature_order]
    X_test_scaled_cnn = X_test_scaled[:, feature_order]

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_scaled_cnn": X_train_scaled_cnn,
        "X_test_scaled_cnn": X_test_scaled_cnn,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_order": feature_order,
        "feature_names": feature_names,
        "X_train": X_train
    }


def train_classical_models(X_train_scaled, X_test_scaled, y_train, y_test):
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gaussian NB": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    }

    results = {}
    trained_cls = {}
    for name, model in models_dict.items():
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)
        results[name] = float(acc)
        trained_cls[name] = model

    return results, trained_cls


def train_cnn_ensemble(X_train_scaled_cnn, y_train, n_models=6, patience=10, feature_frac=0.85):
    X_train_cnn = np.expand_dims(X_train_scaled_cnn, axis=2)
    y_train_arr = y_train.values if hasattr(y_train, "values") else y_train

    n_features = X_train_cnn.shape[1]
    n_selected = max(3, int(round(n_features * feature_frac)))

    trained_models = []
    feature_indices_list = []
    val_accs = []

    for i in range(n_models):
        rng = np.random.RandomState(i)
        feat_idx = np.sort(rng.choice(n_features, size=n_selected, replace=False))
        boot_idx = rng.choice(len(X_train_cnn), size=len(X_train_cnn), replace=True)

        X_boot = X_train_cnn[boot_idx][:, feat_idx, :]
        y_boot = y_train_arr[boot_idx]

        X_boot_flat = X_boot.reshape(X_boot.shape[0], -1)
        sm = SMOTE(random_state=i)
        X_res_flat, y_res = sm.fit_resample(X_boot_flat, y_boot)
        X_res = X_res_flat.reshape(-1, n_selected, 1)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_res, y_res, test_size=0.15, random_state=i, stratify=y_res
        )

        model = build_cnn(n_selected)
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5)

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        val_probs = model.predict(X_val, verbose=0).flatten()
        val_preds = (val_probs >= 0.5).astype(int)
        val_acc = float(np.mean(val_preds == y_val))

        trained_models.append(model)
        feature_indices_list.append(feat_idx)
        val_accs.append(val_acc)

    return trained_models, feature_indices_list, val_accs


def ensemble_predict_proba(trained_models, feature_indices_list, val_accs, X_cnn):
    weights = np.array(val_accs)
    weights = weights / (weights.sum() + 1e-12)

    weighted_preds = []
    for model, feat_idx, w in zip(trained_models, feature_indices_list, weights):
        X_sub = X_cnn[:, feat_idx, :]
        p = model.predict(X_sub, verbose=0).flatten()
        weighted_preds.append(p * w)

    return np.sum(weighted_preds, axis=0)


def train_and_save_pipeline(csv_path="parkinsons.csv"):
    data = prepare_data(csv_path)
    cls_results, trained_cls = train_classical_models(
        data["X_train_scaled"], data["X_test_scaled"], data["y_train"], data["y_test"]
    )

    cnn_models, cnn_feat_idx, cnn_val_accs = train_cnn_ensemble(
        data["X_train_scaled_cnn"], data["y_train"], n_models=6
    )

    X_test_cnn = np.expand_dims(data["X_test_scaled_cnn"], axis=2)
    cnn_probs = ensemble_predict_proba(cnn_models, cnn_feat_idx, cnn_val_accs, X_test_cnn)
    cnn_acc = float(np.mean((cnn_probs >= 0.5).astype(int) == data["y_test"].values))
    cls_results["1D-CNN (Multivariate Ensemble)"] = cnn_acc

    # Save artifacts for real-time inference
    package = {
        "scaler": data["scaler"],
        "feature_order": data["feature_order"],
        "feature_names": data["feature_names"],
        "cnn_feat_idx": cnn_feat_idx,
        "cnn_val_accs": cnn_val_accs,
        "rf_model": trained_cls.get("Random Forest"),
        "xgb_model": trained_cls.get("XGBoost"),
        "svm_model": trained_cls.get("SVM"),
        "cls_results": cls_results
    }
    with open(os.path.join(MODEL_SAVE_DIR, "voice_models.pkl"), "wb") as f:
        pickle.dump(package, f)

    # Save CNN sub-models
    for i, m in enumerate(cnn_models):
        m.save(os.path.join(MODEL_SAVE_DIR, f"cnn_sub_{i}.keras"))

    return cls_results, package
