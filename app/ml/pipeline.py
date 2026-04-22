import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, mean_absolute_error, r2_score, silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor
from app.config import settings


class TrainPipeline:

    EXCLUDE_COLS = frozenset([
        "cliente", "primera_compra", "ultima_compra", "cluster", "segmento",
        "producto_top", "folio", "fecha", "producto", "next_purchase_date",
        "days_to_next", "recompra_7d", "recompra_14d", "recompra_30d", "recompra_60d",
    ])

    def __init__(self, job_id: str):
        self.job_id = job_id
        os.makedirs(settings.MODELS_DIR, exist_ok=True)

    def run(self, dataset_path: str, horizontes: list = [7, 14, 30, 60],
            incluir_neural: bool = True, grid_search: bool = False) -> dict:

        dataset = pd.read_parquet(dataset_path)
        feature_cols = [c for c in dataset.select_dtypes(include=[np.number]).columns
                        if c not in self.EXCLUDE_COLS]

        X = dataset[feature_cols].fillna(0)

        # Scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, os.path.join(settings.MODELS_DIR, "scaler.pkl"))

        # Clustering
        self._train_clustering(X_scaled)

        # Classification per horizon
        clf_metrics = {}
        models_trained = ["scaler", "kmeans"]

        for h in horizontes:
            target = f"recompra_{h}d"
            if target not in dataset.columns:
                continue
            y = dataset[target].values
            result = self._train_xgb_clf(X_scaled, y, h, grid_search)
            clf_metrics[f"{h}d"] = result
            models_trained.append(f"xgb_clf_{h}d")

        # Regression
        reg_metrics = {}
        if "days_to_next" in dataset.columns:
            reg_metrics = self._train_regression(X_scaled, dataset["days_to_next"].clip(0, 365).values)
            models_trained.append("xgb_reg_days")

        # Neural
        neural_metrics = None
        if incluir_neural and "recompra_30d" in dataset.columns:
            neural_metrics = self._train_neural(X_scaled, dataset["recompra_30d"].values)
            if neural_metrics:
                models_trained.append("neural_model")

        # Summary
        aucs = [m.get("auc_roc", 0) for m in clf_metrics.values()]
        best_h = max(clf_metrics, key=lambda k: clf_metrics[k].get("auc_roc", 0)) if clf_metrics else None
        summary = {
            "mejor_horizonte": best_h,
            "mejor_auc": max(aucs) if aucs else 0,
            "promedio_auc": round(np.mean(aucs), 4) if aucs else 0,
            "regresion_mae": reg_metrics.get("mae"),
            "neural_auc": neural_metrics.get("auc_roc") if neural_metrics else None,
        }

        return {
            "models_trained": models_trained,
            "metrics": {"classification": clf_metrics, "regression": reg_metrics, "neural": neural_metrics},
            "accuracy_summary": summary,
        }

    def _train_clustering(self, X, max_k=8):
        sils = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            sils.append(silhouette_score(X, km.fit_predict(X)))
        best_k = range(2, max_k + 1)[np.argmax(sils)]
        km = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(X)
        joblib.dump(km, os.path.join(settings.MODELS_DIR, "kmeans.pkl"))

    def _train_xgb_clf(self, X, y, horizon, grid_search):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=42, stratify=y)
        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        spw = neg / pos if pos > 0 else 1

        if grid_search:
            grid = GridSearchCV(
                XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
                {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.05, 0.1], "subsample": [0.8, 0.9]},
                cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc", n_jobs=-1,
            )
            grid.fit(X_tr, y_tr)
            model = grid.best_estimator_
        else:
            model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                                  random_state=42, eval_metric="logloss", use_label_encoder=False)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)
        rpt = classification_report(y_te, y_pred, output_dict=True)

        joblib.dump(model, os.path.join(settings.MODELS_DIR, f"xgb_clf_{horizon}d.pkl"))

        return {
            "auc_roc": round(roc_auc_score(y_te, y_prob), 4),
            "accuracy": round(accuracy_score(y_te, y_pred), 4),
            "precision": round(rpt["1"]["precision"], 4),
            "recall": round(rpt["1"]["recall"], 4),
            "f1_score": round(f1_score(y_te, y_pred), 4),
        }

    def _train_regression(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=42)
        reg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
        reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred = reg.predict(X_te)
        joblib.dump(reg, os.path.join(settings.MODELS_DIR, "xgb_reg_days.pkl"))
        return {"mae": round(mean_absolute_error(y_te, y_pred), 2), "r2": round(r2_score(y_te, y_pred), 4)}

    def _train_neural(self, X, y):
        try:
            from tensorflow.keras import Sequential, layers, callbacks, optimizers, metrics as km
        except ImportError:
            return None

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=42, stratify=y)
        n = X.shape[1]

        model = Sequential([
            layers.Input(shape=(n,)),
            layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
            layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.2),
            layers.Dense(32, activation="relu"), layers.Dropout(0.1),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=optimizers.Adam(0.001), loss="binary_crossentropy",
                      metrics=["accuracy", km.AUC(name="auc")])

        model.fit(X_tr, y_tr, epochs=settings.NEURAL_EPOCHS, batch_size=32,
                  validation_data=(X_te, y_te), verbose=0,
                  callbacks=[
                      callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True),
                      callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
                  ])

        y_prob = model.predict(X_te, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        model.save(os.path.join(settings.MODELS_DIR, "neural_model.keras"))

        return {
            "auc_roc": round(roc_auc_score(y_te, y_prob), 4),
            "accuracy": round(accuracy_score(y_te, y_pred), 4),
            "f1_score": round(f1_score(y_te, y_pred), 4),
        }
