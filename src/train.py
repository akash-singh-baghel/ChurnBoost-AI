"""Train the XGBoost churn classifier + revenue regressor and save artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from xgboost import XGBClassifier, XGBRegressor

from src.data_loader import get_transactions
from src.features import FEATURES, build_dataset

ART = Path(__file__).resolve().parent.parent / "artifacts"
SEED = 42


def risk_tier(p: float) -> str:
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def tune_class_weight(X, y):
    """Pick scale_pos_weight by CV PR-AUC (handles class imbalance)."""
    base = (y == 0).sum() / (y == 1).sum()
    grid = sorted({round(base * m, 3) for m in (0.5, 1.0, 1.5, 2.0)})
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scores = {}
    for w in grid:
        clf = make_clf(w)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        scores[w] = average_precision_score(y, proba)
    best = max(scores, key=scores.get)
    return best, scores


DEFAULT_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8)

SEARCH_SPACE = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.5, 1, 2],
    "reg_lambda": [0.5, 1, 3, 8],
}


def make_clf(spw: float, **params) -> XGBClassifier:
    return XGBClassifier(
        **{**DEFAULT_PARAMS, **params}, scale_pos_weight=spw, eval_metric="aucpr",
        random_state=SEED, n_jobs=-1,
    )


def tune_hyperparameters(X, y, spw, n_iter=40):
    """Randomised search over tree/regularisation params, scored by CV PR-AUC."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(make_clf(spw), SEARCH_SPACE, n_iter=n_iter, scoring="average_precision",
                                cv=cv, random_state=SEED, n_jobs=1, refit=False)
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def main():
    ART.mkdir(exist_ok=True)
    df = get_transactions()
    data, cutoff = build_dataset(df)
    X, y = data[FEATURES], data["churn"]
    print(f"{len(data)} customers | cutoff {cutoff.date()} | churn rate {y.mean():.1%}")

    # --- classifier: tune class weight, evaluate on a stratified hold-out
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    spw, cv_scores = tune_class_weight(X_tr, y_tr)
    print("scale_pos_weight CV PR-AUC:", cv_scores, "-> best", spw)
    baseline_cv = max(cv_scores.values())
    best_params, tuned_cv = tune_hyperparameters(X_tr, y_tr, spw)
    print(f"hyperparameter search: CV PR-AUC {baseline_cv:.4f} -> {tuned_cv:.4f} with {best_params}")
    clf = make_clf(spw, **best_params).fit(X_tr, y_tr)
    proba_te = clf.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= 0.5).astype(int)
    metrics = {
        "n_customers": int(len(data)), "cutoff": str(cutoff.date()),
        "churn_rate": float(y.mean()), "scale_pos_weight": float(spw),
        "class_weight_cv_pr_auc": {str(k): float(v) for k, v in cv_scores.items()},
        "cv_pr_auc_default_params": float(baseline_cv), "cv_pr_auc_tuned": tuned_cv,
        "best_params": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in best_params.items()},
        "roc_auc": float(roc_auc_score(y_te, proba_te)),
        "pr_auc": float(average_precision_score(y_te, proba_te)),
        "precision": float(precision_score(y_te, pred_te)),
        "recall": float(recall_score(y_te, pred_te)),
        "f1": float(f1_score(y_te, pred_te)),
        "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
    }

    # --- revenue regressor: spend of retained customers (log scale)
    retained = data["churn"] == 0
    reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X[retained], np.log1p(data.loc[retained, "future_revenue"]), test_size=0.2, random_state=SEED)
    reg.fit(Xr_tr, yr_tr)
    metrics["revenue_mae_retained"] = float(
        mean_absolute_error(np.expm1(yr_te), np.expm1(reg.predict(Xr_te))))

    # --- score every customer with out-of-fold probabilities (no in-sample optimism)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    data["churn_prob"] = cross_val_predict(make_clf(spw, **best_params), X, y, cv=cv, method="predict_proba")[:, 1]
    data["risk_tier"] = data["churn_prob"].map(risk_tier)
    data["predicted_spend_if_retained"] = np.expm1(reg.predict(X)).clip(min=0)
    data["expected_revenue"] = (1 - data["churn_prob"]) * data["predicted_spend_if_retained"]
    data["revenue_at_risk"] = data["churn_prob"] * data["predicted_spend_if_retained"]
    data["high_value_at_risk"] = (
        (data["risk_tier"] == "High") & (data["monetary"] >= data["monetary"].quantile(0.75)))

    clf_full = make_clf(spw, **best_params).fit(X, y)
    importances = dict(zip(FEATURES, map(float, clf_full.feature_importances_)))
    metrics["feature_importance"] = importances
    metrics["test_proba"] = [float(v) for v in proba_te]
    metrics["test_y"] = [int(v) for v in y_te]

    joblib.dump({"clf": clf_full, "reg": reg, "features": FEATURES}, ART / "models.joblib")
    data.reset_index().to_csv(ART / "customers_scored.csv", index=False)
    (ART / "metrics.json").write_text(json.dumps(metrics))
    print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})
    print("best params:", best_params)


if __name__ == "__main__":
    main()
