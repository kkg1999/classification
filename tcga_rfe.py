import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tcga_download_helper import get_sample_labels, load_gene_symbols, load_tcga_cohort
from tcga_fs import evaluate_models

TCGA_COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]

FILTER_K = 3000          # stage-1 pool handed to the wrapper
MIN_CLASS_SIZE = 3       # need >=1 sample per class in train/test + CV folds
RANDOM_STATE = 42
PRINT_TOP_10_GENE_NAMES = False


# --------------------------------------------------------------------------
# Shared data preparation (imported by tcga_active_svm.py too)
# --------------------------------------------------------------------------
def prepare_cohort_split(
    cohort: str,
    binary: bool = False,
    test_size: float = 0.2,
    min_class_size: int = MIN_CLASS_SIZE,
):
    """Load a cohort, build labels, drop rare classes, and return a stratified split."""
    X = load_tcga_cohort(cohort)
    codes = get_sample_labels(X.index)

    valid = codes >= 0
    X, codes = X.loc[valid], codes[valid]

    if binary:
        # TCGA sample codes: 01-09 tumour, 10-19 normal, 20+ control -> drop
        y = np.where(codes < 10, 1, np.where(codes < 20, 0, -1))
        keep = y >= 0
        X, y = X.loc[keep], y[keep]
    else:
        y = codes

    classes, counts = np.unique(y, return_counts=True)
    keep_classes = classes[counts >= min_class_size]
    dropped = {int(c): int(n) for c, n in zip(classes, counts) if c not in keep_classes}
    if dropped:
        print(f"Dropping rare classes (<{min_class_size} samples): {dropped}")
    mask = np.isin(y, keep_classes)
    X, y = X.loc[mask], y[mask]

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Cohort {cohort} has fewer than two usable classes.")

    print(f"Cohort: {cohort.upper()} | Samples: {len(X):,} | Genes: {X.shape[1]:,}")
    print(f"Class distribution: {({int(c): int(n) for c, n in zip(classes, counts)})}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def filter_features_named(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    k: int = FILTER_K,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Same as tcga_fs.filter_features but also returns the surviving column names."""
    variance_filter = VarianceThreshold(threshold=0.0)
    X_train_nc = variance_filter.fit_transform(X_train)
    X_test_nc = variance_filter.transform(X_test)
    nonconstant_cols = X_train.columns[variance_filter.get_support()]

    selected_k = min(k, X_train_nc.shape[1])
    selector = SelectKBest(score_func=f_classif, k=selected_k)
    X_train_sel = selector.fit_transform(X_train_nc, y_train)
    X_test_sel = selector.transform(X_test_nc)
    selected_cols = nonconstant_cols[selector.get_support()].tolist()

    return X_train_sel, X_test_sel, selected_cols


def to_symbols(feature_ids, gene_symbols: dict[str, str] | None) -> list[str]:
    if not gene_symbols:
        return list(feature_ids)
    return [gene_symbols.get(fid, fid) for fid in feature_ids]


# --------------------------------------------------------------------------
# Stage 2: recursive feature elimination driven by a Random Forest
# --------------------------------------------------------------------------
def rf_rfe_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    min_features: int = 10,
    step: float = 0.1,
    n_estimators: int = 200,
    cv_splits: int = 5,
    scoring: str = "balanced_accuracy",
) -> tuple[np.ndarray, list[str], RFECV]:
    n_classes = len(np.unique(y_train))
    min_class_count = np.min(np.bincount(y_train.astype(int))[np.unique(y_train).astype(int)])
    n_splits = max(2, min(cv_splits, int(min_class_count)))

    print(
        f"\n--- RF-RFECV | pool={X_train.shape[1]} genes | classes={n_classes} "
        f"| cv={n_splits}-fold | step={step} ---"
    )

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rfecv = RFECV(
        estimator=estimator,
        step=step,                       # fraction -> logarithmic shrink, ~20 refits
        min_features_to_select=min_features,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring,
        importance_getter="feature_importances_",
        n_jobs=-1,
    )
    start = time.perf_counter()
    rfecv.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    support = rfecv.get_support()
    selected_idx = np.flatnonzero(support)
    selected_names = [feature_names[i] for i in selected_idx]

    best_score = float(np.max(rfecv.cv_results_["mean_test_score"]))
    print(
        f"RFECV kept {rfecv.n_features_} genes "
        f"(CV {scoring}={best_score:.4f}) in {elapsed:.1f}s"
    )
    return selected_idx, selected_names, rfecv


def report_ranked_genes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_idx: np.ndarray,
    selected_names: list[str],
    gene_symbols: dict[str, str] | None = None,
    top_n: int = 10,
) -> None:
    """Refit a final RF on the selected genes and print the strongest ones."""
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train[:, selected_idx], y_train)

    order = np.argsort(rf.feature_importances_)[::-1]
    symbols = to_symbols(selected_names, gene_symbols)
    print(f"\nTop {min(top_n, len(order))} genes (RF impurity importance):")
    for rank, idx in enumerate(order[:top_n], start=1):
        print(
            f"  #{rank:2d} | {symbols[idx]:<15} | {selected_names[idx]} "
            f"| importance: {rf.feature_importances_[idx]:.4f}"
        )


def score_multiclass(model_scores: dict[str, float], label: str) -> None:
    for name, value in model_scores.items():
        print(f"  {label:<22} | {name:<13} | accuracy: {value:.4f}")



def run_cohort(cohort: str, filter_k: int = FILTER_K, gene_symbols=None) -> dict:
    X_train, X_test, y_train, y_test = prepare_cohort_split(cohort, binary=False)

    # Stage 1: statistical filter (train-only fit)
    Xf_train, Xf_test, filtered_names = filter_features_named(
        X_train, X_test, y_train, k=filter_k
    )

    # Stage 2: RF wrapper
    selected_idx, selected_names, rfecv = rf_rfe_select(
        Xf_train, y_train, filtered_names
    )
    # plot_rfecv_curve(rfecv, cohort)
    Xs_train, Xs_test = Xf_train[:, selected_idx], Xf_test[:, selected_idx]

    print("\nHeld-out test performance:")
    score_multiclass(evaluate_models(X_train, X_test, y_train, y_test),
                     f"all ({X_train.shape[1]})")
    score_multiclass(evaluate_models(Xf_train, Xf_test, y_train, y_test),
                     f"filter ({Xf_train.shape[1]})")
    score_multiclass(evaluate_models(Xs_train, Xs_test, y_train, y_test),
                     f"filter+RFE ({Xs_train.shape[1]})")

    # Class-aware metrics, since accuracy is misleading under imbalance
    final_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(Xs_train, y_train)
    y_pred = final_rf.predict(Xs_test)
    print(
        f"  {'filter+RFE (RF)':<22} | balanced acc: "
        f"{balanced_accuracy_score(y_test, y_pred):.4f} "
        f"| macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}"
    )

    if PRINT_TOP_10_GENE_NAMES:
        report_ranked_genes(Xf_train, y_train, selected_idx, selected_names, gene_symbols)
    return {"cohort": cohort, "genes": to_symbols(selected_names, gene_symbols)}


if __name__ == "__main__":
    symbols = load_gene_symbols()
    for cohort_name in TCGA_COHORTS:
        print(f"\n=== {cohort_name} ===")
        try:
            run_cohort(cohort_name, gene_symbols=symbols)
        except (ValueError, RuntimeError) as exc:
            print(f"Skipped: {exc}")