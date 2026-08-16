"""
Active gene acquisition using all training samples and labels.

Unlike the original tcga_active_fs.py, this module does not simulate a
label-acquisition budget. Every round uses the complete labelled training
set, but only the currently visible genes are given to the classifier.
Hidden genes are evaluated as possible acquisitions and are committed only
when they improve cross-validated balanced accuracy.

This is a global feature-acquisition experiment: when a gene is acquired, it
becomes visible for every sample. It is therefore appropriate for asking
which genes are worth retaining in a cohort-wide assay or classifier panel.
It is not per-patient missing-value acquisition and it is not pool-based
active learning over labels.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tcga_fs import evaluate_models
from tcga_rfe import filter_features_named, prepare_cohort_split, to_symbols

RANDOM_STATE = 42
FILTER_K = 3000
N_ROUNDS = 8
INITIAL_N_FEATURES = 10
GENES_PER_ROUND = 10
CANDIDATES_PER_ROUND = 200
SHORTLIST_MULTIPLIER = 3
FINAL_N_FEATURES = 100
FEATURE_SWEEP_STEP = 10
MIN_SCORE_GAIN = 0.0
BASE_CLASSIFIER = "rf"  # "rf" | "svm" | "mlp"


def _make_classifier(name: str):
    """Build the classifier used for ranking and CV evaluation."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", dual="auto", max_iter=10_000),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
        )
    raise ValueError(f"Unknown base classifier: {name}")


def _cv_splits(y: np.ndarray, requested: int = 5) -> int:
    """Choose a valid stratified-fold count for the observed class counts."""
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(counts))
    if min_class_count < 2:
        raise ValueError("At least two samples are required in every class for CV.")
    return max(2, min(requested, min_class_count))


def _cv_score(clf, X: np.ndarray, y: np.ndarray, n_splits: int) -> float:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    return float(np.mean(scores))


def _classifier_importances(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return classifier-derived importance scores for fitted trial columns."""
    if name == "rf":
        return clf.feature_importances_

    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    return result.importances_mean


def active_gene_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    initial_idx: Sequence[int] | None = None,
    n_rounds: int = N_ROUNDS,
    initial_n_features: int = INITIAL_N_FEATURES,
    genes_per_round: int = GENES_PER_ROUND,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    final_n_features: int = FINAL_N_FEATURES,
    min_score_gain: float = MIN_SCORE_GAIN,
    base_classifier: str = BASE_CLASSIFIER,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Acquire genes globally while using all samples and labels each round.

    The algorithm starts with an initial visible gene panel. In every round:

    1. Score the current panel with stratified CV.
    2. Fit a classifier using the current panel plus a random candidate batch.
    3. Use model-derived importance to shortlist promising hidden genes.
    4. Test shortlisted genes one at a time with CV and commit only genes whose
       marginal score gain is at least ``min_score_gain``.

    ``selected_idx`` is ordered by acquisition: seed genes first, followed by
    genes acquired in later rounds. The held-out test set must not be passed
    to this function.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    feature_names = list(feature_names)

    if X_train.ndim != 2:
        raise ValueError("X_train must be a 2D array.")
    if len(y_train) != X_train.shape[0]:
        raise ValueError("X_train and y_train have incompatible sample counts.")
    if len(feature_names) != X_train.shape[1]:
        raise ValueError("feature_names must match X_train's column count.")
    if n_rounds < 1 or genes_per_round < 1 or final_n_features < 1:
        raise ValueError("n_rounds, genes_per_round, and final_n_features must be positive.")

    rng = np.random.RandomState(random_state)
    n_features = X_train.shape[1]
    max_features = min(final_n_features, n_features)

    if initial_idx is None:
        seed_size = min(initial_n_features, max_features)
        committed_idx = rng.choice(n_features, size=seed_size, replace=False).tolist()
    else:
        committed_idx = list(dict.fromkeys(int(i) for i in initial_idx))
        if any(i < 0 or i >= n_features for i in committed_idx):
            raise ValueError("initial_idx contains an out-of-range column index.")
        committed_idx = committed_idx[:max_features]

    candidate_idx = np.array(
        [i for i in range(n_features) if i not in set(committed_idx)],
        dtype=int,
    )
    n_splits = _cv_splits(y_train)
    history_rows: list[dict] = []
    start = time.perf_counter()

    for round_i in range(1, n_rounds + 1):
        if len(committed_idx) >= max_features or len(candidate_idx) == 0:
            break

        baseline_score = _cv_score(
            _make_classifier(base_classifier),
            X_train[:, committed_idx],
            y_train,
            n_splits,
        )

        batch_size = min(candidates_per_round, len(candidate_idx))
        batch_local = rng.choice(len(candidate_idx), size=batch_size, replace=False)
        batch_global = candidate_idx[batch_local]
        trial_cols = committed_idx + batch_global.tolist()

        clf_trial = _make_classifier(base_classifier)
        clf_trial.fit(X_train[:, trial_cols], y_train)
        importances = _classifier_importances(
            clf_trial,
            X_train[:, trial_cols],
            y_train,
            base_classifier,
        )
        batch_importances = importances[len(committed_idx):]
        ranked_batch = batch_global[np.argsort(batch_importances)[::-1]]

        shortlist_size = min(
            len(ranked_batch),
            max(genes_per_round * SHORTLIST_MULTIPLIER, genes_per_round),
        )
        shortlist = ranked_batch[:shortlist_size]

        current_score = baseline_score
        acquired_this_round: list[int] = []
        candidate_gains: list[float] = []
        remaining_slots = max_features - len(committed_idx)

        # Greedy marginal evaluation: after accepting a gene, evaluate the
        # next candidate relative to the newly expanded visible panel.
        for gene_idx in shortlist:
            if len(acquired_this_round) >= min(genes_per_round, remaining_slots):
                break

            trial_cols = committed_idx + acquired_this_round + [int(gene_idx)]
            trial_score = _cv_score(
                _make_classifier(base_classifier),
                X_train[:, trial_cols],
                y_train,
                n_splits,
            )
            gain = trial_score - current_score
            candidate_gains.append(float(gain))

            if gain >= min_score_gain:
                acquired_this_round.append(int(gene_idx))
                current_score = trial_score

        if acquired_this_round:
            acquired_set = set(acquired_this_round)
            committed_idx.extend(acquired_this_round)
            candidate_idx = np.array(
                [i for i in candidate_idx if i not in acquired_set],
                dtype=int,
            )

        history_rows.append(
            {
                "round": round_i,
                "n_features_before": len(committed_idx) - len(acquired_this_round),
                "batch_evaluated": batch_size,
                "shortlist_evaluated": len(shortlist),
                "n_acquired_this_round": len(acquired_this_round),
                "n_features_after": len(committed_idx),
                "cv_score_before": round(baseline_score, 4),
                "cv_score_after": round(current_score, 4),
                "score_gain": round(current_score - baseline_score, 4),
                "best_candidate_gain": round(max(candidate_gains, default=0.0), 4),
            }
        )

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(history_rows)
    print(
        f"Active gene acquisition ({base_classifier}) finished in {elapsed:.1f}s | "
        f"selected {len(committed_idx)} genes over {len(history)} rounds"
    )

    selected_idx = np.asarray(committed_idx[:max_features], dtype=int)
    selected_names = [feature_names[i] for i in selected_idx]
    return selected_idx, selected_names, history


def sweep_feature_counts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    acquisition_order: Sequence[int],
    sizes: Sequence[int],
    base_classifier: str = BASE_CLASSIFIER,
) -> pd.DataFrame:
    """Evaluate acquisition-order prefixes using all training samples."""
    rows = []
    n_splits = _cv_splits(y_train)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for requested_size in sizes:
        size = min(int(requested_size), len(acquisition_order))
        if size == 0:
            continue
        cols = list(acquisition_order[:size])
        scores = cross_val_score(
            _make_classifier(base_classifier),
            X_train[:, cols],
            y_train,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        rows.append(
            {
                "n_features": size,
                "cv_balanced_accuracy": float(np.mean(scores)),
            }
        )

    return pd.DataFrame(rows)


def report_ranked_genes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_idx: np.ndarray,
    selected_names: list[str],
    gene_symbols: dict[str, str] | None = None,
    top_n: int = 10,
) -> None:
    """Refit a final RF on acquired genes and print its strongest genes."""
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


def run_cohort(
    cohort: str,
    filter_k: int = FILTER_K,
    base_classifier: str = BASE_CLASSIFIER,
    feature_sizes: list[int] | None = None,
    gene_symbols=None,
) -> dict:
    """Run global active gene acquisition and evaluate once on the held-out set."""
    X_train, X_test, y_train, y_test = prepare_cohort_split(cohort, binary=False)

    # This is a train-only statistical pre-filter. The active acquisition
    # stage still uses all samples and labels in the training split.
    Xf_train, Xf_test, filtered_names = filter_features_named(
        X_train,
        X_test,
        y_train,
        k=filter_k,
    )

    sizes = feature_sizes or list(
        range(FEATURE_SWEEP_STEP, FINAL_N_FEATURES + 1, FEATURE_SWEEP_STEP)
    )
    selected_idx, selected_names, history = active_gene_select(
        Xf_train,
        y_train,
        filtered_names,
        initial_n_features=INITIAL_N_FEATURES,
        genes_per_round=GENES_PER_ROUND,
        final_n_features=max(sizes),
        base_classifier=base_classifier,
    )

    print("\nFeature acquisition curve:")
    print(history.to_string(index=False))

    sweep = sweep_feature_counts(
        Xf_train,
        y_train,
        selected_idx.tolist(),
        sizes,
        base_classifier,
    )
    if sweep.empty:
        raise RuntimeError("No feature-count checkpoints were evaluated.")

    print("\nFeature-count sweep (CV balanced accuracy):")
    print(sweep.to_string(index=False))

    best_row = sweep.loc[sweep["cv_balanced_accuracy"].idxmax()]
    best_n = int(best_row["n_features"])
    print(
        f"\nBest feature count by CV: {best_n} "
        f"(balanced acc={best_row['cv_balanced_accuracy']:.4f})"
    )

    best_idx = selected_idx[:best_n]
    best_names = selected_names[:best_n]
    Xs_train, Xs_test = Xf_train[:, best_idx], Xf_test[:, best_idx]

    print("\nHeld-out test performance:")
    for name, acc in evaluate_models(X_train, X_test, y_train, y_test).items():
        print(
            f"  {'all (' + str(X_train.shape[1]) + ')':<22} | "
            f"{name:<13} | accuracy: {acc:.4f}"
        )
    for name, acc in evaluate_models(Xs_train, Xs_test, y_train, y_test).items():
        print(
            f"  {'active-FS (' + str(Xs_train.shape[1]) + ')':<22} | "
            f"{name:<13} | accuracy: {acc:.4f}"
        )

    final_rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(Xs_train, y_train)
    y_pred = final_rf.predict(Xs_test)
    print(
        f"  {'active-FS (RF)':<22} | balanced acc: "
        f"{balanced_accuracy_score(y_test, y_pred):.4f} "
        f"| macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}"
    )

    return {
        "cohort": cohort,
        "genes": to_symbols(best_names, gene_symbols),
        "best_n_features": best_n,
        "history": history,
        "sweep": sweep,
    }


if __name__ == "__main__":
    from tcga_download_helper import load_gene_symbols

    TCGA_COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]
    symbols = load_gene_symbols()
    for cohort_name in TCGA_COHORTS:
        print(f"\n=== {cohort_name} ===")
        try:
            run_cohort(cohort_name, gene_symbols=symbols)
        except (ValueError, RuntimeError) as exc:
            print(f"Skipped: {exc}")
