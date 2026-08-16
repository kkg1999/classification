"""
Global active feature selection for cost-conscious TCGA gene-panel design.

This experiment treats conventional active learning along the feature axis:
all training labels are available, while the globally visible gene panel grows
adaptively. Candidate genes are screened in batches and retained only when
their conditional cross-validated improvement is large enough relative to its
uncertainty.

The complete retrospective TCGA expression matrix is available during panel
discovery. Consequently, this code primarily estimates deployment-time savings
for future samples measured with the final panel. It separately tracks genes
screened during discovery so that search cost is not confused with retained
panel size. This is not per-patient dynamic feature acquisition, missing-value
acquisition, pool-based active learning over labels, or multi-label learning.
"""

from __future__ import annotations

import math
import time
import traceback
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
FILTER_K = 5000
N_ROUNDS = 10
INITIAL_N_FEATURES = 10
GENES_PER_ROUND = 10
CANDIDATES_PER_ROUND = 200
SHORTLIST_MULTIPLIER = 3
FINAL_N_FEATURES = 200
FEATURE_SWEEP_STEP = 10
MIN_SCORE_GAIN = 0.0
ACCEPTANCE_SE_MULTIPLIER = 1.0
NO_GAIN_PATIENCE = 3
BASE_CLASSIFIER = "rf"  # "rf" | "svm" | "mlp"
PRINT_TOP_10_GENE_NAMES = True
EVALUATE_ALL_FEATURES = True


def _make_classifier(name: str):
    """Build the classifier used for ranking and CV evaluation."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=10_000,
                random_state=RANDOM_STATE,
            ),
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


def _cv_scores(clf, X: np.ndarray, y: np.ndarray, n_splits: int) -> np.ndarray:
    """Return fold scores using a fixed split for paired panel comparisons."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )


def _score_summary(scores: np.ndarray) -> tuple[float, float, float]:
    """Return mean, sample standard deviation, and standard error."""
    scores = np.asarray(scores, dtype=float)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    se = std / math.sqrt(len(scores)) if len(scores) else 0.0
    return mean, std, se


def _classifier_importances(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    name: str,
) -> np.ndarray:
    """Return classifier-derived scores for the fitted trial columns.

    Random forests use their embedded impurity importance for inexpensive
    screening. SVM and MLP pipelines use permutation importance. These scores
    only create a shortlist; final acquisition is decided by paired CV gain.
    """
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
    acceptance_se_multiplier: float = ACCEPTANCE_SE_MULTIPLIER,
    no_gain_patience: int = NO_GAIN_PATIENCE,
    base_classifier: str = BASE_CLASSIFIER,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Construct a globally shared gene panel through adaptive acquisition.

    In each round, the method:

    1. evaluates the current panel with stratified CV;
    2. samples a batch from the unretained genes;
    3. fits a screening model on the panel plus the candidate batch;
    4. shortlists candidates using model-derived importance;
    5. evaluates shortlisted genes sequentially using paired CV folds; and
    6. retains a gene only when the lower-confidence gain is sufficient.

    The acceptance rule is::

        mean(fold gains) - acceptance_se_multiplier * SE(fold gains)
            >= min_score_gain

    ``selected_idx`` is ordered by acquisition. The held-out test set must not
    be passed to this function.
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
        raise ValueError(
            "n_rounds, genes_per_round, and final_n_features must be positive."
        )
    if candidates_per_round < 1:
        raise ValueError("candidates_per_round must be positive.")
    if acceptance_se_multiplier < 0:
        raise ValueError("acceptance_se_multiplier cannot be negative.")
    if no_gain_patience < 1:
        raise ValueError("no_gain_patience must be positive.")

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

    available_mask = np.ones(n_features, dtype=bool)
    available_mask[committed_idx] = False

    n_splits = _cv_splits(y_train)
    history_rows: list[dict] = []
    screened_genes: set[int] = set()
    total_screenings = 0
    empty_rounds = 0
    start = time.perf_counter()

    for round_i in range(1, n_rounds + 1):
        candidate_idx = np.flatnonzero(available_mask)
        if len(committed_idx) >= max_features or len(candidate_idx) == 0:
            break

        current_scores = _cv_scores(
            _make_classifier(base_classifier),
            X_train[:, committed_idx],
            y_train,
            n_splits,
        )
        baseline_score, _, baseline_se = _score_summary(current_scores)
        current_score = baseline_score

        batch_size = min(candidates_per_round, len(candidate_idx))
        batch_global = rng.choice(candidate_idx, size=batch_size, replace=False)
        screened_genes.update(int(i) for i in batch_global)
        total_screenings += batch_size

        screening_cols = committed_idx + batch_global.tolist()
        clf_trial = _make_classifier(base_classifier)
        clf_trial.fit(X_train[:, screening_cols], y_train)
        importances = _classifier_importances(
            clf_trial,
            X_train[:, screening_cols],
            y_train,
            base_classifier,
        )
        batch_importances = importances[len(committed_idx) :]
        ranked_batch = batch_global[np.argsort(batch_importances)[::-1]]

        shortlist_size = min(
            len(ranked_batch),
            max(genes_per_round * SHORTLIST_MULTIPLIER, genes_per_round),
        )
        shortlist = ranked_batch[:shortlist_size]

        acquired_this_round: list[int] = []
        candidate_mean_gains: list[float] = []
        candidate_gain_ses: list[float] = []
        candidate_conservative_gains: list[float] = []
        remaining_slots = max_features - len(committed_idx)

        # Greedy conditional evaluation. Once a gene is accepted, the next
        # candidate is compared with the newly expanded panel on the same folds.
        for gene_idx in shortlist:
            if len(acquired_this_round) >= min(genes_per_round, remaining_slots):
                break

            gene_idx = int(gene_idx)
            trial_cols = committed_idx + acquired_this_round + [gene_idx]
            trial_scores = _cv_scores(
                _make_classifier(base_classifier),
                X_train[:, trial_cols],
                y_train,
                n_splits,
            )

            fold_gains = trial_scores - current_scores
            mean_gain, _, gain_se = _score_summary(fold_gains)
            conservative_gain = mean_gain - acceptance_se_multiplier * gain_se

            candidate_mean_gains.append(mean_gain)
            candidate_gain_ses.append(gain_se)
            candidate_conservative_gains.append(conservative_gain)

            if conservative_gain >= min_score_gain:
                acquired_this_round.append(gene_idx)
                current_scores = trial_scores
                current_score = float(np.mean(current_scores))

        if acquired_this_round:
            committed_idx.extend(acquired_this_round)
            available_mask[acquired_this_round] = False
            empty_rounds = 0
        else:
            empty_rounds += 1

        history_rows.append(
            {
                "round": round_i,
                "n_features_before": len(committed_idx) - len(acquired_this_round),
                "batch_evaluated": batch_size,
                "shortlist_evaluated": len(shortlist),
                "n_screened_this_round": batch_size,
                "n_screened_total": total_screenings,
                "n_screened_unique": len(screened_genes),
                "n_acquired_this_round": len(acquired_this_round),
                "n_retained_total": len(committed_idx),
                "n_features_after": len(committed_idx),
                "retention_per_unique_screened": (
                    len(committed_idx) / max(1, len(screened_genes))
                ),
                "cv_score_before": baseline_score,
                "cv_score_before_se": baseline_se,
                "cv_score_after": current_score,
                "score_gain": current_score - baseline_score,
                "best_candidate_mean_gain": max(
                    candidate_mean_gains, default=0.0
                ),
                "best_candidate_gain_se": (
                    candidate_gain_ses[
                        int(np.argmax(candidate_mean_gains))
                    ]
                    if candidate_mean_gains
                    else 0.0
                ),
                "best_conservative_gain": max(
                    candidate_conservative_gains, default=0.0
                ),
                "consecutive_empty_rounds": empty_rounds,
            }
        )

        if empty_rounds >= no_gain_patience:
            break

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(history_rows)
    print(
        f"Active gene acquisition ({base_classifier}) finished in {elapsed:.1f}s | "
        f"retained {len(committed_idx)} genes | "
        f"screened {len(screened_genes)} unique genes ({total_screenings} total) | "
        f"rounds: {len(history)}"
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
    """Evaluate unique acquisition-order prefixes using all training samples."""
    rows = []
    n_splits = _cv_splits(y_train)

    evaluated_sizes = sorted(
        {
            min(int(requested_size), len(acquisition_order))
            for requested_size in sizes
            if int(requested_size) > 0
        }
    )

    for size in evaluated_sizes:
        if size == 0:
            continue
        cols = list(acquisition_order[:size])
        scores = _cv_scores(
            _make_classifier(base_classifier),
            X_train[:, cols],
            y_train,
            n_splits,
        )
        score_mean, score_std, score_se = _score_summary(scores)
        rows.append(
            {
                "n_features": size,
                "cv_balanced_accuracy": score_mean,
                "cv_score_std": score_std,
                "cv_score_se": score_se,
            }
        )

    return pd.DataFrame(rows)


def choose_minimal_sufficient_panel(sweep: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Apply the one-standard-error rule to choose a compact panel.

    Returns the smallest panel whose mean CV score is within one standard error
    of the peak panel, followed by the peak-scoring row itself.
    """
    if sweep.empty:
        raise ValueError("Cannot choose a panel from an empty sweep.")

    peak_row = sweep.loc[sweep["cv_balanced_accuracy"].idxmax()]
    sufficient_score = float(peak_row["cv_balanced_accuracy"]) - float(
        peak_row["cv_score_se"]
    )
    eligible = sweep[
        sweep["cv_balanced_accuracy"] >= sufficient_score
    ].sort_values("n_features")
    selected_row = eligible.iloc[0]
    return selected_row, peak_row


def report_ranked_genes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_idx: np.ndarray,
    selected_names: list[str],
    gene_symbols: dict[str, str] | None = None,
    top_n: int = 10,
) -> None:
    """Refit a final RF on the chosen panel and print permutation importance."""
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=RANDOM_STATE,
    ).fit(X_train[:, selected_idx], y_train)

    importance = permutation_importance(
        rf,
        X_train[:, selected_idx],
        y_train,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    ).importances_mean
    order = np.argsort(importance)[::-1]
    symbols = to_symbols(selected_names, gene_symbols)

    print(f"\nTop {min(top_n, len(order))} genes (permutation importance):")
    for rank, idx in enumerate(order[:top_n], start=1):
        print(
            f"  #{rank:2d} | {symbols[idx]:<15} | {selected_names[idx]} "
            f"| importance: {importance[idx]:.4f}"
        )


def run_cohort(
    cohort: str,
    filter_k: int = FILTER_K,
    base_classifier: str = BASE_CLASSIFIER,
    feature_sizes: list[int] | None = None,
    gene_symbols=None,
    evaluate_all_features: bool = EVALUATE_ALL_FEATURES,
) -> dict:
    """Discover a minimal panel and evaluate it once on the held-out set."""
    X_train, X_test, y_train, y_test = prepare_cohort_split(cohort, binary=False)

    # Train-only statistical pre-filter. The held-out set is transformed using
    # the training-derived filter and is not used during feature acquisition.
    Xf_train, Xf_test, filtered_names = filter_features_named(
        X_train,
        X_test,
        y_train,
        k=filter_k,
    )

    sizes = feature_sizes or list(
        range(FEATURE_SWEEP_STEP, FINAL_N_FEATURES + 1, FEATURE_SWEEP_STEP)
    )
    if not sizes or any(int(size) <= 0 for size in sizes):
        raise ValueError("feature_sizes must contain positive integers.")

    target_n_features = min(max(int(size) for size in sizes), Xf_train.shape[1])
    required_rounds = max(
        1,
        math.ceil(
            max(0, target_n_features - INITIAL_N_FEATURES) / GENES_PER_ROUND
        ),
    )

    selected_idx, selected_names, history = active_gene_select(
        Xf_train,
        y_train,
        filtered_names,
        n_rounds=max(N_ROUNDS, required_rounds),
        initial_n_features=INITIAL_N_FEATURES,
        genes_per_round=GENES_PER_ROUND,
        final_n_features=target_n_features,
        base_classifier=base_classifier,
    )

    sweep = sweep_feature_counts(
        Xf_train,
        y_train,
        selected_idx.tolist(),
        sizes,
        base_classifier,
    )
    if sweep.empty:
        raise RuntimeError("No feature-count checkpoints were evaluated.")

    selected_row, peak_row = choose_minimal_sufficient_panel(sweep)
    best_n = int(selected_row["n_features"])
    peak_n = int(peak_row["n_features"])
    best_idx = selected_idx[:best_n]
    best_names = selected_names[:best_n]
    Xs_train, Xs_test = Xf_train[:, best_idx], Xf_test[:, best_idx]

    print("\nFeature-count sweep (CV balanced accuracy):")
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nPeak CV panel: {peak_n} genes "
        f"(balanced acc={peak_row['cv_balanced_accuracy']:.4f})"
    )
    print(
        f"Minimal sufficient panel (one-SE rule): {best_n} genes "
        f"(balanced acc={selected_row['cv_balanced_accuracy']:.4f})"
    )

    print("\nHeld-out test performance:")
    if evaluate_all_features:
        for name, acc in evaluate_models(X_train, X_test, y_train, y_test).items():
            print(
                f"  {'all (' + str(X_train.shape[1]) + ')':<22} | "
                f"{name:<13} | accuracy: {acc:.4f}"
            )

    for name, acc in evaluate_models(Xf_train, Xf_test, y_train, y_test).items():
        print(
            f"  {'filter (' + str(Xf_train.shape[1]) + ')':<22} | "
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
        n_jobs=1,
        random_state=RANDOM_STATE,
    ).fit(Xs_train, y_train)
    y_pred = final_rf.predict(Xs_test)
    held_out_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    held_out_macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(
        f"  {'active-FS (RF)':<22} | balanced acc: "
        f"{held_out_balanced_accuracy:.4f} "
        f"| macro F1: {held_out_macro_f1:.4f}"
    )

    if PRINT_TOP_10_GENE_NAMES:
        report_ranked_genes(
            Xf_train,
            y_train,
            best_idx,
            best_names,
            gene_symbols,
        )

    final_history = history.iloc[-1] if not history.empty else None
    return {
        "cohort": cohort,
        "genes": to_symbols(best_names, gene_symbols),
        "best_n_features": best_n,
        "peak_n_features": peak_n,
        "selected_cv_score": float(selected_row["cv_balanced_accuracy"]),
        "peak_cv_score": float(peak_row["cv_balanced_accuracy"]),
        "held_out_balanced_accuracy": float(held_out_balanced_accuracy),
        "held_out_macro_f1": float(held_out_macro_f1),
        "n_screened_unique": (
            int(final_history["n_screened_unique"])
            if final_history is not None
            else 0
        ),
        "n_screened_total": (
            int(final_history["n_screened_total"])
            if final_history is not None
            else 0
        ),
        "history": history,
        "sweep": sweep,
    }


test_algo = False
if __name__ == "__main__":
    from tcga_download_helper import load_gene_symbols

    TCGA_COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]
    if test_algo:
        TCGA_COHORTS = ["BRCA", "COAD"]

    symbols = load_gene_symbols()
    for cohort_name in TCGA_COHORTS:
        print(f"\n=== {cohort_name} ===")
        try:
            run_cohort(cohort_name, gene_symbols=symbols)
        except (ValueError, RuntimeError):
            print(f"Skipped {cohort_name} because of an error:")
            traceback.print_exc()
