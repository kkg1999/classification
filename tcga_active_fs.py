"""
Active feature selection for TCGA multi-class cohorts, driven by a
classifier's own importance/utility signal rather than mutual information.

Adapted from:
    Maytal Saar-Tsechansky, Prem Melville, Foster Provost, Raymond J. Mooney.
    "Active Feature-Value Acquisition for Classifier Induction."
    Management Science, 55(4), 2009, pp. 664-684.
    DOI: 10.1287/mnsc.1080.0960

The original paper's setting is *feature-value* acquisition: given a
classifier being induced, decide which missing attribute *values* are
worth the cost of acquiring, using an estimate of the "expected value of
acquisition" (EVOA) computed from the classifier itself (e.g. an ensemble
of induced models), rather than an intrinsic/statistical criterion like
mutual information.

TCGA gene expression data has no missing values, so this script adapts
the same core principle to *feature (gene) selection*:

    Instead of scoring candidate genes with a fixed statistical criterion
    computed once (as MI-based or ANOVA-based filters do), let a
    classifier (RF / SVM / MLP -- your choice) tell us, round after round,
    which genes are actually worth "acquiring" into the active feature
    set. A gene's utility is measured by how much adding it improves the
    classifier's *own* cross-validated performance (its "value" to the
    induced model), evaluated under a growing label budget -- mirroring
    the paper's budgeted, classifier-in-the-loop acquisition loop.

Concretely, per round:
  1. Reveal a larger slice of the labelled training pool (label budget
     grows each round, simulating the cost of acquiring more labelled
     examples).
  2. Fit the chosen classifier on the currently committed gene set using
     the revealed labels; get baseline CV score.
  3. Take a random slice of not-yet-committed candidate genes, fit the
     same classifier on committed+candidate genes, and rank candidates by
     the classifier's own importance signal (RF: feature_importances_,
     SVM: |coef_|, MLP: permutation_importance) computed on that model.
  4. Commit the top-utility candidates only if they measurably improve
     the classifier's CV score versus the round's baseline (the paper's
     "expected value of acquisition" idea, simplified to a direct
     score-gain check) -- otherwise keep them as still-uncertain
     candidates for a later round with more labels.
  5. Stop once the feature budget or label budget is exhausted.

Per-round commit rate:
    The number of genes committed each round is paced so that filling the
    full feature budget (final_n_features) takes roughly n_rounds rounds
    (final_n_features / n_rounds per round), rather than a fixed fraction
    of the candidate batch size. This ensures the acquisition loop
    actually spans the whole label-budget schedule (round 1 through
    n_rounds) instead of exhausting the feature budget in just the first
    couple of rounds.

Feature-count sweep:
    Rather than committing to a single hardcoded feature-set size, the
    active loop runs once up to a maximum size (FINAL_N_FEATURES), which
    yields genes ordered by acquisition round. `sweep_feature_counts`
    then evaluates CV performance at increasing prefix sizes (10, 20, ...
    100 by default) of that ordered list, and the best-performing size is
    selected for the final held-out evaluation.

This is a minimal, understandable baseline -- not a faithful
re-implementation of the paper's cost-sensitive EVOA formalism -- meant
as a starting point you can tune and extend.
"""

from __future__ import annotations

import time

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

from tcga_rfe import filter_features_named, prepare_cohort_split, to_symbols
from tcga_fs import evaluate_models

RANDOM_STATE = 42
FILTER_K = 3000              # stage-1 statistical pool handed to the active selector
N_ROUNDS = 8                 # label-budget increments
CANDIDATES_PER_ROUND = 200   # how many not-yet-committed genes to evaluate each round
FINAL_N_FEATURES = 100       # max size of the active-selected gene set (sweep's upper bound)
FEATURE_SWEEP_STEP = 10      # granularity of the feature-count sweep (10, 20, ..., FINAL_N_FEATURES)
MIN_SCORE_GAIN = 0.0         # minimum CV-score improvement required to commit a gene batch
BASE_CLASSIFIER = "rf"       # "rf" | "svm" | "mlp"


def _make_classifier(name: str):
    """Build the classifier used to score candidate genes and evaluate utility."""
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


def _classifier_importances(clf, X: np.ndarray, y: np.ndarray, name: str) -> np.ndarray:
    """
    Return a per-column importance/utility score from the fitted classifier.
    Falls back to permutation importance for models without a native
    importance attribute (SVM, MLP).
    """
    if name == "rf":
        return clf.feature_importances_
    # LinearSVC / MLP pipelines: use permutation importance on held-in data
    # as a model-agnostic proxy for "how much this feature matters to the
    # classifier" -- same spirit as the paper's induced-model-driven scoring.
    result = permutation_importance(
        clf, X, y, n_repeats=5, random_state=RANDOM_STATE, scoring="balanced_accuracy"
    )
    return result.importances_mean


def _cv_score(clf, X: np.ndarray, y: np.ndarray, n_splits: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    return float(np.mean(scores))


def active_feature_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_rounds: int = N_ROUNDS,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    final_n_features: int = FINAL_N_FEATURES,
    min_score_gain: float = MIN_SCORE_GAIN,
    base_classifier: str = BASE_CLASSIFIER,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Budgeted active feature selection driven by a classifier's own
    importance signal and its cross-validated score gain.

    Returns:
        selected_idx: indices (into X_train's columns) of chosen features,
                      ordered by acquisition round (earliest-committed
                      first). Use prefixes of this array to evaluate
                      smaller feature-set sizes without rerunning the loop.
        selected_names: corresponding feature names
        history: DataFrame logging, per round, the label budget, the
                 classifier's CV score before/after committing genes, and
                 how many genes were committed (the "acquisition curve").
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X_train.shape

    candidate_idx = np.arange(n_features)
    committed_idx: list[int] = []

    # Label budget schedule: reveal an increasing fraction of the training
    # pool each round (simulates the cost of acquiring more labelled data).
    budgets = np.linspace(0.3, 1.0, n_rounds)

    # Pace the per-round commit rate so filling final_n_features takes
    # roughly n_rounds rounds, instead of exhausting the feature budget in
    # just the first couple of rounds regardless of the label schedule.
    per_round_target = max(1, int(np.ceil(final_n_features / n_rounds)))

    history_rows = []
    start = time.perf_counter()

    for round_i, frac in enumerate(budgets, start=1):
        if len(committed_idx) >= final_n_features or len(candidate_idx) == 0:
            break

        budget_n = max(int(round(frac * n_samples)), 30)
        budget_n = min(budget_n, n_samples)
        revealed_idx = rng.choice(n_samples, size=budget_n, replace=False)
        X_rev, y_rev = X_train[revealed_idx], y_train[revealed_idx]

        min_class_count = np.min(np.bincount(y_rev.astype(int)))
        n_splits = max(2, min(5, int(min_class_count)))

        # Baseline: classifier's CV score using only currently committed genes
        # (an empty committed set means "no genes yet" -> skip baseline check).
        if committed_idx:
            clf_baseline = _make_classifier(base_classifier)
            baseline_score = _cv_score(clf_baseline, X_rev[:, committed_idx], y_rev, n_splits)
        else:
            baseline_score = -np.inf  # force first round to commit something

        # Sample a batch of not-yet-committed candidate genes to evaluate
        # this round (keeps each round's classifier fit tractable).
        batch_size = min(candidates_per_round, len(candidate_idx))
        batch_local = rng.choice(len(candidate_idx), size=batch_size, replace=False)
        batch_global = candidate_idx[batch_local]
        trial_cols = committed_idx + batch_global.tolist()

        clf_trial = _make_classifier(base_classifier)
        clf_trial.fit(X_rev[:, trial_cols], y_rev)
        importances = _classifier_importances(clf_trial, X_rev[:, trial_cols], y_rev, base_classifier)
        # importances correspond to trial_cols; take only the batch's slice
        batch_importances = importances[len(committed_idx):]

        order = np.argsort(batch_importances)[::-1]
        n_remaining_slots = final_n_features - len(committed_idx)
        # Commit at most per_round_target genes this round (paced across
        # all n_rounds), capped by the batch size and remaining slots.
        n_try = max(1, min(n_remaining_slots, batch_size, per_round_target))
        proposed_local = batch_local[order[:n_try]]
        proposed_global = candidate_idx[proposed_local]

        # Score-gain check: only commit if the classifier's CV score actually
        # improves with the proposed genes added (paper's "value of
        # acquisition" idea, simplified to a direct gain check).
        clf_check = _make_classifier(base_classifier)
        trial_score = _cv_score(
            clf_check, X_rev[:, committed_idx + proposed_global.tolist()], y_rev, n_splits
        )
        gain = trial_score - baseline_score

        if gain >= min_score_gain or not committed_idx:
            committed_idx.extend(proposed_global.tolist())
            candidate_idx = np.array([i for i in candidate_idx if i not in set(proposed_global.tolist())])
            n_committed_this_round = len(proposed_global)
            accepted = True
        else:
            # Not worth committing yet; drop this round's batch back into
            # the candidate pool untouched and try a fresh batch next round.
            n_committed_this_round = 0
            accepted = False

        history_rows.append(
            {
                "round": round_i,
                "labels_queried": budget_n,
                "batch_evaluated": batch_size,
                "n_committed_this_round": n_committed_this_round,
                "n_committed_total": len(committed_idx),
                "cv_score_baseline": None if baseline_score == -np.inf else round(baseline_score, 4),
                "cv_score_with_batch": round(trial_score, 4),
                "score_gain": round(gain, 4),
                "accepted": accepted,
            }
        )

    elapsed = time.perf_counter() - start
    history = pd.DataFrame(history_rows)
    print(
        f"Active FS ({base_classifier}) finished in {elapsed:.1f}s | "
        f"selected {len(committed_idx)} genes over {len(history)} rounds"
    )

    selected_idx = np.array(committed_idx[:final_n_features])
    selected_names = [feature_names[i] for i in selected_idx]
    return selected_idx, selected_names, history


def sweep_feature_counts(
    X_train: np.ndarray,
    y_train: np.ndarray,
    committed_order: list[int],
    sizes: list[int],
    base_classifier: str = BASE_CLASSIFIER,
) -> pd.DataFrame:
    """
    Evaluate CV performance at each feature-count checkpoint by taking
    prefixes of the acquisition-ordered committed gene list, so the best
    feature-set size can be picked instead of a single hardcoded value.

    Returns a DataFrame of n_features -> cv_balanced_accuracy.
    """
    rows = []
    min_class_count = np.min(np.bincount(y_train.astype(int)))
    n_splits = max(2, min(5, int(min_class_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for size in sizes:
        size = min(size, len(committed_order))
        if size == 0:
            continue
        cols = committed_order[:size]
        clf = _make_classifier(base_classifier)
        scores = cross_val_score(
            clf, X_train[:, cols], y_train, cv=cv,
            scoring="balanced_accuracy", n_jobs=-1,
        )
        rows.append({"n_features": size, "cv_balanced_accuracy": float(np.mean(scores))})

    return pd.DataFrame(rows)


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


def run_cohort(
    cohort: str,
    filter_k: int = FILTER_K,
    base_classifier: str = BASE_CLASSIFIER,
    feature_sizes: list[int] | None = None,
    gene_symbols=None,
) -> dict:
    X_train, X_test, y_train, y_test = prepare_cohort_split(cohort, binary=False)

    # Stage 1: cheap statistical pre-filter (same as tcga_rfe.py) to keep
    # the active loop's per-round classifier fits tractable.
    Xf_train, Xf_test, filtered_names = filter_features_named(
        X_train, X_test, y_train, k=filter_k
    )

    # Stage 2: active, classifier-driven, budget-aware feature selection.
    # Run once up to the largest candidate size so the acquisition-ordered
    # gene list can be swept over cheaply, instead of rerunning per size.
    sizes = feature_sizes or list(range(FEATURE_SWEEP_STEP, FINAL_N_FEATURES + 1, FEATURE_SWEEP_STEP))
    max_size = max(sizes)
    selected_idx, selected_names, history = active_feature_select(
        Xf_train, y_train, filtered_names,
        final_n_features=max_size, base_classifier=base_classifier,
    )

    print("\nAcquisition curve (per round):")
    print(history.to_string(index=False))

    # Sweep feature-set sizes (10, 20, ..., FINAL_N_FEATURES by default)
    # over the acquisition-ordered gene list and pick the best by CV score.
    sweep = sweep_feature_counts(Xf_train, y_train, selected_idx.tolist(), sizes, base_classifier)
    print("\nFeature-count sweep (CV balanced accuracy):")
    print(sweep.to_string(index=False))

    best_row = sweep.loc[sweep["cv_balanced_accuracy"].idxmax()]
    best_n = int(best_row["n_features"])
    print(f"\nBest feature count by CV: {best_n} (balanced acc={best_row['cv_balanced_accuracy']:.4f})")

    best_idx = selected_idx[:best_n]
    best_names = selected_names[:best_n]
    Xs_train, Xs_test = Xf_train[:, best_idx], Xf_test[:, best_idx]

    print("\nHeld-out test performance:")
    for name, acc in evaluate_models(X_train, X_test, y_train, y_test).items():
        print(f"  {'all (' + str(X_train.shape[1]) + ')':<22} | {name:<13} | accuracy: {acc:.4f}")
    for name, acc in evaluate_models(Xf_train, Xf_test, y_train, y_test).items():
        print(f"  {'filter (' + str(Xf_train.shape[1]) + ')':<22} | {name:<13} | accuracy: {acc:.4f}")
    for name, acc in evaluate_models(Xs_train, Xs_test, y_train, y_test).items():
        print(f"  {'active-FS (' + str(Xs_train.shape[1]) + ')':<22} | {name:<13} | accuracy: {acc:.4f}")

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
