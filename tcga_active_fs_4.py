from __future__ import annotations

import gc
import json
import math
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tcga_download_helper import load_gene_symbols
from tcga_rfe import filter_features_named, prepare_cohort_split, to_symbols

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]

FILTER_K = 10_000            # train-only statistical pre-filter handed to the loop
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_SCORING = "f1_macro"      # the gate optimises the metric the paper reports

# accuracy, then macro precision / recall / F1. Macro recall is identical to
# balanced accuracy, so nothing is lost relative to the earlier scripts.
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
# Ranking of the printed panel. Named distinctly from N_REPEATS (outer splits):
# this is the number of shuffles per gene inside permutation_importance.
PERM_REPEATS = 20
PERM_SCORER = make_scorer(f1_score, average="macro", zero_division=0)

INITIAL_PANEL_SIZE = 5       # random cold start, so the method is not handed the baseline
CANDIDATES_PER_ROUND = 200   # genes screened per round by the cheap ranker
GENES_PER_ROUND = 5          # acquisition budget per round
SHORTLIST_MULTIPLIER = 4     # expensive evaluations per round = 4 x GENES_PER_ROUND
MAX_PANEL_SIZE = 80          # hard ceiling on assay cost
MAX_ROUNDS = 50               # 50% of a 10k pool is ~25 rounds of 200, plus patience
PATIENCE = 10                # consecutive barren rounds after the coverage floor
MIN_POOL_FRACTION = 0.5      # do not stop on patience before this fraction is screened
ACCEPTANCE_SE_MULTIPLIER = 1.0  # lambda in gain - lambda * SE(gain) > 0

BASE_CLASSIFIER = "rf"       # "rf" | "svm" | "mlp"
SCREENING_TREES = 100        # cheap forest for the inner loop
FINAL_TREES = 300            # forest used for held-out reporting
RANDOM_STATE = 42            # model seed, kept separate from the split seed
N_REPEATS = 10               # outer train/test splits
N_RANDOM_PANELS = 5          # draws averaged for the random-panel floor
TRANSFER_CLASSIFIERS = ["rf", "svm", "mlp"]

OUTPUT_DIR = Path("results_fs4")


# --------------------------------------------------------------------------
# Data: one TCGA cohort at a time (same labels as tcga_active_fs_3)
# --------------------------------------------------------------------------
def load_cohort(cohort: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one cohort with the same sample-type labels as ``tcga_active_fs_3``.

    ``prepare_cohort_split(..., binary=False)`` uses the barcode sample-type
    code (01 tumour, 11 adjacent normal, ...), drops classes with fewer than
    three samples, and requires at least two remaining classes. The inner
    train/test split is discarded so ``run_repeat`` can draw its own splits.
    """
    X_train, X_test, y_train, y_test = prepare_cohort_split(cohort, binary=False)
    X = pd.concat([X_train, X_test], axis=0)
    y = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    return X, y


# --------------------------------------------------------------------------
# Models and cross-validation helpers
# --------------------------------------------------------------------------
def _make_classifier(
    name: str = BASE_CLASSIFIER,
    random_state: int = RANDOM_STATE,
    n_estimators: int = SCREENING_TREES,
    n_jobs: int = 1,
):
    """Build the classifier used for ranking, CV utility and final reporting.

    ``n_jobs`` stays at 1 inside ``cross_val_score``, which already parallelises
    over folds, and is set to -1 for standalone fits such as the per-round
    screening model and the final held-out model.
    """
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=10_000,
                random_state=random_state,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(max_iter=500, random_state=random_state),
        )
    raise ValueError(f"Unknown base classifier: {name}")


def _cv_splits(y: np.ndarray, requested: int = CV_SPLITS) -> int:
    """Pick a stratified fold count that every class can support."""
    _, counts = np.unique(y, return_counts=True)
    smallest = int(np.min(counts))
    if smallest < 2:
        raise ValueError("Every class needs at least two samples for CV.")
    return max(2, min(requested, smallest))


def _cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    cols: Sequence[int],
    n_splits: int,
    fold_seed: int,
    base_classifier: str = BASE_CLASSIFIER,
) -> np.ndarray:
    """Fold-wise ``CV_SCORING`` on a fixed partition.

    ``fold_seed`` is held constant for the whole acquisition run so that any two
    panels are scored on identical folds and their difference is paired.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
    return cross_val_score(
        _make_classifier(base_classifier),
        X[:, list(cols)],
        y,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=-1,
    )


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    """Mean and the usual (optimistic, see module docstring) standard error."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    se = float(np.std(values, ddof=1)) / math.sqrt(values.size)
    return mean, se


def _candidate_importances(
    clf, X: np.ndarray, y: np.ndarray, name: str, n_panel: int
) -> np.ndarray:
    """Cheap relevance score for the candidate block of a fitted screening model."""
    if name == "rf":
        return clf.feature_importances_[n_panel:]
    result = permutation_importance(
        clf,
        X,
        y,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring=CV_SCORING,
        n_jobs=-1,
    )
    return result.importances_mean[n_panel:]


def _sample_candidates(
    rng: np.random.RandomState,
    available: np.ndarray,
    unscreened: np.ndarray,
    size: int,
) -> np.ndarray:
    """Draw a candidate batch, preferring genes that have never been screened.

    Sampling uniformly at random every round re-screens genes that were already
    rejected and leaves much of the pool unseen. Spending the budget on unseen
    genes first makes pool coverage grow monotonically, which is what the cost
    argument in the paper needs.
    """
    fresh = np.flatnonzero(available & unscreened)
    if len(fresh) >= size:
        return rng.choice(fresh, size=size, replace=False)

    seen = np.flatnonzero(available & ~unscreened)
    take = min(size - len(fresh), len(seen))
    extra = (
        rng.choice(seen, size=take, replace=False)
        if take > 0
        else np.empty(0, dtype=int)
    )
    return np.concatenate([fresh, extra]).astype(int)


# --------------------------------------------------------------------------
# Active feature acquisition
# --------------------------------------------------------------------------
def active_acquire(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    base_classifier: str = BASE_CLASSIFIER,
    initial_panel_size: int = INITIAL_PANEL_SIZE,
    candidates_per_round: int = CANDIDATES_PER_ROUND,
    genes_per_round: int = GENES_PER_ROUND,
    shortlist_multiplier: int = SHORTLIST_MULTIPLIER,
    max_panel_size: int = MAX_PANEL_SIZE,
    max_rounds: int = MAX_ROUNDS,
    patience: int = PATIENCE,
    min_pool_fraction: float = MIN_POOL_FRACTION,
    se_multiplier: float = ACCEPTANCE_SE_MULTIPLIER,
) -> dict:
    """Grow a gene panel by gated, cost-aware acquisition.

    Each round screens a batch of candidate genes with a cheap model, shortlists
    the most promising ones, and then evaluates them one at a time against the
    current panel on fixed folds. A candidate is acquired only when

        mean(paired fold gain) - se_multiplier * SE(paired fold gain) > 0

    so a gene whose apparent benefit is within CV noise is left out. Patience
    (``patience`` consecutive barren rounds) is ignored until at least
    ``min_pool_fraction`` of the pool has been screened, so a cold start plus
    random batches cannot stop the search after seeing only a sliver of the
    genes.

    Only training data may be passed in. The returned ``history`` doubles as the
    cost-versus-accuracy curve, so no separate feature-count sweep is needed and
    no second argmax over panel sizes is taken.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n_features = X_train.shape[1]

    if X_train.ndim != 2:
        raise ValueError("X_train must be 2D.")
    if len(y_train) != X_train.shape[0]:
        raise ValueError("X_train and y_train disagree on the sample count.")
    if max_panel_size < 1 or genes_per_round < 1 or candidates_per_round < 1:
        raise ValueError("Panel, batch and round budgets must be positive.")
    if se_multiplier < 0:
        raise ValueError("se_multiplier cannot be negative.")
    if not 0.0 <= min_pool_fraction <= 1.0:
        raise ValueError("min_pool_fraction must be between 0 and 1.")

    rng = np.random.RandomState(seed)
    n_splits = _cv_splits(y_train)
    max_panel_size = min(max_panel_size, n_features)
    shortlist_size = max(genes_per_round, genes_per_round * shortlist_multiplier)
    min_screened = math.ceil(min_pool_fraction * n_features)

    panel = rng.choice(
        n_features, size=min(initial_panel_size, max_panel_size), replace=False
    ).tolist()
    available = np.ones(n_features, dtype=bool)
    available[panel] = False
    unscreened = np.ones(n_features, dtype=bool)
    unscreened[panel] = False

    panel_scores = _cv_scores(
        X_train, y_train, panel, n_splits, seed, base_classifier
    )
    n_cv_evaluations = 1
    n_screened_unique = int((~unscreened).sum())
    barren_rounds = 0
    history: list[dict] = []
    stop_reason = "round budget exhausted"
    start = time.perf_counter()

    for round_i in range(1, max_rounds + 1):
        if len(panel) >= max_panel_size:
            stop_reason = "panel size ceiling reached"
            break
        if not available.any():
            stop_reason = "candidate pool exhausted"
            break

        cv_before, _ = _mean_se(panel_scores)

        # Query step: choose which genes are worth an expensive evaluation.
        batch = _sample_candidates(
            rng, available, unscreened, min(candidates_per_round, int(available.sum()))
        )
        if batch.size == 0:
            stop_reason = "candidate pool exhausted"
            break
        # Counted from the mask, not by adding the batch size: a fallback batch
        # can re-screen a gene, and pool coverage is a reported cost metric.
        unscreened[batch] = False
        n_screened_unique = int((~unscreened).sum())

        screening_cols = panel + batch.tolist()
        screener = _make_classifier(base_classifier, n_jobs=-1).fit(
            X_train[:, screening_cols], y_train
        )
        importances = _candidate_importances(
            screener,
            X_train[:, screening_cols],
            y_train,
            base_classifier,
            len(panel),
        )
        shortlist = batch[np.argsort(importances)[::-1]][:shortlist_size]

        # Decision step: acquire only genes whose paired gain beats CV noise.
        accepted: list[int] = []
        n_rejected = 0
        best_gain = -np.inf
        for gene in shortlist:
            if len(accepted) >= genes_per_round:
                break
            if len(panel) + len(accepted) >= max_panel_size:
                break

            gene = int(gene)
            trial_scores = _cv_scores(
                X_train,
                y_train,
                panel + accepted + [gene],
                n_splits,
                seed,
                base_classifier,
            )
            n_cv_evaluations += 1

            mean_gain, gain_se = _mean_se(trial_scores - panel_scores)
            adjusted_gain = mean_gain - se_multiplier * gain_se
            best_gain = max(best_gain, adjusted_gain)

            if adjusted_gain > 0:
                accepted.append(gene)
                panel_scores = trial_scores
            else:
                n_rejected += 1

        if accepted:
            panel.extend(accepted)
            available[accepted] = False
            barren_rounds = 0
        else:
            barren_rounds += 1

        cv_after, cv_after_se = _mean_se(panel_scores)
        history.append(
            {
                "round": round_i,
                "panel_size_before": len(panel) - len(accepted),
                "panel_size_after": len(panel),
                "n_screened_this_round": int(batch.size),
                "n_screened_unique": n_screened_unique,
                "n_evaluated_this_round": len(accepted) + n_rejected,
                "n_accepted": len(accepted),
                "n_rejected": n_rejected,
                "best_adjusted_gain": (
                    float(best_gain) if np.isfinite(best_gain) else 0.0
                ),
                "cv_before": cv_before,
                "cv_after": cv_after,
                "cv_after_se": cv_after_se,
                "n_cv_evaluations": n_cv_evaluations,
                "elapsed_s": time.perf_counter() - start,
            }
        )

        if barren_rounds >= patience and n_screened_unique >= min_screened:
            stop_reason = (
                f"no acquisition in {patience} consecutive rounds "
                f"(screened {n_screened_unique}/{n_features})"
            )
            break

    elapsed = time.perf_counter() - start
    cv_mean, cv_se = _mean_se(panel_scores)
    print(
        f"  acquisition | {len(panel)} genes | {len(history)} rounds | "
        f"{n_cv_evaluations} CV evaluations | {n_screened_unique} genes screened "
        f"| CV {cv_mean:.4f} | {elapsed:.1f}s | stopped: {stop_reason}"
    )
    return {
        "panel": np.asarray(panel, dtype=int),
        "history": pd.DataFrame(history),
        "cv_mean": cv_mean,
        "cv_se": cv_se,
        "n_cv_evaluations": n_cv_evaluations,
        "n_screened_unique": n_screened_unique,
        "n_rounds": len(history),
        "elapsed_s": elapsed,
        "stop_reason": stop_reason,
    }


# --------------------------------------------------------------------------
# Size-matched baselines and evaluation
# --------------------------------------------------------------------------
def anova_top_k(X_train: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Highest-scoring k genes by univariate ANOVA F, fitted on training data."""
    scores, _ = f_classif(X_train, y_train)
    scores = np.nan_to_num(scores, nan=-np.inf)
    return np.argsort(scores)[::-1][:k]


def evaluate_panel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cols: Sequence[int],
    classifier: str = BASE_CLASSIFIER,
) -> dict[str, float]:
    """Held-out accuracy and macro precision, recall and F1 for one panel.

    Macro recall is the same quantity as balanced accuracy.
    """
    cols = list(cols)
    model = _make_classifier(classifier, n_estimators=FINAL_TREES, n_jobs=-1)
    model.fit(X_train[:, cols], y_train)
    predicted = model.predict(X_test[:, cols])
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_macro": float(
            precision_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predicted, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predicted, average="macro", zero_division=0)
        ),
    }


# --------------------------------------------------------------------------
# One repeat = one outer split
# --------------------------------------------------------------------------
def run_repeat(X: pd.DataFrame, y: np.ndarray, seed: int) -> dict:
    """Split, filter, acquire, and compare against size-matched baselines."""
    print(f"\n--- repeat seed={seed} ---")
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    # Train-only pre-filter; the test split is only transformed by it.
    Xf_train, Xf_test, filtered_names = filter_features_named(
        X_train_df, X_test_df, y_train, k=FILTER_K
    )
    # The full-width frames are ~1 GB with every patient kept; release them
    # before the acquisition loop, which is the long-running part.
    del X_train_df, X_test_df
    gc.collect()
    print(
        f"  split | train={Xf_train.shape[0]} test={Xf_test.shape[0]} "
        f"| pool={Xf_train.shape[1]} genes"
    )

    result = active_acquire(Xf_train, y_train, seed=seed)
    panel = result["panel"]
    k = len(panel)

    rows = [
        {
            "method": "active",
            "n_features": k,
            **evaluate_panel(Xf_train, Xf_test, y_train, y_test, panel),
        },
        {
            "method": "anova_top_k",
            "n_features": k,
            **evaluate_panel(
                Xf_train, Xf_test, y_train, y_test,
                anova_top_k(Xf_train, y_train, k),
            ),
        },
    ]

    # Random floor: does the acquisition order matter beyond the pre-filter?
    rng = np.random.RandomState(seed)
    random_runs = [
        evaluate_panel(
            Xf_train,
            Xf_test,
            y_train,
            y_test,
            rng.choice(Xf_train.shape[1], size=k, replace=False),
        )
        for _ in range(N_RANDOM_PANELS)
    ]
    rows.append(
        {
            "method": "random_panel",
            "n_features": k,
            **{
                metric: float(np.mean([run[metric] for run in random_runs]))
                for metric in METRICS
            },
        }
    )
    rows.append(
        {
            "method": "full_filtered_pool",
            "n_features": Xf_train.shape[1],
            **evaluate_panel(
                Xf_train, Xf_test, y_train, y_test, range(Xf_train.shape[1])
            ),
        }
    )

    # Does the panel survive a change of classifier, or is it an RF artefact?
    transfer = [
        {
            "classifier": name,
            **evaluate_panel(
                Xf_train, Xf_test, y_train, y_test, panel, classifier=name
            ),
        }
        for name in TRANSFER_CLASSIFIERS
    ]

    scores = pd.DataFrame(rows)
    print(scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Ordering for the printed panel: permutation importance measured on the
    # held-out split. Impurity (MDI) importance is biased towards high-variance,
    # high-cardinality predictors (Strobl et al. 2007) -- a poor prior on log2
    # counts -- and is measured in-sample. Permutation importance is
    # model-agnostic, so it also works when BASE_CLASSIFIER is svm or mlp.
    #
    # This ordering is descriptive only: it selects nothing and does not enter
    # any reported metric, so the held-out scores above remain unbiased.
    ranker = _make_classifier(
        BASE_CLASSIFIER, n_estimators=FINAL_TREES, n_jobs=-1
    )
    ranker.fit(Xf_train[:, list(panel)], y_train)
    perm = permutation_importance(
        ranker,
        Xf_test[:, list(panel)],
        y_test,
        n_repeats=PERM_REPEATS,
        random_state=RANDOM_STATE,
        scoring=PERM_SCORER,
        n_jobs=-1,
    )
    panel_importances = perm.importances_mean.astype(float)
    panel_importances_std = perm.importances_std.astype(float)

    return {
        "seed": seed,
        "panel_names": [filtered_names[i] for i in panel],
        "scores": scores,
        "transfer": pd.DataFrame(transfer),
        "history": result["history"],
        "n_features": k,
        "cv_mean": result["cv_mean"],
        "n_cv_evaluations": result["n_cv_evaluations"],
        "n_screened_unique": result["n_screened_unique"],
        "elapsed_s": result["elapsed_s"],
        "stop_reason": result["stop_reason"],
        "panel_importances": panel_importances,
        "panel_importances_std": panel_importances_std,
        "pool_size": int(Xf_train.shape[1]),
    }


def summarise(repeats: list[dict]) -> pd.DataFrame:
    """Mean and standard deviation of every metric across the outer splits."""
    combined = pd.concat(
        [r["scores"].assign(seed=r["seed"]) for r in repeats], ignore_index=True
    )
    aggregations = {"n_features": ("n_features", "mean")}
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    return (
        combined.groupby("method")
        .agg(**aggregations)
        .reset_index()
        .sort_values("f1_macro_mean", ascending=False)
    )


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse each metric to a readable 'mean +/- sd' column for printing."""
    formatted = summary[["method", "n_features"]].copy()
    formatted["n_features"] = formatted["n_features"].round(1)
    for metric in METRICS:
        formatted[metric] = [
            f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f} +/- {std:.4f}"
            for mean, std in zip(
                summary[f"{metric}_mean"], summary[f"{metric}_std"]
            )
        ]
    return formatted


def panel_stability(repeats: list[dict]) -> float:
    """Mean pairwise Jaccard overlap between the panels found on each split."""
    panels = [set(r["panel_names"]) for r in repeats]
    overlaps = [
        len(a & b) / len(a | b)
        for i, a in enumerate(panels)
        for b in panels[i + 1:]
        if a | b
    ]
    return float(np.mean(overlaps)) if overlaps else float("nan")


def report_best_panel(
    repeats: list[dict], symbols: dict[str, str] | None = None
) -> tuple[dict, pd.DataFrame]:
    """Pick and print the single best panel across the outer splits.

    The winner is chosen by ``cv_mean``, the selection-time cross-validation
    score, which is computed on training data only. Choosing by held-out score
    would make the reported test metric a biased, optimistically selected
    number rather than an honest generalisation estimate.

    Ties break towards the smaller panel (cheaper assay), then the lower seed,
    so the choice is fully deterministic.
    """
    best = sorted(
        repeats, key=lambda r: (-r["cv_mean"], r["n_features"], r["seed"])
    )[0]

    print(f"\n=== best panel (selected by CV {CV_SCORING} on training data) ===")
    print(
        f"  seed={best['seed']} | {best['n_features']} genes "
        f"| selection CV={best['cv_mean']:.4f} | stopped: {best['stop_reason']}"
    )

    gene_symbols = to_symbols(best["panel_names"], symbols)
    ranked = pd.DataFrame(
        {
            "ensembl": best["panel_names"],
            "symbol": gene_symbols,
            "importance": best["panel_importances"],
            "importance_std": best["panel_importances_std"],
            # Position in the panel list is the order genes were acquired; the
            # first INITIAL_PANEL_SIZE are the random cold start and were never
            # put through the acceptance gate.
            "acquisition_order": range(1, best["n_features"] + 1),
        }
    )
    ranked["cold_start"] = ranked["acquisition_order"] <= INITIAL_PANEL_SIZE
    # Descending importance, ties broken by acquisition order so the printed
    # list is deterministic. Permutation importance is signed: a gene whose
    # removal *improves* the held-out score scores below zero and sorts last,
    # which is the correct place for it.
    ranked = ranked.sort_values(
        ["importance", "acquisition_order"], ascending=[False, True]
    ).reset_index(drop=True)

    print(
        f"  ordered by held-out permutation importance "
        f"(drop in macro F1 over {PERM_REPEATS} shuffles):"
    )
    for rank, row in enumerate(ranked.itertuples(), start=1):
        flag = " [cold-start]" if row.cold_start else ""
        print(
            f"    {rank:3d}. {row.symbol:<15} | {row.ensembl:<22} "
            f"| {row.importance:+.4f} +/- {row.importance_std:.4f} "
            f"| acq #{row.acquisition_order}{flag}"
        )

    n_positive = int((ranked["importance"] > 0).sum())
    print(
        f"\n  {n_positive} of {len(ranked)} genes have positive held-out "
        f"importance; the remainder are redundant given the rest of the panel."
    )

    active_row = best["scores"].loc[best["scores"]["method"] == "active"]
    print("\n  held-out performance of this panel (reported, not used to pick it):")
    print(
        "   "
        + active_row.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        .replace("\n", "\n   ")
    )
    return best, ranked


def recurrent_genes(
    repeats: list[dict], symbols: dict[str, str] | None = None, min_splits: int = 2
) -> pd.DataFrame:
    """Genes selected on several splits, with a chance-level significance test.

    Under the null that a panel is drawn uniformly from the filtered pool, a
    given gene enters one panel with probability p = k / pool_size. Recurrence
    across the independent outer splits is then Binomial(N_REPEATS, p), so
    P(X >= m) is the probability of seeing a gene at least m times by chance.
    Benjamini-Hochberg controls the FDR across every gene that was selected.

    Two honest caveats for the write-up:
      * the splits share ~80% of their samples, so they are not fully
        independent and the p-values are somewhat anti-conservative;
      * only genes selected at least once are tested, so a gene at m = 1 is
        conditioned on being selected and its p-value is not meaningful. Read
        the m >= 2 rows, which is what ``min_splits`` defaults to.
    """
    n_repeats = len(repeats)
    counts: dict[str, int] = {}
    for r in repeats:
        for gene in r["panel_names"]:
            counts[gene] = counts.get(gene, 0) + 1

    # Mean per-split inclusion probability under uniform random selection.
    pool_size = float(np.mean([r["pool_size"] for r in repeats]))
    mean_panel_size = float(np.mean([r["n_features"] for r in repeats]))
    p_null = mean_panel_size / pool_size if pool_size > 0 else 0.0

    genes = list(counts)
    if not genes:
        table = pd.DataFrame(
            columns=[
                "ensembl",
                "symbol",
                "n_splits",
                "frequency",
                "expected_by_chance",
                "p_binomial",
                "p_adj_bh",
            ]
        )
        print(
            f"\n=== genes selected on >= {min_splits} of {n_repeats} splits "
            f"(0 of 0 ever selected) ==="
        )
        print(
            f"  null: pool={pool_size:.0f} genes, p={p_null:.5f} per split, "
            f"expected recurrence={n_repeats * p_null:.3f}"
        )
        print("  none -- every gene was selected on at most one split")
        print("\n  significant at BH-adjusted 5%: 0 genes")
        return table

    observed = np.array([counts[g] for g in genes])
    # sf(m - 1) == P(X >= m)
    p_values = binom.sf(observed - 1, n_repeats, p_null)

    order = np.argsort(p_values)
    ranks = np.arange(1, len(genes) + 1)
    adjusted = np.empty(len(genes), dtype=float)
    adjusted[order] = np.minimum.accumulate(
        (p_values[order] * len(genes) / ranks)[::-1]
    )[::-1].clip(0.0, 1.0)

    table = pd.DataFrame(
        {
            "ensembl": genes,
            "symbol": to_symbols(genes, symbols),
            "n_splits": observed,
            "frequency": observed / n_repeats,
            "expected_by_chance": n_repeats * p_null,
            "p_binomial": p_values,
            "p_adj_bh": adjusted,
        }
    ).sort_values(
        ["n_splits", "p_adj_bh"], ascending=[False, True]
    ).reset_index(drop=True)

    recurrent = table[table["n_splits"] >= min_splits]
    significant = recurrent[recurrent["p_adj_bh"] < 0.05]
    print(
        f"\n=== genes selected on >= {min_splits} of {n_repeats} splits "
        f"({len(recurrent)} of {len(table)} ever selected) ==="
    )
    print(
        f"  null: pool={pool_size:.0f} genes, p={p_null:.5f} per split, "
        f"expected recurrence={n_repeats * p_null:.3f}"
    )
    if len(recurrent):
        print(recurrent.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        print("  none -- every gene was selected on at most one split")
    print(f"\n  significant at BH-adjusted 5%: {len(significant)} genes")
    return table


def save_cohort_results(
    cohort: str,
    repeats: list[dict],
    symbols: dict[str, str] | None,
    out_dir: Path,
) -> None:
    """Print and write the usual tables for one cohort."""
    summary = summarise(repeats)
    print("\n=== held-out performance across "
          f"{N_REPEATS} splits (mean +/- sd) ===")
    print(format_summary(summary).to_string(index=False))

    transfer = (
        pd.concat([r["transfer"].assign(seed=r["seed"]) for r in repeats])
        .groupby("classifier")
        .agg(**{metric: (metric, "mean") for metric in METRICS})
        .reset_index()
    )
    print("\n=== active panel evaluated with other classifiers (mean) ===")
    print(transfer.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cost = pd.DataFrame(
        [
            {
                "seed": r["seed"],
                "n_features": r["n_features"],
                f"cv_{CV_SCORING}_selection": r["cv_mean"],
                "n_cv_evaluations": r["n_cv_evaluations"],
                "n_screened_unique": r["n_screened_unique"],
                "elapsed_s": r["elapsed_s"],
                "stop_reason": r["stop_reason"],
            }
            for r in repeats
        ]
    )
    print("\n=== acquisition cost ===")
    print(cost.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nPanel stability (mean pairwise Jaccard): {panel_stability(repeats):.4f}")
    best, best_ranked = report_best_panel(repeats, symbols)
    recurrence = recurrent_genes(repeats, symbols)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    transfer.to_csv(out_dir / "transfer.csv", index=False)
    cost.to_csv(out_dir / "cost.csv", index=False)
    recurrence.to_csv(out_dir / "gene_recurrence.csv", index=False)
    best_ranked.to_csv(out_dir / "best_panel_ranked.csv", index=False)
    pd.concat(
        [r["history"].assign(seed=r["seed"]) for r in repeats], ignore_index=True
    ).to_csv(out_dir / "acquisition_history.csv", index=False)
    panels = {
        str(r["seed"]): {
            "ensembl": r["panel_names"],
            "symbols": to_symbols(r["panel_names"], symbols),
            "is_best": r["seed"] == best["seed"],
        }
        for r in repeats
    }
    (out_dir / "panels.json").write_text(json.dumps(panels, indent=2))
    print(f"\nWrote {cohort} results to {out_dir.resolve()}")


def run_cohort(cohort: str, symbols: dict[str, str] | None = None) -> None:
    """Run active feature selection on one TCGA cohort."""
    X, y = load_cohort(cohort)
    repeats = [run_repeat(X, y, seed=RANDOM_STATE + i) for i in range(N_REPEATS)]
    del X, y
    gc.collect()
    save_cohort_results(cohort, repeats, symbols, OUTPUT_DIR / cohort)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = load_gene_symbols()
    for cohort in COHORTS:
        print(f"\n=== {cohort} ===")
        try:
            run_cohort(cohort, symbols)
        except (ValueError, RuntimeError):
            print(f"Skipped {cohort} because of an error:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
