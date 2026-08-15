import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tcga_download_helper import get_sample_labels, load_tcga_cohort
TCGA_COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]

DEFAULT_K = 100
RANDOM_STATE = 42


def filter_features(
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: np.ndarray,
	k: int = DEFAULT_K,
) -> tuple[np.ndarray, np.ndarray]:
	"""Select the K highest-scoring non-constant features using training data only."""
	variance_filter = VarianceThreshold(threshold=0.0)
	X_train_nonconstant = variance_filter.fit_transform(X_train)
	X_test_nonconstant = variance_filter.transform(X_test)

	selected_k = min(k, X_train_nonconstant.shape[1])
	selector = SelectKBest(score_func=f_classif, k=selected_k)
	X_train_selected = selector.fit_transform(X_train_nonconstant, y_train)
	X_test_selected = selector.transform(X_test_nonconstant)

	return X_train_selected, X_test_selected


def evaluate_models(
	X_train: pd.DataFrame | np.ndarray,
	X_test: pd.DataFrame | np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
) -> dict[str, float]:
	models = {
		"SVM": make_pipeline(
			StandardScaler(),
			LinearSVC(class_weight="balanced", dual="auto", max_iter=10_000),
		),
		"MLP": make_pipeline(
			StandardScaler(),
			MLPClassifier(max_iter=1_000, random_state=RANDOM_STATE),
		),
		"Random Forest": RandomForestClassifier(
			n_estimators=200,
			class_weight="balanced",
			n_jobs=-1,
			random_state=RANDOM_STATE,
		),
	}
	accuracies = {}
	for name, model in models.items():
		model.fit(X_train, y_train)
		accuracies[name] = model.score(X_test, y_test)
	return accuracies


def evaluate_filter(cohort: str, k: int = DEFAULT_K) -> None:
	X = load_tcga_cohort(cohort)
	sample_codes = get_sample_labels(X.index)
	valid_samples = sample_codes >= 0
	X = X.loc[valid_samples]
	sample_codes = sample_codes[valid_samples]

	y = sample_codes
	classes, counts = np.unique(y, return_counts=True)
	if len(classes) < 2:
		raise ValueError(f"Cohort {cohort} contains only one sample type.")

	print(f"Cohort: {cohort.upper()} | Samples: {len(X):,} | Genes: {X.shape[1]:,}")
	distribution = {int(label): int(count) for label, count in zip(classes, counts)}
	print(f"Sample-type distribution: {distribution}")

	singleton_classes = classes[counts < 2]
	singleton_mask = np.isin(y, singleton_classes)
	X_train, X_test, y_train, y_test = train_test_split(
		X.loc[~singleton_mask],
		y[~singleton_mask],
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=y[~singleton_mask],
	)
	if singleton_mask.any():
		X_train = pd.concat([X_train, X.loc[singleton_mask]])
		y_train = np.concatenate([y_train, y[singleton_mask]])

	before = evaluate_models(X_train, X_test, y_train, y_test)
	X_train_selected, X_test_selected = filter_features(X_train, X_test, y_train, k)
	after = evaluate_models(X_train_selected, X_test_selected, y_train, y_test)

	print(f"Features: {X.shape[1]:,} before, {X_train_selected.shape[1]:,} after")
	for name in before:
		print(f"{name}: {before[name]:.4f} before, {after[name]:.4f} after")


if __name__ == "__main__":
	for cohort_name in TCGA_COHORTS:
		print(f"\n=== {cohort_name} ===")
		try:
			evaluate_filter(cohort_name)
		except ValueError as exc:
			print(f"Skipped: {exc}")
