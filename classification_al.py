import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


class DataLoaders:
    cache = {}
    @staticmethod
    def _encode_target(y):
        # Convert target to a 1D encoded array suitable for classification.
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        y = y.astype(str)
        encoder = LabelEncoder()
        return encoder.fit_transform(y)

    @staticmethod
    def _split(X, y_encoded, test_size=0.2, random_state=42):
        return train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

    @classmethod
    def _load_and_cache(cls, dataset_id, prep_method):
        if dataset_id not in cls.cache:
            # Execute the specific prep method passed in
            X, y = prep_method()
            cls.cache[dataset_id] = (X, cls._encode_target(y))
            
        return cls.cache[dataset_id]

    @staticmethod
    def _prep_uci_15():
        dataset = fetch_ucirepo(id=15)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        # Custom logic: Handle '?' and impute
        X = X.replace("?", np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        return X, y
    
    @staticmethod
    def _prep_uci_17():
        dataset = fetch_ucirepo(id=17)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        # Custom logic: Ensure numeric, strict missing value check
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise ValueError("Unexpected missing values found in UCI dataset 17 features.")
        return X, y
    
    @staticmethod
    def _prep_uci_451():
        dataset = fetch_ucirepo(id=451)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy()

        # Custom logic: Ensure numeric, strict missing value check
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise ValueError("Unexpected missing values found in UCI dataset 451 features.")
        return X, y
    
    @classmethod
    def build_uci_15_breast_cancer_wisconsin_dataloaders(cls, test_size=0.2, random_state=42):
        X, y_encoded = cls._load_and_cache(15, cls._prep_uci_15)
        return cls._split(X, y_encoded, test_size=test_size, random_state=random_state)
    
    @classmethod
    def build_uci_17_breast_cancer_wisconsin_diag_dataloaders(cls, test_size=0.2, random_state=42):
        X, y_encoded = cls._load_and_cache(17, cls._prep_uci_17)
        return cls._split(X, y_encoded, test_size=test_size, random_state=random_state)

    @classmethod
    def build_uci_451_breat_cancer_coimbra_dataloaders(cls, test_size=0.2, random_state=42):
        X, y_encoded = cls._load_and_cache(451, cls._prep_uci_451)
        return cls._split(X, y_encoded, test_size=test_size, random_state=random_state)


class RFClassifier:
    def __init__(self, data_loader=DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders):
        self.data_loader = data_loader

    def classify(self, run_seed):
        X_train, X_test, y_train, y_test = self.data_loader(random_state=run_seed)
        model = RandomForestClassifier(n_estimators=200, random_state=run_seed)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)


class MLPClassifierRunner:
    def __init__(self, data_loader=DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders):
        self.data_loader = data_loader

    @staticmethod
    def _build_model(seed):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=1500,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def classify(self, run_seed):
        X_train, X_test, y_train, y_test = self.data_loader(random_state=run_seed)
        model = self._build_model(seed=run_seed)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)


def build_uci_15_breast_cancer_wisconsin_dataloaders(test_size=0.2, random_state=42):
    return DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders(
        test_size=test_size,
        random_state=random_state,
    )


def build_uci_17_breast_cancer_wisconsin_diag_dataloaders(test_size=0.2, random_state=42):
    return DataLoaders.build_uci_17_breast_cancer_wisconsin_diag_dataloaders(
        test_size=test_size,
        random_state=random_state,
    )


def build_uci_451_breat_cancer_coimbra_dataloaders(test_size=0.2, random_state=42):
    return DataLoaders.build_uci_451_breat_cancer_coimbra_dataloaders(
        test_size=test_size,
        random_state=random_state,
    )


def main_rf(data_loader, num_runs):
    # Run Random Forest classification and report holdout metrics.
    starttime = time.perf_counter()
    rf_runner = RFClassifier(data_loader=data_loader)
    accuracies = []
    for run in range(num_runs):
        accuracy = rf_runner.classify(run)
        accuracies.append(accuracy)
    elapsedtime = time.perf_counter() - starttime
    
    print("Algorithm: Random Forest")
    print(f"best accuracy: {max(accuracies):.4f}")
    print(f"average accuracy: {np.mean(accuracies):.4f}")
    print(f"elapsed time: {elapsedtime:.4f} seconds")
    print("\n\n")


def main_nn(data_loader, num_runs):
    # Run FCNN classification and report holdout metrics.
    starttime = time.perf_counter()
    mlp_runner = MLPClassifierRunner(data_loader=data_loader)
    fcnn_accuracies = []
    for run in range(num_runs):
        accuracy = mlp_runner.classify(run)
        fcnn_accuracies.append(accuracy)
    elapsedtime = time.perf_counter() - starttime

    print("Algorithm: MLP")
    print(f"best accuracy : {max(fcnn_accuracies):.4f}")
    print(f"average accuracy : {np.mean(fcnn_accuracies):.4f}")
    print(f"elapsed time : {elapsedtime:.4f} seconds")
    print("\n\n")


# ----------------------------------------------------------------------
# Active Learning classes
# ----------------------------------------------------------------------


class ActiveLearningRF:
    """
    Uncertainty-based active learning using a Random Forest classifier.
    """
    def __init__(
        self,
        data_loader=DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders,
        initial_size=20,
        budget_frac=0.5,
        query_batch_size=1,
    ):
        self.data_loader = data_loader
        self.initial_size = initial_size
        self.budget_frac = budget_frac
        self.query_batch_size = query_batch_size

    @staticmethod
    def _stratified_initial_mask(y, n_initial, rng):
        """Return a boolean mask with at least one sample per class."""
        classes = np.unique(y)
        per_class = max(1, n_initial // len(classes))
        selected = []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            selected.extend(idx[:per_class])
        if len(selected) < n_initial:
            remaining = np.setdiff1d(np.arange(len(y)), selected)
            rng.shuffle(remaining)
            selected.extend(remaining[: n_initial - len(selected)])
        mask = np.zeros(len(y), dtype=bool)
        mask[np.array(selected)] = True
        return mask

    def classify(self, run_seed):
        X_train, X_test, y_train, y_test = self.data_loader(random_state=run_seed)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        rng = np.random.RandomState(run_seed)
        labeled_mask = self._stratified_initial_mask(
            y_train, min(self.initial_size, len(X_train)), rng
        )

        n_budget = int(len(X_train) * self.budget_frac)
        model = RandomForestClassifier(n_estimators=200, random_state=run_seed)

        for _ in range(n_budget):
            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break

            model.fit(X_train[labeled_mask], y_train[labeled_mask])
            probs = model.predict_proba(X_train[unlabeled_idx])
            uncertainty = 1.0 - np.max(probs, axis=1)

            query_relative = np.argsort(uncertainty)[-self.query_batch_size :][::-1]
            query_idx = unlabeled_idx[query_relative]
            labeled_mask[query_idx] = True

        model.fit(X_train[labeled_mask], y_train[labeled_mask])
        return model.score(X_test, y_test)


class ActiveLearningMLP:
    """
    Uncertainty-based active learning using an MLP classifier.
    """
    def __init__(
        self,
        data_loader=DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders,
        initial_size=20,
        budget_frac=0.5,
        query_batch_size=1,
    ):
        self.data_loader = data_loader
        self.initial_size = initial_size
        self.budget_frac = budget_frac
        self.query_batch_size = query_batch_size

    @staticmethod
    def _stratified_initial_mask(y, n_initial, rng):
        classes = np.unique(y)
        per_class = max(1, n_initial // len(classes))
        selected = []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            selected.extend(idx[:per_class])
        if len(selected) < n_initial:
            remaining = np.setdiff1d(np.arange(len(y)), selected)
            rng.shuffle(remaining)
            selected.extend(remaining[: n_initial - len(selected)])
        mask = np.zeros(len(y), dtype=bool)
        mask[np.array(selected)] = True
        return mask

    def classify(self, run_seed):
        X_train, X_test, y_train, y_test = self.data_loader(random_state=run_seed)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        rng = np.random.RandomState(run_seed)
        labeled_mask = self._stratified_initial_mask(
            y_train, min(self.initial_size, len(X_train)), rng
        )

        n_budget = int(len(X_train) * self.budget_frac)
        model = MLPClassifierRunner._build_model(seed=run_seed)

        for _ in range(n_budget):
            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break

            model.fit(X_train[labeled_mask], y_train[labeled_mask])
            probs = model.predict_proba(X_train[unlabeled_idx])
            uncertainty = 1.0 - np.max(probs, axis=1)

            query_relative = np.argsort(uncertainty)[-self.query_batch_size :][::-1]
            query_idx = unlabeled_idx[query_relative]
            labeled_mask[query_idx] = True

        model.fit(X_train[labeled_mask], y_train[labeled_mask])
        return model.score(X_test, y_test)


class ActiveLearningQBC:
    """
    Query-by-Committee active learning using RF and MLP as the committee.
    Disagreement is measured as the mean absolute difference in predicted
    class probabilities across the committee.
    """
    def __init__(
        self,
        data_loader=DataLoaders.build_uci_15_breast_cancer_wisconsin_dataloaders,
        initial_size=20,
        budget_frac=0.5,
        query_batch_size=1,
    ):
        self.data_loader = data_loader
        self.initial_size = initial_size
        self.budget_frac = budget_frac
        self.query_batch_size = query_batch_size

    @staticmethod
    def _stratified_initial_mask(y, n_initial, rng):
        classes = np.unique(y)
        per_class = max(1, n_initial // len(classes))
        selected = []
        for c in classes:
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            selected.extend(idx[:per_class])
        if len(selected) < n_initial:
            remaining = np.setdiff1d(np.arange(len(y)), selected)
            rng.shuffle(remaining)
            selected.extend(remaining[: n_initial - len(selected)])
        mask = np.zeros(len(y), dtype=bool)
        mask[np.array(selected)] = True
        return mask

    def classify(self, run_seed):
        X_train, X_test, y_train, y_test = self.data_loader(random_state=run_seed)
        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        rng = np.random.RandomState(run_seed)
        labeled_mask = self._stratified_initial_mask(
            y_train, min(self.initial_size, len(X_train)), rng
        )

        n_budget = int(len(X_train) * self.budget_frac)
        rf = RandomForestClassifier(n_estimators=200, random_state=run_seed)
        mlp = MLPClassifierRunner._build_model(seed=run_seed)

        for _ in range(n_budget):
            unlabeled_idx = np.where(~labeled_mask)[0]
            if len(unlabeled_idx) == 0:
                break

            rf.fit(X_train[labeled_mask], y_train[labeled_mask])
            mlp.fit(X_train[labeled_mask], y_train[labeled_mask])

            probs_rf = rf.predict_proba(X_train[unlabeled_idx])
            probs_mlp = mlp.predict_proba(X_train[unlabeled_idx])

            # Mean absolute difference across all class probabilities
            disagreement = np.mean(np.abs(probs_rf - probs_mlp), axis=1)

            query_relative = np.argsort(disagreement)[-self.query_batch_size :][::-1]
            query_idx = unlabeled_idx[query_relative]
            labeled_mask[query_idx] = True

        # Evaluate with a soft-voting ensemble of the two committee members
        rf.fit(X_train[labeled_mask], y_train[labeled_mask])
        mlp.fit(X_train[labeled_mask], y_train[labeled_mask])
        avg_probs = (rf.predict_proba(X_test) + mlp.predict_proba(X_test)) / 2.0
        y_pred = np.argmax(avg_probs, axis=1)
        return accuracy_score(y_test, y_pred)


# ----------------------------------------------------------------------
# Active Learning main callers
# ----------------------------------------------------------------------


def main_active_rf(data_loader, num_runs):
    starttime = time.perf_counter()
    runner = ActiveLearningRF(data_loader=data_loader)
    accuracies = []
    for run in range(num_runs):
        accuracy = runner.classify(run_seed=run)
        accuracies.append(accuracy)
    elapsedtime = time.perf_counter() - starttime
    
    print("Algorithm: Active Learning with Random Forest")
    print(f"best accuracy: {max(accuracies):.4f}")
    print(f"average accuracy: {np.mean(accuracies):.4f}")
    print(f"elapsed time: {elapsedtime:.4f} seconds")
    print("\n\n")


def main_active_mlp(data_loader, num_runs):
    starttime = time.perf_counter()
    runner = ActiveLearningMLP(data_loader=data_loader)
    accuracies = []
    for run in range(num_runs):
        accuracy = runner.classify(run_seed=run)
        accuracies.append(accuracy)
    elapsedtime = time.perf_counter() - starttime

    print("Algorithm: Active Learning with MLP")
    print(f"best accuracy: {max(accuracies):.4f}")
    print(f"average accuracy: {np.mean(accuracies):.4f}")
    print(f"elapsed time: {elapsedtime:.4f} seconds")
    print("\n\n")


def main_active_qbc(data_loader, num_runs):
    starttime = time.perf_counter()
    runner = ActiveLearningQBC(data_loader=data_loader)
    accuracies = []
    for run in range(num_runs):
        accuracy = runner.classify(run_seed=run)
        accuracies.append(accuracy)
    elapsedtime = time.perf_counter() - starttime

    print("Algorithm: Active Learning with QBC (RF + MLP)")
    print(f"best accuracy: {max(accuracies):.4f}")
    print(f"average accuracy: {np.mean(accuracies):.4f}")
    print(f"elapsed time: {elapsedtime:.4f} seconds")
    print("\n\n")


if __name__ == "__main__":
    dataset_loaders = [
        build_uci_15_breast_cancer_wisconsin_dataloaders,
        build_uci_17_breast_cancer_wisconsin_diag_dataloaders,
        build_uci_451_breat_cancer_coimbra_dataloaders,
    ]

    for loader in dataset_loaders:
        print(f"\n=== Dataset loader: {loader.__name__} ===")
        main_rf(data_loader=loader, num_runs=2)
        main_nn(data_loader=loader, num_runs=2)
        main_active_rf(data_loader=loader, num_runs=2)
        main_active_mlp(data_loader=loader, num_runs=2)
        main_active_qbc(data_loader=loader, num_runs=2)