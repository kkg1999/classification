import gc
import os
import urllib.error
import urllib.request
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, mutual_info_classif


PROBEMAP_FILE = "gencode.v36.annotation.gtf.gene.probemap"
PROBEMAP_URL = f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{PROBEMAP_FILE}"
TCGA_COHORTS = ["BRCA", "COAD", "LUSC", "GBM", "OV", "LUAD", "THCA"]


def load_tcga_cohort(cohort: str, save_dir: str = "./tcga_data") -> pd.DataFrame:
    """
    Downloads and loads the UCSC Xena pre-processed STAR counts for a given TCGA cohort.
    Returns a DataFrame with samples as rows and genes as columns.
    """
    os.makedirs(save_dir, exist_ok=True)
    cohort = cohort.upper()
    file_name = f"TCGA-{cohort}.star_counts.tsv.gz"
    local_path = os.path.join(save_dir, file_name)
    url = f"https://gdc-hub.s3.us-east-1.amazonaws.com/download/{file_name}"

    # Download only if file doesn't already exist locally
    if not os.path.exists(local_path):
        print(f"Downloading {cohort} from UCSC Xena...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except urllib.error.HTTPError as exc:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(
                f"UCSC Xena download failed for {cohort} ({exc.code}): {url}"
            ) from exc
        print(f"Downloaded: {local_path}")
    else:
        print(f"Loading cached file: {local_path}")

    # Read TSV (Xena files have Genes as rows and Patients/Samples as columns)
    df = pd.read_csv(local_path, sep="\t", index_col=0)

    # Transpose to ML standard: (Samples, Genes)
    df_ml = df.T
    return df_ml


def load_gene_symbols(save_dir: str = "./tcga_data") -> dict[str, str]:
    """Downloads the matching GENCODE probemap and returns Ensembl-to-symbol mappings."""
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, PROBEMAP_FILE)

    if not os.path.exists(local_path):
        print("Downloading GENCODE gene-symbol map from UCSC Xena...")
        try:
            urllib.request.urlretrieve(PROBEMAP_URL, local_path)
        except urllib.error.HTTPError as exc:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(
                f"UCSC Xena probemap download failed ({exc.code}): {PROBEMAP_URL}"
            ) from exc

    probemap = pd.read_csv(local_path, sep="\t", usecols=["id", "gene"])
    return dict(zip(probemap["id"], probemap["gene"]))


def get_sample_labels(sample_barcodes: pd.Index) -> np.ndarray:
    labels = []
    for barcode in sample_barcodes:
        parts = str(barcode).split("-")
        if len(parts) >= 4:
            sample_code = int(parts[3][:2])
            labels.append(sample_code)
        else:
            labels.append(-1) # Flag malformed barcodes
    return np.array(labels)


def select_top_features_anova(
    X_df: pd.DataFrame,
    y: np.ndarray,
    top_k: int = 10,
    gene_symbols: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    print(f"\n--- Running Feature Selection (Target features to keep: {top_k}) ---")

    variance_filter = VarianceThreshold(threshold=0.0)
    X_filtered = variance_filter.fit_transform(X_df)
    surviving_feature_ids = X_df.columns[variance_filter.get_support()]
    removed_count = X_df.shape[1] - X_filtered.shape[1]
    print(f"Removed {removed_count} constant genes.")

    k = min(top_k, X_filtered.shape[1])
    print("SelectKBest...")
    selector = SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=42), k=k)
    selector.fit(X_filtered, y)
    scores = np.nan_to_num(selector.scores_, nan=-np.inf)
    ranked_indices = np.argsort(scores)[::-1]
    
    # Print Top 10 genes
    print("\nTop 10 Genes by filtering:")
    for rank in range(min(10, len(ranked_indices))):
        idx = ranked_indices[rank]
        feature_id = surviving_feature_ids[idx]
        gene_name = gene_symbols.get(feature_id, feature_id) if gene_symbols else feature_id
        print(f"  #{rank + 1:2d} | Gene: {gene_name} | MI-score: {scores[idx]:.4f}")

    selected_indices = ranked_indices[:k]
    selected_feature_ids = surviving_feature_ids[selected_indices].tolist()
    selected_gene_names = [
        gene_symbols.get(feature_id, feature_id) if gene_symbols else feature_id
        for feature_id in selected_feature_ids
    ]
    X_selected = pd.DataFrame(
        X_filtered[:, selected_indices],
        index=X_df.index,
        columns=selected_gene_names,
    )

    return X_selected, selected_gene_names


if __name__ == "__main__":
    top_k = 10
    gene_symbols = load_gene_symbols()

    print("=== PROCESSING TCGA COHORTS ONE AT A TIME ===")
    for cohort in TCGA_COHORTS:
        print(f"\n=== {cohort} ===")
        cohort_df = load_tcga_cohort(cohort)
        print(
            f"Cohort: {cohort} | Samples: {cohort_df.shape[0]} | "
            f"Genes: {cohort_df.shape[1]}"
        )

        labels = get_sample_labels(cohort_df.index)
        cohort_reduced, cohort_genes = select_top_features_anova(
            cohort_df, labels, top_k=top_k, gene_symbols=gene_symbols
        )

        tumor_count = int(labels.sum())
        normal_count = len(labels) - tumor_count
        # Replace your current tumor/normal print statements with this:
        final_classes, final_counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(final_classes, final_counts))
        print(f"Samples: {len(labels)} | Class Distribution: {distribution}")
        print(f"Final output shape: {cohort_reduced.shape}")

        del cohort_df, labels, cohort_reduced, cohort_genes
        gc.collect()
        print(f"Released {cohort} data from memory.")