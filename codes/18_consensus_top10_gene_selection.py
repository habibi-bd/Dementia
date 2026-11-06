"""
===============================================================================
STEP 18 - CONSENSUS TOP-10 GENE SELECTION
===============================================================================

Purpose
-------
Select the most consistently important genes across all machine-learning
models using leakage-safe nested feature selection and permutation importance.

Input
-----
D:/Dementia2/codes/ml_data/integrated_gene_expression.csv

Output
------
D:/Dementia2/codes/ml_results/gene_importance/

The script is COMPLETELY INDEPENDENT from Step 17.

It defines:
    - models
    - genes
    - X / y
    - train/test split
    - K
    - CV folds
    - permutation repeats
    - random seed

Important
---------
The independent test set is NEVER used for gene selection.

Gene selection happens inside each CV training fold.
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
import json

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)
from sklearn.utils.multiclass import check_classification_targets

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    roc_auc_score,
)

from sklearn.inspection import permutation_importance

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = r"D:\Dementia2"

INFILE = os.path.join(BASE, "codes", "ml_data", "integrated_gene_expression.csv")

OUT = os.path.join(BASE, "codes", "ml_results", "gene_importance")

os.makedirs(OUT, exist_ok=True)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

SEED = 42


# -----------------------------------------------------------------------------
# Hold-out test set
# -----------------------------------------------------------------------------

TEST_SIZE = 0.20


# -----------------------------------------------------------------------------
# Number of genes selected inside EACH CV training fold
# -----------------------------------------------------------------------------

K = 500


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------

N_SPLITS = 5


# -----------------------------------------------------------------------------
# Permutation importance
# -----------------------------------------------------------------------------

PERM_REPEATS = 5


# -----------------------------------------------------------------------------
# Final number of genes
# -----------------------------------------------------------------------------

N_TOP_GENES = 10


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 18 - CONSENSUS TOP-10 GENE SELECTION")
print("=" * 80)

print()
print("Configuration")
print("-" * 80)

print(f"Input file              : {INFILE}")
print(f"Output directory        : {OUT}")
print(f"Random seed             : {SEED}")
print(f"Test size               : {TEST_SIZE}")
print(f"Genes selected/fold     : {K}")
print(f"CV folds                : {N_SPLITS}")
print(f"Permutation repeats     : {PERM_REPEATS}")
print(f"Final genes             : {N_TOP_GENES}")


# =============================================================================
# LOAD DATA
# =============================================================================

print()
print("=" * 80)
print("LOADING INTEGRATED GENE EXPRESSION DATA")
print("=" * 80)

if not os.path.exists(INFILE):

    raise FileNotFoundError(
        f"\nInput file not found:\n{INFILE}\n\n" "Run Step 16 first."
    )


df = pd.read_csv(INFILE)


# =============================================================================
# REQUIRED METADATA
# =============================================================================

META = [
    "sample_id",
    "condition",
    "label",
    "dataset",
    "platform",
]


missing = [column for column in META if column not in df.columns]


if missing:

    raise ValueError(
        "Missing required metadata columns:\n" + "\n".join(f"  - {x}" for x in missing)
    )


# =============================================================================
# IDENTIFY GENES
# =============================================================================

genes = [column for column in df.columns if column not in META]


if len(genes) == 0:

    raise ValueError("No gene-expression columns were found.")


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

df[genes] = df[genes].apply(pd.to_numeric, errors="coerce").fillna(0)


df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)


# =============================================================================
# BASIC VALIDATION
# =============================================================================

unique_labels = sorted(df["label"].unique())


if not set(unique_labels).issubset({0, 1}):

    raise ValueError("Label column must contain only 0 and 1.")


if len(unique_labels) != 2:

    raise ValueError("Both Alzheimer (1) and Control (0) classes are required.")


# =============================================================================
# CREATE X / y
# =============================================================================

X = df[genes].values.astype(np.float32)

y = df["label"].values.astype(int)


print()
print(f"Samples : {len(df):,}")
print(f"Genes   : {len(genes):,}")
print(f"AD      : {(y == 1).sum():,}")
print(f"Control : {(y == 0).sum():,}")
print(f"Datasets: {df['dataset'].nunique():,}")


# =============================================================================
# DATASET INFORMATION
# =============================================================================

dataset_counts = df["dataset"].value_counts().sort_index()


dataset_counts.to_csv(os.path.join(OUT, "DATASET_SAMPLE_COUNTS.csv"))


# =============================================================================
# LEAKAGE-SAFE TRAIN / TEST SPLIT
# =============================================================================
#
# Exactly the same strategy as Step 17:
#
#     dataset + diagnosis
#
# This ensures the random hold-out preserves the distribution of both.
#
# The test set is NOT used for gene selection.
# =============================================================================

print()
print("=" * 80)
print("CREATING TRAIN / TEST SPLIT")
print("=" * 80)


strata = df["dataset"].astype(str) + "_" + df["label"].astype(str)


try:

    tr, te = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=strata,
    )

    split_method = "dataset + diagnosis stratification"


except ValueError:

    print("WARNING: Dataset + diagnosis stratification failed.")

    print("Falling back to diagnosis-only stratification.")

    tr, te = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    split_method = "diagnosis stratification"


Xtr = X[tr]
Xte = X[te]

ytr = y[tr]
yte = y[te]


print()
print(f"Split method : {split_method}")
print(f"Train        : {len(tr):,}")
print(f"Test         : {len(te):,}")


# =============================================================================
# SAVE TEST SET IDENTIFIERS
# =============================================================================

test_samples = pd.DataFrame(
    {
        "sample_id": df.iloc[te]["sample_id"].values,
        "dataset": df.iloc[te]["dataset"].values,
        "condition": df.iloc[te]["condition"].values,
        "label": yte,
    }
)


test_samples.to_csv(os.path.join(OUT, "test_set_samples.csv"), index=False)


# =============================================================================
# CUSTOM SCRATCH NAIVE BAYES
# =============================================================================


class CustomNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Scratch-built Gaussian Naive Bayes classifier.

    Designed to be compatible with scikit-learn Pipeline and
    permutation_importance.
    """

    def __init__(self, var_smoothing=1e-9):

        self.var_smoothing = var_smoothing

    def fit(self, X, y):

        X, y = check_X_y(X, y, dtype=np.float32)

        check_classification_targets(y)

        self.classes_ = np.unique(y)

        self.n_features_in_ = X.shape[1]

        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}

        global_var = np.var(X, axis=0)

        max_global_var = global_var.max() if global_var.size > 0 else 1.0

        epsilon = self.var_smoothing * max_global_var

        if epsilon <= 0:

            epsilon = 1e-12

        for c in self.classes_:

            X_c = X[y == c]

            if X_c.shape[0] <= 1:

                self.priors_[c] = 1.0 / len(self.classes_)

                self.means_[c] = np.mean(X, axis=0)

                self.vars_[c] = np.var(X, axis=0) + epsilon

            else:

                self.priors_[c] = X_c.shape[0] / X.shape[0]

                self.means_[c] = np.mean(X_c, axis=0)

                self.vars_[c] = np.var(X_c, axis=0) + epsilon

            self.vars_[c] = np.where(self.vars_[c] <= 0, epsilon, self.vars_[c])

        return self

    def _calculate_log_likelihood(self, X, mean, var):

        log_indiv = -0.5 * np.log(2 * np.pi * var + 1e-15) - ((X - mean) ** 2) / (
            2 * var
        )

        return np.sum(log_indiv, axis=1)

    def predict_proba(self, X):

        check_is_fitted(self, attributes=["classes_", "means_", "vars_", "priors_"])

        X = check_array(X, dtype=np.float32)

        log_posteriors = []

        for c in self.classes_:

            log_prior = np.log(self.priors_[c] + 1e-15)

            log_likelihood = self._calculate_log_likelihood(
                X, self.means_[c], self.vars_[c]
            )

            log_posteriors.append(log_prior + log_likelihood)

        log_posteriors = np.column_stack(log_posteriors)

        max_log = np.max(log_posteriors, axis=1, keepdims=True)

        probs = np.exp(log_posteriors - max_log)

        row_sums = np.sum(probs, axis=1, keepdims=True)

        probs = np.where(row_sums == 0, 1.0 / len(self.classes_), probs / row_sums)

        return probs

    def predict(self, X):

        check_is_fitted(self, attributes=["classes_", "means_", "vars_", "priors_"])

        X = check_array(X, dtype=np.float32)

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
#
# IMPORTANT:
# This file defines its OWN models.
#
# Nothing is imported from Step 17.
# =============================================================================

K_EFFECTIVE = min(K, Xtr.shape[1])


def make_pipeline(model):

    return Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif, k=K_EFFECTIVE)),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


models = {
    # -------------------------------------------------------------------------
    # Linear model
    # -------------------------------------------------------------------------
    "LogisticRegression": make_pipeline(
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )
    ),
    # -------------------------------------------------------------------------
    # SVM
    # -------------------------------------------------------------------------
    "SVM_RBF": make_pipeline(
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=SEED,
        )
    ),
    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------
    "RandomForest": make_pipeline(
        RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
    ),
    # -------------------------------------------------------------------------
    # Extra Trees
    # -------------------------------------------------------------------------
    "ExtraTrees": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
    ),
    # -------------------------------------------------------------------------
    # Gradient Boosting
    # -------------------------------------------------------------------------
    "GradientBoosting": make_pipeline(GradientBoostingClassifier(random_state=SEED)),
    # -------------------------------------------------------------------------
    # Histogram Gradient Boosting
    # -------------------------------------------------------------------------
    "HistGradientBoosting": make_pipeline(
        HistGradientBoostingClassifier(random_state=SEED)
    ),
    # -------------------------------------------------------------------------
    # Gaussian NB
    # -------------------------------------------------------------------------
    "GaussianNB": make_pipeline(GaussianNB()),
    # -------------------------------------------------------------------------
    # KNN
    # -------------------------------------------------------------------------
    "KNN": make_pipeline(
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
    ),
}


print()
print("=" * 80)
print("MODELS")
print("=" * 80)

for i, name in enumerate(models.keys(), start=1):

    print(f"{i:2d}. {name}")


print()
print(f"Total models: {len(models)}")


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# =============================================================================
# STORAGE
# =============================================================================

all_model_results = []

model_summary = []


# =============================================================================
# RUN ALL MODELS
# =============================================================================

for model_number, (model_name, model) in enumerate(models.items(), start=1):

    print()
    print()
    print("=" * 80)
    print(f"MODEL {model_number}/{len(models)}: " f"{model_name}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Store gene permutation importance
    # -------------------------------------------------------------------------

    gene_importance = {}

    fold_auc = []

    # -------------------------------------------------------------------------
    # CV
    # -------------------------------------------------------------------------

    for fold, (train_idx, valid_idx) in enumerate(cv.split(Xtr, ytr), start=1):

        print()
        print(f"  Fold {fold}/{N_SPLITS}")

        X_train_fold = Xtr[train_idx]

        y_train_fold = ytr[train_idx]

        X_valid_fold = Xtr[valid_idx]

        y_valid_fold = ytr[valid_idx]

        # ---------------------------------------------------------------------
        # Clone fresh pipeline
        # ---------------------------------------------------------------------

        clf = clone(model)

        # ---------------------------------------------------------------------
        # FIT
        # ---------------------------------------------------------------------

        clf.fit(X_train_fold, y_train_fold)

        # ---------------------------------------------------------------------
        # Validation AUC
        # ---------------------------------------------------------------------

        try:

            probabilities = clf.predict_proba(X_valid_fold)[:, 1]

            auc = roc_auc_score(y_valid_fold, probabilities)

        except Exception:

            try:

                decision = clf.decision_function(X_valid_fold)

                auc = roc_auc_score(y_valid_fold, decision)

            except Exception:

                auc = np.nan

        fold_auc.append(auc)

        print(
            f"    Validation ROC-AUC: " f"{auc:.4f}"
            if not np.isnan(auc)
            else "    Validation ROC-AUC: NaN"
        )

        # ---------------------------------------------------------------------
        # GET SELECTED GENES
        # ---------------------------------------------------------------------

        selector = clf.named_steps["select"]

        selected_mask = selector.get_support()

        selected_indices = np.where(selected_mask)[0]

        selected_genes = [genes[i] for i in selected_indices]

        print(f"    Selected genes: " f"{len(selected_genes):,}")

        # ---------------------------------------------------------------------
        # Transform validation data
        #
        # After SelectKBest + StandardScaler:
        #
        #     columns correspond exactly to selected genes
        # ---------------------------------------------------------------------

        X_valid_selected = selector.transform(X_valid_fold)

        scaler = clf.named_steps["scale"]

        X_valid_scaled = scaler.transform(X_valid_selected)

        final_model = clf.named_steps["model"]

        # ---------------------------------------------------------------------
        # Permutation importance
        #
        # IMPORTANT:
        #
        # We perform permutation importance in the transformed feature space.
        #
        # This avoids confusing the original 20,225 columns with the
        # 500 selected columns.
        # ---------------------------------------------------------------------

        try:

            perm = permutation_importance(
                final_model,
                X_valid_scaled,
                y_valid_fold,
                scoring="roc_auc",
                n_repeats=PERM_REPEATS,
                random_state=(SEED + fold),
                n_jobs=-1,
            )

            importances = perm.importances_mean

            importance_std = perm.importances_std

        except Exception as e:

            print("    WARNING: permutation " "importance failed:")

            print(f"      {e}")

            continue

        # ---------------------------------------------------------------------
        # STORE GENE IMPORTANCE
        # ---------------------------------------------------------------------

        for gene, importance, std in zip(selected_genes, importances, importance_std):

            if gene not in gene_importance:

                gene_importance[gene] = {
                    "importance": [],
                    "std": [],
                }

            gene_importance[gene]["importance"].append(float(importance))

            gene_importance[gene]["std"].append(float(std))

    # =========================================================================
    # AGGREGATE MODEL RESULTS
    # =========================================================================

    rows = []

    for gene, values in gene_importance.items():

        importance_values = np.asarray(values["importance"], dtype=float)

        std_values = np.asarray(values["std"], dtype=float)

        rows.append(
            {
                "gene": gene,
                "importance_mean": np.nanmean(importance_values),
                "importance_std": np.nanstd(importance_values),
                "permutation_std_mean": np.nanmean(std_values),
                "folds_present": len(importance_values),
            }
        )

    model_df = pd.DataFrame(rows)

    if model_df.empty:

        print(f"\nWARNING: No gene importance " f"results for {model_name}")

        continue

    # -------------------------------------------------------------------------
    # Rank
    # -------------------------------------------------------------------------

    model_df = model_df.sort_values("importance_mean", ascending=False).reset_index(
        drop=True
    )

    model_df["model_rank"] = np.arange(len(model_df)) + 1

    model_df["model"] = model_name

    # -------------------------------------------------------------------------
    # Save model result
    # -------------------------------------------------------------------------

    safe_name = model_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

    model_file = os.path.join(OUT, f"{safe_name}_gene_importance.csv")

    model_df.to_csv(model_file, index=False)

    # -------------------------------------------------------------------------
    # Top 10 for this model
    # -------------------------------------------------------------------------

    print()
    print(f"TOP 10 - {model_name}")

    print("-" * 80)

    for rank, row in model_df.head(10).iterrows():

        print(
            f"{rank + 1:2d}. "
            f"{row['gene']:<25} "
            f"importance="
            f"{row['importance_mean']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    all_model_results.append(model_df)

    model_summary.append(
        {
            "model": model_name,
            "mean_cv_auc": np.nanmean(fold_auc),
            "std_cv_auc": np.nanstd(fold_auc),
            "genes_with_importance": len(model_df),
        }
    )


# =============================================================================
# CHECK RESULTS
# =============================================================================

if not all_model_results:

    raise RuntimeError("No model produced gene-importance results.")


# =============================================================================
# COMBINE ALL MODELS
# =============================================================================

combined = pd.concat(all_model_results, ignore_index=True)


combined.to_csv(os.path.join(OUT, "ALL_MODELS_GENE_IMPORTANCE.csv"), index=False)


# =============================================================================
# CONSENSUS RANKING
# =============================================================================

print()
print()
print("=" * 80)
print("BUILDING CONSENSUS GENE RANKING")
print("=" * 80)


consensus_rows = []


for gene, group in combined.groupby("gene"):

    ranks = group["model_rank"].values

    importances = group["importance_mean"].values

    models_supporting = group["model"].nunique()

    consensus_rows.append(
        {
            "gene": gene,
            "mean_rank": np.mean(ranks),
            "median_rank": np.median(ranks),
            "best_rank": np.min(ranks),
            "worst_rank": np.max(ranks),
            "models_supporting": models_supporting,
            "total_models": len(models),
            "support_fraction": (models_supporting / len(models)),
            "mean_importance": np.mean(importances),
            "std_importance": np.std(importances),
            "max_importance": np.max(importances),
            "sum_importance": np.sum(importances),
        }
    )


consensus_df = pd.DataFrame(consensus_rows)


# =============================================================================
# CONSENSUS SCORE
# =============================================================================
#
# Three components:
#
# 1. Rank consistency
# 2. Number of supporting models
# 3. Mean permutation importance
#
# Equal model weighting is used.
# =============================================================================

max_possible_rank = max(len(genes), 1)


consensus_df["rank_score"] = 1.0 - (consensus_df["mean_rank"] - 1) / max_possible_rank


max_importance = consensus_df["mean_importance"].clip(lower=0).max()


if pd.isna(max_importance) or max_importance <= 0:

    max_importance = 1.0


consensus_df["importance_score"] = (
    consensus_df["mean_importance"].clip(lower=0) / max_importance
)


# =============================================================================
# FINAL CONSENSUS SCORE
# =============================================================================
#
# 50% ranking
# 30% model support
# 20% permutation importance
# =============================================================================

consensus_df["consensus_score"] = (
    0.50 * consensus_df["rank_score"]
    + 0.30 * consensus_df["support_fraction"]
    + 0.20 * consensus_df["importance_score"]
)


# =============================================================================
# FINAL SORT
# =============================================================================

consensus_df = consensus_df.sort_values(
    [
        "consensus_score",
        "models_supporting",
        "mean_importance",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)


consensus_df["consensus_rank"] = np.arange(len(consensus_df)) + 1


# =============================================================================
# SAVE COMPLETE CONSENSUS RANKING
# =============================================================================

consensus_file = os.path.join(OUT, "CONSENSUS_GENE_RANKING.csv")


consensus_df.to_csv(consensus_file, index=False)


# =============================================================================
# TOP 10 GENES
# =============================================================================

top10 = consensus_df.head(N_TOP_GENES).copy()


top10_file = os.path.join(OUT, "TOP_10_CONSENSUS_GENES.csv")


top10.to_csv(top10_file, index=False)


# =============================================================================
# PRINT FINAL TOP 10
# =============================================================================

print()
print("=" * 80)
print("FINAL CONSENSUS TOP-10 GENES")
print("=" * 80)

print()


for _, row in top10.iterrows():

    print(
        f"{int(row['consensus_rank']):2d}. "
        f"{row['gene']:<25} "
        f"score={row['consensus_score']:.6f} | "
        f"support="
        f"{int(row['models_supporting'])}/"
        f"{int(row['total_models'])} | "
        f"mean_rank="
        f"{row['mean_rank']:.2f} | "
        f"importance="
        f"{row['mean_importance']:.6f}"
    )


# =============================================================================
# TOP 10 GENE EXPRESSION MATRIX
# =============================================================================

top_genes = top10["gene"].tolist()


top10_matrix = df[META + top_genes].copy()


top10_matrix.to_csv(os.path.join(OUT, "TOP_10_GENE_EXPRESSION_MATRIX.csv"), index=False)


# =============================================================================
# TRAINING TOP-10 MATRIX
# =============================================================================

train_top10 = df.iloc[tr][META + top_genes].copy()


train_top10.to_csv(os.path.join(OUT, "TOP_10_TRAIN_EXPRESSION.csv"), index=False)


# =============================================================================
# TEST TOP-10 MATRIX
# =============================================================================

test_top10 = df.iloc[te][META + top_genes].copy()


test_top10.to_csv(os.path.join(OUT, "TOP_10_TEST_EXPRESSION.csv"), index=False)


# =============================================================================
# MODEL PERFORMANCE SUMMARY
# =============================================================================

model_summary_df = (
    pd.DataFrame(model_summary)
    .sort_values("mean_cv_auc", ascending=False)
    .reset_index(drop=True)
)


model_summary_df.to_csv(os.path.join(OUT, "MODEL_CV_PERFORMANCE.csv"), index=False)


# =============================================================================
# SAVE CONFIGURATION
# =============================================================================

configuration = {
    "input_file": INFILE,
    "samples": int(len(df)),
    "genes": int(len(genes)),
    "alzheimer": int((y == 1).sum()),
    "control": int((y == 0).sum()),
    "datasets": int(df["dataset"].nunique()),
    "train_samples": int(len(tr)),
    "test_samples": int(len(te)),
    "test_size": TEST_SIZE,
    "seed": SEED,
    "K": int(K_EFFECTIVE),
    "cv_folds": N_SPLITS,
    "permutation_repeats": PERM_REPEATS,
    "number_of_models": len(models),
    "number_of_top_genes": N_TOP_GENES,
    "models": list(models.keys()),
    "split_method": split_method,
}


with open(os.path.join(OUT, "GENE_SELECTION_CONFIGURATION.json"), "w") as f:

    json.dump(configuration, f, indent=4)


# =============================================================================
# FINAL OUTPUT SUMMARY
# =============================================================================

print()
print()
print("=" * 80)
print("STEP 18 COMPLETE")
print("=" * 80)

print()

print(f"Input samples          : " f"{len(df):,}")

print(f"Original genes         : " f"{len(genes):,}")

print(f"Training samples       : " f"{len(tr):,}")

print(f"Independent test       : " f"{len(te):,}")

print(f"Models analyzed        : " f"{len(models)}")

print(f"Genes selected/fold    : " f"{K_EFFECTIVE:,}")

print(f"Final consensus genes  : " f"{N_TOP_GENES}")


print()
print("TOP 10 GENES")
print("-" * 80)

for _, row in top10.iterrows():

    print(f"{int(row['consensus_rank']):2d}. " f"{row['gene']}")


print()
print("OUTPUT FILES")
print("-" * 80)

print("  CONSENSUS_GENE_RANKING.csv")

print("  TOP_10_CONSENSUS_GENES.csv")

print("  ALL_MODELS_GENE_IMPORTANCE.csv")

print("  MODEL_CV_PERFORMANCE.csv")

print("  TOP_10_GENE_EXPRESSION_MATRIX.csv")

print("  TOP_10_TRAIN_EXPRESSION.csv")

print("  TOP_10_TEST_EXPRESSION.csv")

print("  test_set_samples.csv")

print("  DATASET_SAMPLE_COUNTS.csv")

print("  GENE_SELECTION_CONFIGURATION.json")

print()
print("Per-model gene importance files were also created.")

print()
print(f"Results saved to:\n{OUT}")

print()
print("=" * 80)
