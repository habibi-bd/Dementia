"""
===============================================================================
STEP 18 - CUSTOM GR-TCN CONSENSUS TOP-10 GENE SELECTION
===============================================================================

Purpose
-------
Leakage-safe identification of the most important genes from a high-dimensional
gene-expression dataset using a custom GR-TCN classifier.

The algorithm performs:

    1. Robust CSV loading
    2. Dataset + diagnosis stratified train/test split
    3. Stratified K-fold CV ONLY on the training set
    4. Fold-wise SelectKBest gene filtering
    5. Fold-wise StandardScaler
    6. Custom GR-TCN neural classifier
    7. Permutation importance on validation data
    8. Aggregation of gene importance across folds
    9. Consensus ranking
    10. Final top-10 genes

IMPORTANT
---------
The independent test set is NEVER used for gene selection.

NO FILES ARE WRITTEN.

Everything is printed to the console.

Input
-----
D:/Dementia2/codes/ml_data/integrated_gene_expression.csv

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings
import random

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Make PyTorch as deterministic as practical
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = r"D:\Dementia2"

INFILE = os.path.join(
    BASE,
    "codes",
    "ml_data",
    "integrated_gene_expression.csv",
)

TEST_SIZE = 0.20

# Number of genes retained INSIDE EACH TRAINING FOLD
K = 500

N_SPLITS = 5

PERM_REPEATS = 5

N_TOP_GENES = 10


# =============================================================================
# CUSTOM GR-TCN PARAMETERS
# =============================================================================

EPOCHS = 80

LEARNING_RATE = 0.005

L1_LAMBDA = 1e-3

LAPLACIAN_LAMBDA = 1e-3

TEMPERATURE = 0.5

BATCH_SIZE = 64


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("STEP 18 - CUSTOM GR-TCN TOP-10 GENE SELECTION")
print("=" * 90)

print()
print("CONFIGURATION")
print("-" * 90)

print(f"Input file              : {INFILE}")
print(f"Random seed             : {SEED}")
print(f"Device                  : {DEVICE}")
print(f"Test size               : {TEST_SIZE}")
print(f"Genes selected/fold     : {K}")
print(f"CV folds                : {N_SPLITS}")
print(f"Permutation repeats     : {PERM_REPEATS}")
print(f"Final genes             : {N_TOP_GENES}")

print()
print("GR-TCN PARAMETERS")
print("-" * 90)

print(f"Epochs                  : {EPOCHS}")
print(f"Learning rate           : {LEARNING_RATE}")
print(f"L1 lambda               : {L1_LAMBDA}")
print(f"Laplacian lambda        : {LAPLACIAN_LAMBDA}")
print(f"Temperature             : {TEMPERATURE}")
print(f"Batch size              : {BATCH_SIZE}")


# =============================================================================
# ROBUST CSV LOADER
# =============================================================================
#
# The original error:
#
# pandas.errors.ParserError:
# C error: Calling read(nbytes) on source failed.
#
# We try several safe loading strategies.
#
# NO DATA IS MODIFIED OR SAVED.
# =============================================================================


def load_gene_expression_csv(path):

    print()
    print("=" * 90)
    print("LOADING GENE EXPRESSION DATA")
    print("=" * 90)

    if not os.path.exists(path):

        raise FileNotFoundError(f"\nInput file does not exist:\n{path}\n")

    file_size_mb = os.path.getsize(path) / (1024**2)

    print(f"File size               : {file_size_mb:.2f} MB")

    errors = []

    # -------------------------------------------------------------------------
    # METHOD 1
    # -------------------------------------------------------------------------

    print()
    print("Attempt 1: pandas C engine...")

    try:

        data = pd.read_csv(
            path,
            engine="c",
            low_memory=False,
            memory_map=False,
        )

        print("SUCCESS: C engine loaded the file.")

        return data

    except Exception as e:

        errors.append(("C engine", repr(e)))

        print(f"FAILED: {e}")

    # -------------------------------------------------------------------------
    # METHOD 2
    # -------------------------------------------------------------------------

    print()
    print("Attempt 2: pandas Python engine...")

    try:

        data = pd.read_csv(
            path,
            engine="python",
            low_memory=False,
        )

        print("SUCCESS: Python engine loaded the file.")

        return data

    except Exception as e:

        errors.append(("Python engine", repr(e)))

        print(f"FAILED: {e}")

    # -------------------------------------------------------------------------
    # METHOD 3
    # -------------------------------------------------------------------------
    #
    # Read the file as bytes first.
    #
    # This can avoid certain Windows/network/file-handle issues.
    # -------------------------------------------------------------------------

    print()
    print("Attempt 3: binary read + pandas Python parser...")

    try:

        with open(path, "rb") as f:

            raw = f.read()

        print(f"Binary read successful: " f"{len(raw) / (1024 ** 2):.2f} MB")

        text = raw.decode("utf-8-sig", errors="replace")

        from io import StringIO

        data = pd.read_csv(
            StringIO(text),
            engine="python",
        )

        print("SUCCESS: binary/text loader loaded the file.")

        return data

    except Exception as e:

        errors.append(("Binary loader", repr(e)))

        print(f"FAILED: {e}")

    # -------------------------------------------------------------------------
    # FINAL ERROR
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("ALL CSV LOADING METHODS FAILED")
    print("=" * 90)

    for method, error in errors:

        print()
        print(method)
        print(error)

    raise RuntimeError(
        "\nCould not read the CSV file.\n\n"
        "This usually means the CSV file is incomplete/corrupted or "
        "the filesystem is returning an I/O error.\n\n"
        f"File:\n{path}"
    )


df = load_gene_expression_csv(INFILE)


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


print()
print("=" * 90)
print("VALIDATING DATA STRUCTURE")
print("=" * 90)


missing = [column for column in META if column not in df.columns]


if missing:

    raise ValueError(
        "\nMissing required metadata columns:\n"
        + "\n".join(f"  - {x}" for x in missing)
    )


# =============================================================================
# REMOVE COMPLETELY EMPTY ROWS
# =============================================================================

before_rows = len(df)

df = df.dropna(how="all").reset_index(drop=True)

after_rows = len(df)

if before_rows != after_rows:

    print(f"Removed completely empty rows: " f"{before_rows - after_rows}")


# =============================================================================
# IDENTIFY GENES
# =============================================================================

genes = [column for column in df.columns if column not in META]


if len(genes) == 0:

    raise ValueError("No gene-expression columns were found.")


print(f"Samples                  : {len(df):,}")
print(f"Gene columns             : {len(genes):,}")


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

print()
print("Converting gene expression columns to numeric...")

df[genes] = df[genes].apply(
    pd.to_numeric,
    errors="coerce",
)


# =============================================================================
# REMOVE COMPLETELY INVALID GENES
# =============================================================================

gene_nan_fraction = df[genes].isna().mean()

invalid_genes = gene_nan_fraction[gene_nan_fraction >= 1.0].index.tolist()


if invalid_genes:

    print(f"Removing {len(invalid_genes):,} " "completely non-numeric genes.")

    df = df.drop(columns=invalid_genes)

    genes = [column for column in genes if column not in invalid_genes]


if len(genes) == 0:

    raise ValueError("No valid numeric gene-expression columns remain.")


# =============================================================================
# MISSING VALUES
# =============================================================================
#
# IMPORTANT:
#
# We do NOT use information from the test set to calculate imputation values.
#
# For simplicity, missing expression values are temporarily represented by
# zero before the fold-wise model.
#
# If your expression matrix uses a different missing-value convention,
# replace this with fold-wise median imputation.
# =============================================================================

missing_count = int(df[genes].isna().sum().sum())

if missing_count > 0:

    print(f"Missing gene values found: " f"{missing_count:,}")

    print("Replacing missing expression values with 0.")

    df[genes] = df[genes].fillna(0.0)


# =============================================================================
# LABEL
# =============================================================================

df["label"] = pd.to_numeric(
    df["label"],
    errors="raise",
).astype(int)


unique_labels = sorted(df["label"].unique())


if not set(unique_labels).issubset({0, 1}):

    raise ValueError("Label column must contain only 0 and 1.")


if len(unique_labels) != 2:

    raise ValueError("Both Alzheimer (1) and Control (0) " "classes are required.")


# =============================================================================
# CREATE X / y
# =============================================================================

X = df[genes].values.astype(np.float32)

y = df["label"].values.astype(np.int64)


print()
print("=" * 90)
print("DATA SUMMARY")
print("=" * 90)

print()
print(f"Samples                  : {len(df):,}")
print(f"Genes                    : {len(genes):,}")
print(f"Alzheimer                : {(y == 1).sum():,}")
print(f"Control                  : {(y == 0).sum():,}")
print(f"Datasets                 : " f"{df['dataset'].nunique():,}")


# =============================================================================
# CHECK CONSTANT GENES
# =============================================================================

print()
print("Checking constant genes...")

gene_variance = np.var(
    X,
    axis=0,
)

constant_mask = (~np.isfinite(gene_variance)) | (gene_variance <= 1e-12)

constant_count = int(constant_mask.sum())

if constant_count > 0:

    print(f"Removing {constant_count:,} " "constant/invalid genes.")

    X = X[:, ~constant_mask]

    genes = [
        gene
        for gene, keep in zip(
            genes,
            ~constant_mask,
        )
        if keep
    ]

else:

    print("No constant genes detected.")


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
#
# TEST DATA IS NEVER USED FOR GENE SELECTION.
# =============================================================================

print()
print("=" * 90)
print("CREATING INDEPENDENT TRAIN / TEST SPLIT")
print("=" * 90)


strata = df["dataset"].astype(str) + "_" + df["label"].astype(str)


indices = np.arange(len(df))


try:

    tr, te = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=strata,
    )

    split_method = "dataset + diagnosis stratification"


except ValueError as e:

    print()
    print("WARNING: dataset + diagnosis " "stratification failed.")

    print(f"Reason: {e}")

    print("Using diagnosis-only stratification.")

    tr, te = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    split_method = "diagnosis-only stratification"


Xtr = X[tr]

Xte = X[te]

ytr = y[tr]

yte = y[te]


print()
print(f"Split method             : {split_method}")
print(f"Training samples         : {len(tr):,}")
print(f"Independent test samples : {len(te):,}")


# =============================================================================
# VERIFY TEST SET
# =============================================================================

print()
print("Training class distribution:")
print(f"  Control    : " f"{(ytr == 0).sum():,}")

print(f"  Alzheimer  : " f"{(ytr == 1).sum():,}")

print()
print("Test class distribution:")
print(f"  Control    : " f"{(yte == 0).sum():,}")

print(f"  Alzheimer  : " f"{(yte == 1).sum():,}")


# =============================================================================
# EFFECTIVE K
# =============================================================================

K_EFFECTIVE = min(
    K,
    Xtr.shape[1],
)


print()
print(f"K-effective              : " f"{K_EFFECTIVE:,}")


# =============================================================================
# CUSTOM GR-TCN MODULE
# =============================================================================


class PyTorchGRTCNModule(nn.Module):
    """
    Custom Gene-Representation TCN-like prototype classifier.

    Components
    ----------
    1. Learnable gene gate
    2. Class prototypes
    3. Cosine similarity classifier
    4. L1 sparsity penalty
    5. Gene-correlation graph/Laplacian regularization

    The gene gate is explicitly learned during training.
    """

    def __init__(
        self,
        num_features,
        num_classes=2,
        temperature=0.5,
    ):

        super().__init__()

        self.num_features = num_features

        self.num_classes = num_classes

        self.temperature = temperature

        # ---------------------------------------------------------------------
        # Learnable gene-selection logits
        # ---------------------------------------------------------------------

        self.gene_logits = nn.Parameter(
            torch.zeros(
                num_features,
                dtype=torch.float32,
            )
        )

        # ---------------------------------------------------------------------
        # Class prototypes
        # ---------------------------------------------------------------------

        self.prototypes = nn.Parameter(
            torch.randn(
                num_classes,
                num_features,
                dtype=torch.float32,
            )
        )

    def forward(self, x):

        # ---------------------------------------------------------------------
        # Soft gene gate
        # ---------------------------------------------------------------------

        gene_gate = torch.sigmoid(self.gene_logits)

        # ---------------------------------------------------------------------
        # Gene-weighted expression
        # ---------------------------------------------------------------------

        x_gated = x * gene_gate

        proto_gated = self.prototypes * gene_gate

        # ---------------------------------------------------------------------
        # Normalize
        # ---------------------------------------------------------------------

        x_norm = F.normalize(
            x_gated,
            p=2,
            dim=1,
            eps=1e-8,
        )

        proto_norm = F.normalize(
            proto_gated,
            p=2,
            dim=1,
            eps=1e-8,
        )

        # ---------------------------------------------------------------------
        # Cosine similarity
        # ---------------------------------------------------------------------

        cosine_sim = torch.mm(
            x_norm,
            proto_norm.T,
        )

        # ---------------------------------------------------------------------
        # Temperature scaling
        # ---------------------------------------------------------------------

        logits = cosine_sim / self.temperature

        return logits, gene_gate


# =============================================================================
# CUSTOM GR-TCN CLASSIFIER
# =============================================================================


class PyTorchGRTCNClassifier:
    """
    Custom GR-TCN classifier.

    This is deliberately implemented independently rather than importing
    anything from Step 17.

    It exposes:

        fit()
        predict()
        predict_proba()
        get_gene_importance()
    """

    def __init__(
        self,
        epochs=80,
        lr=0.005,
        l1_lambda=1e-3,
        laplacian_lambda=1e-3,
        temperature=0.5,
        batch_size=64,
        random_state=42,
        device=None,
    ):

        self.epochs = epochs


        self.lr = lr

        self.l1_lambda = l1_lambda

        self.laplacian_lambda = laplacian_lambda

        self.temperature = temperature

        self.batch_size = batch_size

        self.random_state = random_state

        self.device = device if device is not None else DEVICE

    # =========================================================================
    # LAPLACIAN
    # =========================================================================

    def _compute_laplacian(self, X):

        # ---------------------------------------------------------------------
        # Correlation matrix
        # ---------------------------------------------------------------------

        corr = np.corrcoef(
            X,
            rowvar=False,
        )

        # ---------------------------------------------------------------------
        # Replace numerical problems
        # ---------------------------------------------------------------------

        corr = np.nan_to_num(
            corr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------------------
        # Absolute gene-gene relationship
        # ---------------------------------------------------------------------

        corr = np.abs(corr)

        # ---------------------------------------------------------------------
        # Remove self relationships
        # ---------------------------------------------------------------------

        np.fill_diagonal(
            corr,
            0.0,
        )

        # ---------------------------------------------------------------------
        # Degree matrix
        # ---------------------------------------------------------------------

        degree_values = np.sum(
            corr,
            axis=1,
        )

        degree = np.diag(degree_values)

        # ---------------------------------------------------------------------
        # Graph Laplacian
        # ---------------------------------------------------------------------

        laplacian = degree - corr

        # ---------------------------------------------------------------------
        # Numerical stabilization
        # ---------------------------------------------------------------------

        laplacian = (laplacian + laplacian.T) / 2.0

        return torch.tensor(
            laplacian,
            dtype=torch.float32,
            device=self.device,
        )

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(self, X, y):

        np.random.seed(self.random_state)

        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(self.random_state)

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.int64,
        )

        self.classes_ = np.array(
            [0, 1],
            dtype=np.int64,
        )

        # ---------------------------------------------------------------------
        # Tensor conversion
        # ---------------------------------------------------------------------

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.long,
            device=self.device,
        )

        # ---------------------------------------------------------------------
        # Number of features
        # ---------------------------------------------------------------------

        num_features = X.shape[1]

        # ---------------------------------------------------------------------
        # Graph regularization
        # ---------------------------------------------------------------------

        L_matrix = self._compute_laplacian(X)

        # ---------------------------------------------------------------------
        # Create model
        # ---------------------------------------------------------------------

        self.model_ = PyTorchGRTCNModule(
            num_features=num_features,
            num_classes=2,
            temperature=self.temperature,
        ).to(self.device)

        # ---------------------------------------------------------------------
        # Initialize class prototypes using training-class means
        # ---------------------------------------------------------------------

        with torch.no_grad():

            for class_index, class_value in enumerate(self.classes_):

                class_mask = y_tensor == class_value

                if class_mask.sum() == 0:

                    raise ValueError(f"Class {class_value} " "has no samples.")

                class_mean = X_tensor[class_mask].mean(dim=0)

                self.model_.prototypes[class_index].copy_(class_mean)

        # ---------------------------------------------------------------------
        # Optimizer
        # ---------------------------------------------------------------------

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
        )

        # ---------------------------------------------------------------------
        # Loss
        # ---------------------------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------------------
        # Tensor dataset
        # ---------------------------------------------------------------------

        dataset = torch.utils.data.TensorDataset(
            X_tensor,
            y_tensor,
        )

        # ---------------------------------------------------------------------
        # DataLoader
        # ---------------------------------------------------------------------

        generator = torch.Generator()

        generator.manual_seed(self.random_state)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------

        self.model_.train()

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:

                optimizer.zero_grad(set_to_none=True)

                logits, gene_gate = self.model_(batch_x)

                # -------------------------------------------------------------
                # Classification loss
                # -------------------------------------------------------------

                ce_loss = criterion(
                    logits,
                    batch_y,
                )

                # -------------------------------------------------------------
                # Sparsity penalty
                # -------------------------------------------------------------

                l1_penalty = self.l1_lambda * torch.sum(gene_gate)

                # -------------------------------------------------------------
                # Graph smoothness penalty
                #
                # g^T L g
                # -------------------------------------------------------------

                graph_term = torch.matmul(
                    gene_gate.unsqueeze(0),
                    torch.matmul(
                        L_matrix,
                        gene_gate.unsqueeze(1),
                    ),
                ).squeeze()

                laplacian_penalty = self.laplacian_lambda * graph_term

                # -------------------------------------------------------------
                # Total loss
                # -------------------------------------------------------------

                total_loss = ce_loss + l1_penalty + laplacian_penalty

                # -------------------------------------------------------------
                # Backpropagation
                # -------------------------------------------------------------

                total_loss.backward()

                # -------------------------------------------------------------
                # Gradient clipping
                # -------------------------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                epoch_loss += float(total_loss.detach().cpu().item())

            # -----------------------------------------------------------------
            # Progress
            # -----------------------------------------------------------------

            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:

                avg_loss = epoch_loss / max(
                    len(dataloader),
                    1,
                )

                print(
                    f"      Epoch "
                    f"{epoch + 1:3d}/"
                    f"{self.epochs} | "
                    f"loss={avg_loss:.6f}"
                )

        # ---------------------------------------------------------------------
        # Final learned gene importance
        # ---------------------------------------------------------------------

        self.model_.eval()

        with torch.no_grad():

            gene_gate = torch.sigmoid(self.model_.gene_logits)

            self.feature_importances_ = (
                gene_gate.detach().cpu().numpy().astype(np.float64)
            )

        return self

    # =========================================================================
    # PREDICT PROBA
    # =========================================================================

    def predict_proba(self, X):

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        self.model_.eval()

        with torch.no_grad():

            logits, _ = self.model_(X_tensor)

            probabilities = (
                F.softmax(
                    logits,
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )

        return probabilities

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return self.classes_[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]

    # =========================================================================
    # GENE IMPORTANCE
    # =========================================================================

    def get_gene_importance(self):

        self.model_.eval()

        with torch.no_grad():

            importance = torch.sigmoid(self.model_.gene_logits).detach().cpu().numpy()

        return importance


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

print()
print("=" * 90)
print("STARTING LEAKAGE-SAFE CROSS-VALIDATION")
print("=" * 90)

print()
print("IMPORTANT: feature selection occurs separately " "inside every training fold.")

print("The independent test set is NOT used.")


cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


# =============================================================================
# GLOBAL STORAGE
# =============================================================================

gene_importance = {}

gene_fold_presence = {}

fold_results = []

selected_gene_frequency = {}


# =============================================================================
# CROSS-VALIDATION LOOP
# =============================================================================

for fold, (
    train_idx,
    valid_idx,
) in enumerate(
    cv.split(Xtr, ytr),
    start=1,
):

    print()
    print()
    print("=" * 90)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 90)

    # =========================================================================
    # FOLD DATA
    # =========================================================================

    X_train_fold = Xtr[train_idx]

    y_train_fold = ytr[train_idx]

    X_valid_fold = Xtr[valid_idx]

    y_valid_fold = ytr[valid_idx]

    print()
    print(f"Training fold samples    : " f"{len(train_idx):,}")

    print(f"Validation fold samples  : " f"{len(valid_idx):,}")

    # =========================================================================
    # FEATURE SELECTION
    # =========================================================================
    #
    # VERY IMPORTANT:
    #
    # SelectKBest is fitted ONLY on X_train_fold/y_train_fold.
    #
    # Therefore the validation fold cannot influence gene selection.
    # =========================================================================

    print()
    print(f"Selecting top " f"{K_EFFECTIVE:,} genes " "using training fold only...")

    selector = SelectKBest(
        score_func=f_classif,
        k=K_EFFECTIVE,
    )

    X_train_selected = selector.fit_transform(
        X_train_fold,
        y_train_fold,
    )

    X_valid_selected = selector.transform(X_valid_fold)

    selected_indices = np.where(selector.get_support())[0]

    selected_genes = [genes[i] for i in selected_indices]

    print(f"Genes selected            : " f"{len(selected_genes):,}")

    # =========================================================================
    # RECORD SELECTION FREQUENCY
    # =========================================================================

    for gene in selected_genes:

        selected_gene_frequency[gene] = (
            selected_gene_frequency.get(
                gene,
                0,
            )
            + 1
        )

    # =========================================================================
    # STANDARDIZATION
    # =========================================================================
    #
    # Again, scaler is fitted ONLY on training fold.
    # =========================================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected)

    X_valid_scaled = scaler.transform(X_valid_selected)

    # =========================================================================
    # CUSTOM GR-TCN
    # =========================================================================

    print()
    print("Training custom GR-TCN...")

    clf = PyTorchGRTCNClassifier(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        l1_lambda=L1_LAMBDA,
        laplacian_lambda=LAPLACIAN_LAMBDA,
        temperature=TEMPERATURE,
        batch_size=BATCH_SIZE,
        random_state=(SEED + fold),
        device=DEVICE,
    )

    clf.fit(
        X_train_scaled,
        y_train_fold,
    )

    # =========================================================================
    # VALIDATION AUC
    # =========================================================================

    probabilities = clf.predict_proba(X_valid_scaled)[:, 1]

    auc = roc_auc_score(
        y_valid_fold,
        probabilities,
    )

    fold_results.append(
        {
            "fold": fold,
            "auc": float(auc),
        }
    )

    print()
    print(f"Validation ROC-AUC      : " f"{auc:.6f}")

    # =========================================================================
    # CUSTOM GR-TCN LEARNED GENE IMPORTANCE
    # =========================================================================

    learned_importance = clf.get_gene_importance()

    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================
    #
    # IMPORTANT:
    #
    # Permutation is performed on the validation fold.
    #
    # The model has never trained on the validation observations.
    # =========================================================================

    print()
    print("Calculating validation permutation importance...")

    try:

        perm = permutation_importance(
            clf,
            X_valid_scaled,
            y_valid_fold,
            scoring="roc_auc",
            n_repeats=PERM_REPEATS,
            random_state=(SEED + fold),
            n_jobs=1,
        )

        permutation_mean = perm.importances_mean

        permutation_std = perm.importances_std

    except Exception as e:

        print()
        print("WARNING: permutation importance " "failed.")

        print(f"Reason: {e}")

        # -------------------------------------------------------------
        # Fall back to learned GR-TCN gene gate.
        # -------------------------------------------------------------

        permutation_mean = learned_importance - np.mean(learned_importance)

        permutation_std = np.zeros(len(learned_importance))

    # =========================================================================
    # COMBINE LEARNED GATE + PERMUTATION
    # =========================================================================
    #
    # We retain BOTH:
    #
    #   learned_gate
    #   permutation_importance
    #
    # This is particularly useful for a custom neural gene-selection model.
    # =========================================================================

    for local_idx, gene in enumerate(selected_genes):

        gate_value = float(learned_importance[local_idx])

        perm_value = float(permutation_mean[local_idx])

        perm_std = float(permutation_std[local_idx])

        if gene not in gene_importance:

            gene_importance[gene] = {
                "permutation": [],
                "gate": [],
                "combined": [],
                "fold": [],
            }

        # ---------------------------------------------------------------------
        # Store values
        # ---------------------------------------------------------------------

        gene_importance[gene]["permutation"].append(perm_value)

        gene_importance[gene]["gate"].append(gate_value)

        # ---------------------------------------------------------------------
        # Combined score
        #
        # Permutation importance is centered around zero.
        #
        # Convert gate to a relative rank-like quantity before combining.
        # ---------------------------------------------------------------------

        gene_importance[gene]["combined"].append(perm_value)

        gene_importance[gene]["fold"].append(fold)

        gene_fold_presence[gene] = (
            gene_fold_presence.get(
                gene,
                0,
            )
            + 1
        )

    # =========================================================================
    # TOP GENES FOR THIS FOLD
    # =========================================================================

    fold_gene_table = []

    for local_idx, gene in enumerate(selected_genes):

        fold_gene_table.append(
            {
                "gene": gene,
                "permutation_importance": (float(permutation_mean[local_idx])),
                "permutation_std": (float(permutation_std[local_idx])),
                "learned_gate": (float(learned_importance[local_idx])),
            }
        )

    fold_gene_df = pd.DataFrame(fold_gene_table)

    fold_gene_df = fold_gene_df.sort_values(
        "permutation_importance",
        ascending=False,
    ).reset_index(drop=True)

    print()
    print(f"TOP 10 GENES - FOLD {fold}")

    print("-" * 90)

    for rank, row in enumerate(
        fold_gene_df.head(10).itertuples(index=False),
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{row.gene:<25} "
            f"perm={row.permutation_importance: .6f} | "
            f"gate={row.learned_gate:.6f}"
        )


# =============================================================================
# CV SUMMARY
# =============================================================================

fold_auc_values = np.array(
    [x["auc"] for x in fold_results],
    dtype=float,
)


print()
print()
print("=" * 90)
print("CROSS-VALIDATION PERFORMANCE")
print("=" * 90)

for result in fold_results:

    print(f"Fold {result['fold']}: " f"ROC-AUC = " f"{result['auc']:.6f}")


print()
print(f"Mean CV ROC-AUC         : " f"{np.mean(fold_auc_values):.6f}")

print(f"Std CV ROC-AUC          : " f"{np.std(fold_auc_values):.6f}")


# =============================================================================
# BUILD GENE-LEVEL RESULTS
# =============================================================================

print()
print()
print("=" * 90)
print("AGGREGATING GENE IMPORTANCE ACROSS FOLDS")
print("=" * 90)


gene_rows = []


for gene, values in gene_importance.items():

    permutation_values = np.asarray(
        values["permutation"],
        dtype=float,
    )

    gate_values = np.asarray(
        values["gate"],
        dtype=float,
    )

    combined_values = np.asarray(
        values["combined"],
        dtype=float,
    )

    gene_rows.append(
        {
            "gene": gene,
            "folds_present": len(permutation_values),
            "selection_frequency": (
                selected_gene_frequency.get(
                    gene,
                    0,
                )
                / N_SPLITS
            ),
            "mean_permutation_importance": (np.nanmean(permutation_values)),
            "std_permutation_importance": (np.nanstd(permutation_values)),
            "mean_permutation_std": (
                np.nanmean(np.asarray([0.0 for _ in permutation_values]))
            ),
            "mean_learned_gate": (np.nanmean(gate_values)),
            "std_learned_gate": (np.nanstd(gate_values)),
            "mean_combined_importance": (np.nanmean(combined_values)),
        }
    )


gene_df = pd.DataFrame(gene_rows)


if gene_df.empty:

    raise RuntimeError("No genes produced importance results.")


# =============================================================================
# RANK GENES BY PERMUTATION IMPORTANCE
# =============================================================================

gene_df = gene_df.sort_values(
    "mean_permutation_importance",
    ascending=False,
).reset_index(drop=True)


gene_df["importance_rank"] = np.arange(len(gene_df)) + 1


# =============================================================================
# NORMALIZED IMPORTANCE SCORE
# =============================================================================

# -------------------------------------------------------------------------
# Positive permutation importance
# -------------------------------------------------------------------------

positive_importance = gene_df["mean_permutation_importance"].clip(lower=0)


max_positive_importance = positive_importance.max()


if not np.isfinite(max_positive_importance) or max_positive_importance <= 0:

    max_positive_importance = 1.0


gene_df["importance_score"] = positive_importance / max_positive_importance


# =============================================================================
# RANK SCORE
# =============================================================================

gene_count = max(
    len(gene_df),
    1,
)


gene_df["rank_score"] = 1.0 - (gene_df["importance_rank"] - 1) / gene_count


gene_df["rank_score"] = gene_df["rank_score"].clip(
    lower=0.0,
    upper=1.0,
)


# =============================================================================
# FINAL CUSTOM GENE CONSENSUS SCORE
# =============================================================================
#
# Because we currently have ONE custom model, this is not pretending that
# several unrelated ML models agree.
#
# Instead:
#
#   50% permutation importance
#   30% fold selection stability
#   20% learned GR-TCN gate
#
# This is more statistically honest for your current custom algorithm.
# =============================================================================

gene_df["consensus_score"] = (
    0.50 * gene_df["importance_score"]
    + 0.30 * gene_df["selection_frequency"]
    + 0.20 * gene_df["mean_learned_gate"]
)


# =============================================================================
# FINAL RANKING
# =============================================================================

gene_df = gene_df.sort_values(
    [
        "consensus_score",
        "selection_frequency",
        "mean_permutation_importance",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)


gene_df["consensus_rank"] = np.arange(len(gene_df)) + 1


# =============================================================================
# FINAL TOP 10
# =============================================================================

top10 = gene_df.head(N_TOP_GENES).copy()


# =============================================================================
# PRINT FINAL RESULTS
# =============================================================================

print()
print()
print("=" * 90)
print("FINAL TOP-10 GENES")
print("=" * 90)

print()

print(
    f"{'Rank':<6}"
    f"{'Gene':<25}"
    f"{'Score':<12}"
    f"{'Selection':<12}"
    f"{'Permutation':<15}"
    f"{'Gate':<12}"
)

print("-" * 90)


for _, row in top10.iterrows():

    print(
        f"{int(row['consensus_rank']):<6}"
        f"{row['gene']:<25}"
        f"{row['consensus_score']:<12.6f}"
        f"{row['selection_frequency']:<12.2f}"
        f"{row['mean_permutation_importance']:<15.6f}"
        f"{row['mean_learned_gate']:<12.6f}"
    )


# =============================================================================
# TOP 10 SIMPLE LIST
# =============================================================================

print()
print()
print("=" * 90)
print("TOP-10 GENE LIST")
print("=" * 90)

print()


top_genes = top10["gene"].tolist()


for rank, gene in enumerate(
    top_genes,
    start=1,
):

    print(f"{rank:2d}. {gene}")


# =============================================================================
# GENE STABILITY
# =============================================================================

print()
print()
print("=" * 90)
print("GENE SELECTION STABILITY")
print("=" * 90)

print()

for _, row in top10.iterrows():

    frequency = row["selection_frequency"] * 100

    print(f"{row['gene']:<25} " f"selected in " f"{frequency:.1f}% of CV folds")


# =============================================================================
# LEARNED GENE GATES
# =============================================================================

print()
print()
print("=" * 90)
print("GR-TCN LEARNED GENE GATES")
print("=" * 90)

print()

for _, row in top10.iterrows():

    print(f"{row['gene']:<25} " f"gate={row['mean_learned_gate']:.6f}")


# =============================================================================
# IMPORTANT WARNING ABOUT SINGLE-MODEL CONSENSUS
# =============================================================================

print()
print()
print("=" * 90)
print("METHOD NOTE")
print("=" * 90)

print()
print("Current configuration contains ONE custom model: GR-TCN.")

print("Therefore this result is a fold-stable GR-TCN gene ranking,")

print("not a multi-model consensus ranking.")

print()
print("The ranking combines:")

print("  1. Validation permutation importance")

print("  2. Cross-validation selection frequency")

print("  3. Learned GR-TCN gene gate")


# =============================================================================
# OPTIONAL IN-MEMORY TEST EVALUATION
# =============================================================================
#
# IMPORTANT:
#
# This section does NOT influence gene selection.
#
# We first determine the top genes using TRAINING CV only.
#
# Then we train a completely new GR-TCN using those top genes on the entire
# training set and evaluate ONCE on the untouched independent test set.
#
# Nothing is saved.
# =============================================================================

print()
print()
print("=" * 90)
print("FINAL INDEPENDENT TEST EVALUATION")
print("=" * 90)

print()
print("The test set has not been used for feature selection.")


# =============================================================================
# MAP TOP GENES TO ORIGINAL FEATURE INDICES
# =============================================================================

gene_to_index = {gene: idx for idx, gene in enumerate(genes)}


top_gene_indices = [gene_to_index[gene] for gene in top_genes]


# =============================================================================
# EXTRACT TOP-10 TRAIN / TEST DATA
# =============================================================================

X_train_top10 = Xtr[
    :,
    top_gene_indices,
]

X_test_top10 = Xte[
    :,
    top_gene_indices,
]


# =============================================================================
# FINAL SCALER
# =============================================================================
#
# Fit ONLY on complete training data.
# Test data is only transformed.
# =============================================================================

final_scaler = StandardScaler()

X_train_top10_scaled = final_scaler.fit_transform(X_train_top10)

X_test_top10_scaled = final_scaler.transform(X_test_top10)


# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

print()
print("Training final GR-TCN on " "training set using top-10 genes...")

final_model = PyTorchGRTCNClassifier(
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    l1_lambda=L1_LAMBDA,
    laplacian_lambda=LAPLACIAN_LAMBDA,
    temperature=TEMPERATURE,
    batch_size=BATCH_SIZE,
    random_state=SEED,
    device=DEVICE,
)


final_model.fit(
    X_train_top10_scaled,
    ytr,
)


# =============================================================================
# TEST PREDICTIONS
# =============================================================================

test_probabilities = final_model.predict_proba(X_test_top10_scaled)[:, 1]


test_auc = roc_auc_score(
    yte,
    test_probabilities,
)


# =============================================================================
# TEST RESULT
# =============================================================================

print()
print("=" * 90)
print("INDEPENDENT TEST RESULT")
print("=" * 90)

print()

print(f"Independent test ROC-AUC : " f"{test_auc:.6f}")

print(f"Number of test samples   : " f"{len(yte):,}")

print(f"Genes used               : " f"{len(top_genes)}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print()
print("=" * 90)
print("STEP 18 COMPLETE")
print("=" * 90)

print()

print(f"Original genes           : " f"{len(genes):,}")

print(f"Genes selected/fold      : " f"{K_EFFECTIVE:,}")

print(f"CV folds                 : " f"{N_SPLITS}")

print(f"Mean CV ROC-AUC          : " f"{np.mean(fold_auc_values):.6f}")

print(f"CV ROC-AUC std           : " f"{np.std(fold_auc_values):.6f}")

print(f"Final independent AUC    : " f"{test_auc:.6f}")

print()

print("FINAL TOP-10 GENES")
print("-" * 90)

for rank, gene in enumerate(
    top_genes,
    start=1,
):

    print(f"{rank:2d}. {gene}")


print()
print("=" * 90)
print("NO FILES WERE WRITTEN.")
print("ALL RESULTS EXIST ONLY IN MEMORY AND ARE PRINTED ABOVE.")
print("=" * 90)
