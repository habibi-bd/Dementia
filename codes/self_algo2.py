# """
# ===============================================================================
# STEP 18 - CUSTOM DEEP-LEARNING GENE SELECTION
# ===============================================================================

# Algorithm
# ---------
# SGAPC
# Sparse Gene-Attention Prototype Classifier

# Purpose
# -------
# Learn biologically meaningful and predictive genes directly from high-
# dimensional gene-expression data.

# Key properties
# --------------
# 1. No SelectKBest pre-filtering.
# 2. All genes enter the model.
# 3. Gene importance is learned end-to-end.
# 4. Sparse gene attention encourages a compact gene signature.
# 5. Correlation regularization encourages co-expressed genes to behave
#    coherently.
# 6. Prototype-based classification separates AD and Control.
# 7. Gene selection occurs independently inside every CV training fold.
# 8. The independent test set is NEVER used for gene selection.
# 9. Nothing is saved to disk.

# Input
# -----
# D:/Dementia2/codes/ml_data/integrated_gene_expression.csv

# Output
# ------
# Printed to console only.

# ===============================================================================
# """

# # =============================================================================
# # IMPORTS
# # =============================================================================

# import os
# import warnings

# warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from sklearn.base import BaseEstimator, ClassifierMixin, clone
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# BASE = r"D:\Dementia2"

# INFILE = os.path.join(
#     BASE,
#     "codes",
#     "ml_data",
#     "integrated_gene_expression.csv",
# )


# SEED = 42

# TEST_SIZE = 0.20

# N_SPLITS = 5

# EPOCHS = 100

# LEARNING_RATE = 0.005

# BATCH_SIZE = 64

# TEMPERATURE = 0.5


# # -----------------------------------------------------------------------------
# # Sparse gene selection
# # -----------------------------------------------------------------------------

# SPARSITY_LAMBDA = 1e-3


# # -----------------------------------------------------------------------------
# # Correlation regularization
# # -----------------------------------------------------------------------------

# CORRELATION_LAMBDA = 1e-3


# # -----------------------------------------------------------------------------
# # Prototype separation
# # -----------------------------------------------------------------------------

# SEPARATION_LAMBDA = 1e-2


# # -----------------------------------------------------------------------------
# # Number of final genes
# # -----------------------------------------------------------------------------

# N_TOP_GENES = 10


# # -----------------------------------------------------------------------------
# # Attention threshold
# # -----------------------------------------------------------------------------

# GENE_THRESHOLD = 0.50


# # -----------------------------------------------------------------------------
# # Maximum number of genes used to construct correlation matrix
# #
# # For ~20,000 genes, a complete 20,000 × 20,000 matrix requires enormous
# # memory. Therefore we construct the correlation graph only among genes
# # having sufficient preliminary variance.
# # -----------------------------------------------------------------------------

# MAX_CORR_GENES = 3000


# # =============================================================================
# # REPRODUCIBILITY
# # =============================================================================

# np.random.seed(SEED)
# torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # =============================================================================
# # HEADER
# # =============================================================================

# print("=" * 80)
# print("STEP 18 - SGAPC")
# print("SPARSE GENE-ATTENTION PROTOTYPE CLASSIFIER")
# print("=" * 80)

# print()

# print(f"Device                  : {DEVICE}")
# print(f"Input                   : {INFILE}")
# print(f"Seed                    : {SEED}")
# print(f"Test size               : {TEST_SIZE}")
# print(f"CV folds                : {N_SPLITS}")
# print(f"Epochs                  : {EPOCHS}")
# print(f"Learning rate           : {LEARNING_RATE}")
# print(f"Batch size              : {BATCH_SIZE}")
# print(f"Sparsity lambda         : {SPARSITY_LAMBDA}")
# print(f"Correlation lambda      : {CORRELATION_LAMBDA}")
# print(f"Separation lambda       : {SEPARATION_LAMBDA}")
# print(f"Final genes             : {N_TOP_GENES}")


# # =============================================================================
# # LOAD DATA
# # =============================================================================

# print()
# print("=" * 80)
# print("LOADING DATA")
# print("=" * 80)


# if not os.path.exists(INFILE):

#     raise FileNotFoundError(f"\nInput file not found:\n{INFILE}")


# # -----------------------------------------------------------------------------
# # Robust CSV loading
# #
# # Your previous error:
# #
# # pandas.errors.ParserError:
# # Error tokenizing data.
# #
# # We first try the fast C parser. If the file has malformed rows or quoting,
# # we fall back to the Python parser.
# # -----------------------------------------------------------------------------

# try:

#     df = pd.read_csv(
#         INFILE,
#         engine="c",
#         low_memory=False,
#     )

# except (pd.errors.ParserError, OSError) as first_error:

#     print()
#     print("WARNING: Standard CSV parser failed.")
#     print("Trying robust Python CSV parser...")

#     try:

#         df = pd.read_csv(
#             INFILE,
#             engine="python",
#             on_bad_lines="warn",
#         )

#     except Exception as second_error:

#         raise RuntimeError(
#             "\nUnable to read the integrated expression file.\n\n"
#             f"File:\n{INFILE}\n\n"
#             f"First parser error:\n{first_error}\n\n"
#             f"Python parser error:\n{second_error}\n"
#         )


# print()

# print(f"Loaded dataframe: " f"{df.shape[0]:,} samples × {df.shape[1]:,} columns")


# # =============================================================================
# # REQUIRED METADATA
# # =============================================================================

# META = [
#     "sample_id",
#     "condition",
#     "label",
#     "dataset",
#     "platform",
# ]


# missing = [column for column in META if column not in df.columns]


# if missing:

#     raise ValueError(
#         "Missing required metadata columns:\n" + "\n".join(f"  - {x}" for x in missing)
#     )


# # =============================================================================
# # IDENTIFY GENES
# # =============================================================================

# genes = [column for column in df.columns if column not in META]


# if len(genes) == 0:

#     raise ValueError("No gene-expression columns were detected.")


# print()
# print(f"Genes detected: {len(genes):,}")


# # =============================================================================
# # NUMERIC CONVERSION
# # =============================================================================

# print()
# print("Converting gene-expression values to numeric...")


# df[genes] = df[genes].apply(pd.to_numeric, errors="coerce")


# # -----------------------------------------------------------------------------
# # Remove completely invalid genes
# # -----------------------------------------------------------------------------

# valid_gene_mask = df[genes].notna().any(axis=0).values


# genes = [gene for gene, valid in zip(genes, valid_gene_mask) if valid]


# # -----------------------------------------------------------------------------
# # Fill remaining missing expression values
# #
# # IMPORTANT:
# # Missing values are temporarily filled here only to make the matrix usable.
# # The major ML leakage prevention still occurs because model normalization
# # and correlation structure are calculated using training folds only.
# # -----------------------------------------------------------------------------

# df[genes] = df[genes].fillna(0.0)


# # =============================================================================
# # LABEL
# # =============================================================================

# df["label"] = pd.to_numeric(
#     df["label"],
#     errors="raise",
# ).astype(int)


# y = df["label"].values


# unique_labels = sorted(np.unique(y))


# if not set(unique_labels).issubset({0, 1}):

#     raise ValueError("Label column must contain only 0 and 1.")


# if len(unique_labels) != 2:

#     raise ValueError("Both Alzheimer (1) and Control (0) classes are required.")


# # =============================================================================
# # CREATE EXPRESSION MATRIX
# # =============================================================================

# X = df[genes].values.astype(np.float32)


# print()
# print("=" * 80)
# print("DATA SUMMARY")
# print("=" * 80)

# print()

# print(f"Samples                 : {len(df):,}")
# print(f"Genes                   : {len(genes):,}")
# print(f"Alzheimer               : {(y == 1).sum():,}")
# print(f"Control                 : {(y == 0).sum():,}")
# print(f"Datasets                : " f"{df['dataset'].nunique():,}")


# # =============================================================================
# # TRAIN / TEST SPLIT
# # =============================================================================

# print()
# print("=" * 80)
# print("CREATING INDEPENDENT TEST SET")
# print("=" * 80)


# indices = np.arange(len(df))


# strata = df["dataset"].astype(str) + "_" + df["label"].astype(str)


# try:

#     tr, te = train_test_split(
#         indices,
#         test_size=TEST_SIZE,
#         random_state=SEED,
#         stratify=strata,
#     )

#     split_method = "dataset + diagnosis stratification"


# except ValueError:

#     print("WARNING: dataset + diagnosis " "stratification failed.")

#     print("Using diagnosis-only stratification.")

#     tr, te = train_test_split(
#         indices,
#         test_size=TEST_SIZE,
#         random_state=SEED,
#         stratify=y,
#     )

#     split_method = "diagnosis stratification"


# Xtr = X[tr]

# Xte = X[te]

# ytr = y[tr]

# yte = y[te]


# print()

# print(f"Split method            : {split_method}")
# print(f"Training samples        : {len(tr):,}")
# print(f"Independent test        : {len(te):,}")


# # =============================================================================
# # CORRELATION GRAPH
# # =============================================================================


# def build_gene_correlation_matrix(
#     X_train,
#     max_genes=MAX_CORR_GENES,
# ):
#     """
#     Construct a gene-gene correlation graph using ONLY the training fold.

#     This is deliberately calculated inside each CV fold.

#     Returns
#     -------
#     selected_indices
#         Original gene indices represented in the graph.

#     adjacency
#         Normalized absolute correlation matrix.
#     """

#     n_samples, n_genes = X_train.shape

#     if n_genes <= max_genes:

#         selected_indices = np.arange(n_genes)

#     else:

#         # ---------------------------------------------------------------------
#         # Select highest-variance genes ONLY from the training fold.
#         #
#         # This is not used for final gene selection.
#         # It is only a computational approximation for constructing the
#         # correlation graph.
#         # ---------------------------------------------------------------------

#         variances = np.var(
#             X_train,
#             axis=0,
#         )

#         selected_indices = np.argsort(variances)[-max_genes:]

#     X_graph = X_train[:, selected_indices]

#     # -------------------------------------------------------------------------
#     # Correlation
#     # -------------------------------------------------------------------------

#     corr = np.corrcoef(
#         X_graph,
#         rowvar=False,
#     )

#     corr = np.nan_to_num(
#         corr,
#         nan=0.0,
#         posinf=0.0,
#         neginf=0.0,
#     )

#     corr = np.abs(corr)

#     np.fill_diagonal(
#         corr,
#         0.0,
#     )

#     # -------------------------------------------------------------------------
#     # Normalize adjacency
#     # -------------------------------------------------------------------------

#     degree = np.sum(
#         corr,
#         axis=1,
#     )

#     degree = np.maximum(
#         degree,
#         1e-8,
#     )

#     D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

#     adjacency = D_inv_sqrt @ corr @ D_inv_sqrt

#     return (selected_indices, adjacency.astype(np.float32))


# # =============================================================================
# # CUSTOM NEURAL NETWORK
# # =============================================================================


# class SGAPCNetwork(nn.Module):
#     """
#     Sparse Gene-Attention Prototype Classifier.

#     Architecture
#     ------------

#     X
#      |
#      v
#     gene attention
#      |
#      v
#     sparse gene representation
#      |
#      +--------------------+
#      |                    |
#      v                    v
#     AD prototype     Control prototype
#      |                    |
#      +---------+----------+
#                |
#                v
#         cosine similarity
#                |
#                v
#              logits

#     Learnable parameters
#     --------------------
#     1. gene_logits
#     2. class prototypes
#     """

#     def __init__(
#         self,
#         num_features,
#         num_classes=2,
#         temperature=0.5,
#     ):

#         super().__init__()

#         self.num_features = num_features

#         self.num_classes = num_classes

#         self.temperature = temperature

#         # ---------------------------------------------------------------------
#         # Gene-specific attention logits
#         #
#         # Every gene gets its own learnable parameter.
#         # ---------------------------------------------------------------------

#         self.gene_logits = nn.Parameter(torch.zeros(num_features))

#         # ---------------------------------------------------------------------
#         # Class prototypes
#         # ---------------------------------------------------------------------

#         self.prototypes = nn.Parameter(torch.randn(num_classes, num_features) * 0.02)

#     def gene_attention(self):
#         """
#         Convert unconstrained gene logits to [0, 1].
#         """

#         return torch.sigmoid(self.gene_logits)

#     def forward(self, X):

#         attention = self.gene_attention()

#         # ---------------------------------------------------------------------
#         # Attention-weighted gene expression
#         # ---------------------------------------------------------------------

#         X_weighted = X * attention

#         # ---------------------------------------------------------------------
#         # Attention-weighted prototypes
#         # ---------------------------------------------------------------------

#         prototypes_weighted = self.prototypes * attention.unsqueeze(0)

#         # ---------------------------------------------------------------------
#         # Normalize
#         # ---------------------------------------------------------------------

#         X_norm = F.normalize(
#             X_weighted,
#             p=2,
#             dim=1,
#         )

#         prototype_norm = F.normalize(
#             prototypes_weighted,
#             p=2,
#             dim=1,
#         )

#         # ---------------------------------------------------------------------
#         # Cosine similarity
#         # ---------------------------------------------------------------------

#         similarity = torch.mm(
#             X_norm,
#             prototype_norm.T,
#         )

#         logits = similarity / self.temperature

#         return (
#             logits,
#             attention,
#         )


# # =============================================================================
# # SKLEARN WRAPPER
# # =============================================================================


# class SGAPCClassifier(
#     BaseEstimator,
#     ClassifierMixin,
# ):
#     """
#     Scikit-learn compatible implementation of SGAPC.
#     """

#     _estimator_type = "classifier"

#     def __init__(
#         self,
#         epochs=100,
#         lr=0.005,
#         batch_size=64,
#         temperature=0.5,
#         sparsity_lambda=1e-3,
#         correlation_lambda=1e-3,
#         separation_lambda=1e-2,
#         random_state=42,
#         correlation_indices=None,
#         correlation_matrix=None,
#     ):

#         self.epochs = epochs

#         self.lr = lr

#         self.batch_size = batch_size

#         self.temperature = temperature

#         self.sparsity_lambda = sparsity_lambda

#         self.correlation_lambda = correlation_lambda

#         self.separation_lambda = separation_lambda

#         self.random_state = random_state

#         self.correlation_indices = correlation_indices

#         self.correlation_matrix = correlation_matrix

#     # =========================================================================
#     # FIT
#     # =========================================================================

#     def fit(self, X, y):

#         X = np.asarray(
#             X,
#             dtype=np.float32,
#         )

#         y = np.asarray(
#             y,
#             dtype=np.int64,
#         )

#         # ---------------------------------------------------------------------
#         # Reproducibility
#         # ---------------------------------------------------------------------

#         np.random.seed(self.random_state)

#         torch.manual_seed(self.random_state)

#         if torch.cuda.is_available():

#             torch.cuda.manual_seed_all(self.random_state)

#         # ---------------------------------------------------------------------
#         # Classes
#         # ---------------------------------------------------------------------

#         self.classes_ = np.array(sorted(np.unique(y)))

#         if len(self.classes_) != 2:

#             raise ValueError("SGAPC requires exactly two classes.")

#         # ---------------------------------------------------------------------
#         # Standardization
#         #
#         # IMPORTANT:
#         # This scaler is fitted ONLY on the current training fold.
#         # ---------------------------------------------------------------------

#         self.scaler_ = StandardScaler()

#         X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

#         # ---------------------------------------------------------------------
#         # Correlation graph
#         #
#         # Already constructed from this training fold.
#         # ---------------------------------------------------------------------

#         if self.correlation_indices is not None and self.correlation_matrix is not None:

#             graph_indices = np.asarray(self.correlation_indices)

#             graph_matrix = np.asarray(self.correlation_matrix)

#         else:

#             (
#                 graph_indices,
#                 graph_matrix,
#             ) = build_gene_correlation_matrix(X_scaled)

#         self.correlation_indices_ = graph_indices

#         self.correlation_matrix_ = graph_matrix

#         # ---------------------------------------------------------------------
#         # Convert to tensors
#         # ---------------------------------------------------------------------

#         X_tensor = torch.tensor(
#             X_scaled,
#             dtype=torch.float32,
#             device=DEVICE,
#         )

#         y_tensor = torch.tensor(
#             y,
#             dtype=torch.long,
#             device=DEVICE,
#         )

#         # ---------------------------------------------------------------------
#         # Create network
#         # ---------------------------------------------------------------------

#         self.model_ = SGAPCNetwork(
#             num_features=X.shape[1],
#             num_classes=2,
#             temperature=self.temperature,
#         ).to(DEVICE)

#         # ---------------------------------------------------------------------
#         # Prototype initialization
#         #
#         # Initialize each class prototype from its training samples.
#         # ---------------------------------------------------------------------

#         with torch.no_grad():

#             for class_index, class_value in enumerate(self.classes_):

#                 class_mask = y_tensor == int(class_value)

#                 if torch.any(class_mask):

#                     class_mean = X_tensor[class_mask].mean(dim=0)

#                     self.model_.prototypes[class_index].copy_(class_mean)

#         # ---------------------------------------------------------------------
#         # Optimizer
#         # ---------------------------------------------------------------------

#         optimizer = torch.optim.Adam(
#             self.model_.parameters(),
#             lr=self.lr,
#         )

#         criterion = nn.CrossEntropyLoss()

#         # ---------------------------------------------------------------------
#         # Correlation graph tensors
#         # ---------------------------------------------------------------------

#         graph_tensor = torch.tensor(
#             self.correlation_matrix_,
#             dtype=torch.float32,
#             device=DEVICE,
#         )

#         graph_indices_tensor = torch.tensor(
#             self.correlation_indices_,
#             dtype=torch.long,
#             device=DEVICE,
#         )

#         # ---------------------------------------------------------------------
#         # Training
#         # ---------------------------------------------------------------------

#         self.model_.train()

#         n_samples = X_scaled.shape[0]

#         for epoch in range(self.epochs):

#             # -------------------------------------------------------------
#             # Shuffle training samples
#             # -------------------------------------------------------------

#             permutation = torch.randperm(
#                 n_samples,
#                 device=DEVICE,
#             )

#             epoch_loss = 0.0

#             for start in range(
#                 0,
#                 n_samples,
#                 self.batch_size,
#             ):

#                 batch_indices = permutation[start : start + self.batch_size]

#                 batch_X = X_tensor[batch_indices]

#                 batch_y = y_tensor[batch_indices]

#                 optimizer.zero_grad()

#                 # ---------------------------------------------------------
#                 # Forward
#                 # ---------------------------------------------------------

#                 logits, attention = self.model_(batch_X)

#                 # ---------------------------------------------------------
#                 # Classification loss
#                 # ---------------------------------------------------------

#                 classification_loss = criterion(
#                     logits,
#                     batch_y,
#                 )

#                 # ---------------------------------------------------------
#                 # Sparse attention loss
#                 #
#                 # Minimizing the mean attention encourages many genes to
#                 # approach zero.
#                 # ---------------------------------------------------------

#                 sparse_loss = torch.mean(attention)

#                 # ---------------------------------------------------------
#                 # Correlation regularization
#                 #
#                 # Genes with high correlation should have more coherent
#                 # attention values.
#                 #
#                 # Only genes represented in the correlation graph are used.
#                 # ---------------------------------------------------------

#                 graph_attention = attention[graph_indices_tensor]

#                 difference = graph_attention.unsqueeze(0) - graph_attention.unsqueeze(1)

#                 correlation_loss = torch.mean(graph_tensor * difference.pow(2))

#                 # ---------------------------------------------------------
#                 # Prototype separation
#                 #
#                 # Encourage AD and Control prototypes to point in different
#                 # directions.
#                 # ---------------------------------------------------------

#                 normalized_prototypes = F.normalize(
#                     self.model_.prototypes,
#                     p=2,
#                     dim=1,
#                 )

#                 prototype_similarity = torch.sum(
#                     normalized_prototypes[0] * normalized_prototypes[1]
#                 )

#                 separation_loss = prototype_similarity.pow(2)

#                 # ---------------------------------------------------------
#                 # Total loss
#                 # ---------------------------------------------------------

#                 total_loss = (
#                     classification_loss
#                     + self.sparsity_lambda * sparse_loss
#                     + self.correlation_lambda * correlation_loss
#                     + self.separation_lambda * separation_loss
#                 )

#                 # ---------------------------------------------------------
#                 # Backpropagation
#                 # ---------------------------------------------------------

#                 total_loss.backward()

#                 # ---------------------------------------------------------
#                 # Gradient clipping
#                 # ---------------------------------------------------------

#                 torch.nn.utils.clip_grad_norm_(
#                     self.model_.parameters(),
#                     max_norm=5.0,
#                 )

#                 optimizer.step()

#                 epoch_loss += float(total_loss.detach().cpu().item())

#             # -------------------------------------------------------------
#             # Progress
#             # -------------------------------------------------------------

#             if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:

#                 with torch.no_grad():

#                     self.model_.eval()

#                     train_logits, _ = self.model_(X_tensor)

#                     train_probability = (
#                         F.softmax(
#                             train_logits,
#                             dim=1,
#                         )[:, 1]
#                         .detach()
#                         .cpu()
#                         .numpy()
#                     )

#                     try:

#                         train_auc = roc_auc_score(
#                             y,
#                             train_probability,
#                         )

#                     except Exception:

#                         train_auc = np.nan

#                     current_attention = (
#                         self.model_.gene_attention().detach().cpu().numpy()
#                     )

#                     n_active = int(np.sum(current_attention >= GENE_THRESHOLD))

#                     self.model_.train()

#                 print(
#                     f"      Epoch "
#                     f"{epoch + 1:3d}/"
#                     f"{self.epochs} | "
#                     f"Loss="
#                     f"{epoch_loss:.4f} | "
#                     f"Train AUC="
#                     f"{train_auc:.4f} | "
#                     f"Active genes="
#                     f"{n_active:,}"
#                 )

#         # ---------------------------------------------------------------------
#         # Final learned gene importance
#         # ---------------------------------------------------------------------

#         self.model_.eval()

#         with torch.no_grad():

#             attention = self.model_.gene_attention().detach().cpu().numpy()

#         self.gene_importance_ = attention

#         # ---------------------------------------------------------------------
#         # sklearn-compatible feature_importances_
#         # ---------------------------------------------------------------------

#         self.feature_importances_ = attention

#         return self

#     # =========================================================================
#     # PREDICT PROBABILITY
#     # =========================================================================

#     def predict_proba(self, X):

#         check_X = np.asarray(
#             X,
#             dtype=np.float32,
#         )

#         X_scaled = self.scaler_.transform(check_X).astype(np.float32)

#         X_tensor = torch.tensor(
#             X_scaled,
#             dtype=torch.float32,
#             device=DEVICE,
#         )

#         self.model_.eval()

#         with torch.no_grad():

#             logits, _ = self.model_(X_tensor)

#             probabilities = (
#                 F.softmax(
#                     logits,
#                     dim=1,
#                 )
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#         return probabilities

#     # =========================================================================
#     # PREDICT
#     # =========================================================================

#     def predict(self, X):

#         probabilities = self.predict_proba(X)

#         return self.classes_[
#             np.argmax(
#                 probabilities,
#                 axis=1,
#             )
#         ]

#     # =========================================================================
#     # GENE IMPORTANCE
#     # =========================================================================

#     def get_gene_importance(self):

#         check = getattr(
#             self,
#             "gene_importance_",
#             None,
#         )

#         if check is None:

#             raise RuntimeError("Model must be fitted first.")

#         return check


# # =============================================================================
# # CROSS-VALIDATION
# # =============================================================================

# print()
# print("=" * 80)
# print("STARTING NESTED GENE SELECTION")
# print("=" * 80)

# print()

# cv = StratifiedKFold(
#     n_splits=N_SPLITS,
#     shuffle=True,
#     random_state=SEED,
# )


# # =============================================================================
# # STORAGE
# # =============================================================================

# fold_gene_importance = []

# fold_auc = []

# fold_selected_genes = []


# # =============================================================================
# # CV LOOP
# # =============================================================================

# for fold, (
#     train_idx,
#     valid_idx,
# ) in enumerate(
#     cv.split(
#         Xtr,
#         ytr,
#     ),
#     start=1,
# ):

#     print()
#     print()
#     print("=" * 80)
#     print(f"CV FOLD " f"{fold}/{N_SPLITS}")
#     print("=" * 80)

#     X_train_fold = Xtr[train_idx]

#     y_train_fold = ytr[train_idx]

#     X_valid_fold = Xtr[valid_idx]

#     y_valid_fold = ytr[valid_idx]

#     # =========================================================================
#     # CORRELATION GRAPH FROM TRAINING FOLD ONLY
#     # =========================================================================

#     print()
#     print("Building gene correlation graph " "from training fold...")

#     (
#         correlation_indices,
#         correlation_matrix,
#     ) = build_gene_correlation_matrix(
#         X_train_fold,
#         max_genes=MAX_CORR_GENES,
#     )

#     print(f"Correlation graph genes: " f"{len(correlation_indices):,}")

#     # =========================================================================
#     # CREATE MODEL
#     # =========================================================================

#     clf = SGAPCClassifier(
#         epochs=EPOCHS,
#         lr=LEARNING_RATE,
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         sparsity_lambda=SPARSITY_LAMBDA,
#         correlation_lambda=CORRELATION_LAMBDA,
#         separation_lambda=SEPARATION_LAMBDA,
#         random_state=SEED + fold,
#         correlation_indices=correlation_indices,
#         correlation_matrix=correlation_matrix,
#     )

#     # =========================================================================
#     # FIT
#     # =========================================================================

#     print()
#     print("Training SGAPC...")

#     clf.fit(
#         X_train_fold,
#         y_train_fold,
#     )

#     # =========================================================================
#     # VALIDATION AUC
#     # =========================================================================

#     validation_probability = clf.predict_proba(X_valid_fold)[:, 1]

#     validation_auc = roc_auc_score(
#         y_valid_fold,
#         validation_probability,
#     )

#     fold_auc.append(validation_auc)

#     print()
#     print(f"Validation ROC-AUC: " f"{validation_auc:.4f}")

#     # =========================================================================
#     # EXTRACT GENE IMPORTANCE
#     # =========================================================================

#     importance = clf.get_gene_importance()

#     fold_gene_importance.append(importance)

#     # =========================================================================
#     # RANK GENES
#     # =========================================================================

#     ranking = np.argsort(importance)[::-1]

#     top_fold_genes = [genes[i] for i in ranking[:N_TOP_GENES]]

#     fold_selected_genes.append(top_fold_genes)

#     print()
#     print(f"TOP {N_TOP_GENES} GENES - " f"FOLD {fold}")

#     print("-" * 80)

#     for rank, index in enumerate(
#         ranking[:N_TOP_GENES],
#         start=1,
#     ):

#         print(
#             f"{rank:2d}. "
#             f"{genes[index]:<25} "
#             f"attention="
#             f"{importance[index]:.6f}"
#         )


# # =============================================================================
# # CV PERFORMANCE
# # =============================================================================

# print()
# print()
# print("=" * 80)
# print("CROSS-VALIDATION PERFORMANCE")
# print("=" * 80)

# print()


# fold_auc = np.asarray(
#     fold_auc,
#     dtype=float,
# )


# for i, auc in enumerate(
#     fold_auc,
#     start=1,
# ):

#     print(f"Fold {i}: " f"ROC-AUC = " f"{auc:.4f}")


# print()

# print(f"Mean CV ROC-AUC : " f"{np.mean(fold_auc):.4f}")


# print(f"Std CV ROC-AUC  : " f"{np.std(fold_auc):.4f}")


# # =============================================================================
# # CONSENSUS GENE IMPORTANCE
# # =============================================================================

# print()
# print("=" * 80)
# print("BUILDING GENE CONSENSUS")
# print("=" * 80)


# importance_matrix = np.vstack(fold_gene_importance)


# # -----------------------------------------------------------------------------
# # Mean attention
# # -----------------------------------------------------------------------------

# mean_importance = np.mean(
#     importance_matrix,
#     axis=0,
# )


# # -----------------------------------------------------------------------------
# # Standard deviation
# # -----------------------------------------------------------------------------

# std_importance = np.std(
#     importance_matrix,
#     axis=0,
# )


# # -----------------------------------------------------------------------------
# # Selection frequency
# # -----------------------------------------------------------------------------

# selection_frequency = np.mean(
#     importance_matrix >= GENE_THRESHOLD,
#     axis=0,
# )


# # -----------------------------------------------------------------------------
# # Top-10 frequency
# # -----------------------------------------------------------------------------

# top10_frequency = np.zeros(
#     len(genes),
#     dtype=float,
# )


# for fold_genes in fold_selected_genes:

#     for gene in fold_genes:

#         gene_index = genes.index(gene)

#         top10_frequency[gene_index] += 1


# top10_frequency /= N_SPLITS


# # =============================================================================
# # CONSENSUS SCORE
# # =============================================================================

# # -----------------------------------------------------------------------------
# # Rank score
# # -----------------------------------------------------------------------------

# rank_matrix = (
#     np.argsort(
#         np.argsort(
#             -importance_matrix,
#             axis=1,
#         ),
#         axis=1,
#     )
#     + 1
# )


# mean_rank = np.mean(
#     rank_matrix,
#     axis=0,
# )


# rank_score = 1.0 - (mean_rank - 1) / max(
#     len(genes) - 1,
#     1,
# )


# # -----------------------------------------------------------------------------
# # Normalize importance
# # -----------------------------------------------------------------------------

# positive_importance = np.maximum(
#     mean_importance,
#     0,
# )


# max_imp = np.max(positive_importance)


# if max_imp > 0:

#     importance_score = positive_importance / max_imp

# else:

#     importance_score = np.zeros_like(positive_importance)


# # -----------------------------------------------------------------------------
# # Final consensus
# #
# # 40% learned attention
# # 30% rank stability
# # 20% selection frequency
# # 10% top-10 frequency
# # -----------------------------------------------------------------------------

# consensus_score = (
#     0.40 * importance_score
#     + 0.30 * rank_score
#     + 0.20 * selection_frequency
#     + 0.10 * top10_frequency
# )


# # =============================================================================
# # BUILD RESULT TABLE IN MEMORY ONLY
# # =============================================================================

# consensus_df = pd.DataFrame(
#     {
#         "gene": genes,
#         "mean_attention": mean_importance,
#         "std_attention": std_importance,
#         "selection_frequency": selection_frequency,
#         "top10_frequency": top10_frequency,
#         "mean_rank": mean_rank,
#         "rank_score": rank_score,
#         "importance_score": importance_score,
#         "consensus_score": consensus_score,
#     }
# )


# # =============================================================================
# # SORT
# # =============================================================================

# consensus_df = consensus_df.sort_values(
#     [
#         "consensus_score",
#         "selection_frequency",
#         "mean_attention",
#     ],
#     ascending=False,
# ).reset_index(drop=True)


# consensus_df["consensus_rank"] = np.arange(len(consensus_df)) + 1


# # =============================================================================
# # FINAL TOP GENES
# # =============================================================================

# top10 = consensus_df.head(N_TOP_GENES).copy()


# # =============================================================================
# # PRINT FINAL RESULT
# # =============================================================================

# print()
# print()
# print("=" * 80)
# print("FINAL CONSENSUS TOP-10 GENES")
# print("=" * 80)

# print()


# for _, row in top10.iterrows():

#     print(
#         f"{int(row['consensus_rank']):2d}. "
#         f"{row['gene']:<25} "
#         f"score="
#         f"{row['consensus_score']:.6f} | "
#         f"attention="
#         f"{row['mean_attention']:.6f} | "
#         f"CV support="
#         f"{row['selection_frequency']:.2%} | "
#         f"Top-10="
#         f"{row['top10_frequency']:.2%}"
#     )


# # =============================================================================
# # FINAL TRAINING MODEL
# # =============================================================================
# #
# # After gene discovery, train ONE final SGAPC model using the complete
# # training set.
# #
# # IMPORTANT:
# #
# # The independent test set is still untouched.
# # =============================================================================

# print()
# print()
# print("=" * 80)
# print("TRAINING FINAL MODEL ON TRAINING DATA")
# print("=" * 80)


# (
#     final_corr_indices,
#     final_corr_matrix,
# ) = build_gene_correlation_matrix(
#     Xtr,
#     max_genes=MAX_CORR_GENES,
# )


# final_model = SGAPCClassifier(
#     epochs=EPOCHS,
#     lr=LEARNING_RATE,
#     batch_size=BATCH_SIZE,
#     temperature=TEMPERATURE,
#     sparsity_lambda=SPARSITY_LAMBDA,
#     correlation_lambda=CORRELATION_LAMBDA,
#     separation_lambda=SEPARATION_LAMBDA,
#     random_state=SEED,
#     correlation_indices=final_corr_indices,
#     correlation_matrix=final_corr_matrix,
# )


# final_model.fit(
#     Xtr,
#     ytr,
# )


# # =============================================================================
# # INDEPENDENT TEST EVALUATION
# # =============================================================================

# print()
# print("=" * 80)
# print("INDEPENDENT TEST EVALUATION")
# print("=" * 80)

# print()

# test_probability = final_model.predict_proba(Xte)[:, 1]


# test_auc = roc_auc_score(
#     yte,
#     test_probability,
# )


# print(f"Independent Test ROC-AUC: " f"{test_auc:.4f}")


# # =============================================================================
# # FINAL LEARNED GENE IMPORTANCE
# # =============================================================================

# final_importance = final_model.get_gene_importance()


# final_ranking = np.argsort(final_importance)[::-1]


# print()
# print("=" * 80)
# print("FINAL MODEL LEARNED TOP-10 GENES")
# print("=" * 80)

# print()


# for rank, index in enumerate(
#     final_ranking[:N_TOP_GENES],
#     start=1,
# ):

#     print(
#         f"{rank:2d}. "
#         f"{genes[index]:<25} "
#         f"attention="
#         f"{final_importance[index]:.6f}"
#     )


# # =============================================================================
# # FINAL SUMMARY
# # =============================================================================

# print()
# print()
# print("=" * 80)
# print("STEP 18 COMPLETE")
# print("=" * 80)

# print()

# print(f"Samples                 : " f"{len(df):,}")

# print(f"Genes                   : " f"{len(genes):,}")

# print(f"Training samples        : " f"{len(tr):,}")

# print(f"Independent test        : " f"{len(te):,}")

# print(f"CV folds                : " f"{N_SPLITS}")

# print(f"Mean CV ROC-AUC         : " f"{np.mean(fold_auc):.4f}")

# print(f"Independent Test AUC    : " f"{test_auc:.4f}")

# print(f"Final genes             : " f"{N_TOP_GENES}")

# print()

# print("CONSENSUS TOP-10")
# print("-" * 80)

# for _, row in top10.iterrows():

#     print(f"{int(row['consensus_rank']):2d}. " f"{row['gene']}")

# print()

# print("No result files were written to disk.")

# print()

# print("=" * 80)


"""
===============================================================================
STEP 18 - CUSTOM DEEP-LEARNING GENE SELECTION
===============================================================================

Algorithm
---------
SGAPC
Sparse Gene-Attention Prototype Classifier

Purpose
-------
Learn biologically meaningful and predictive genes directly from high-
dimensional gene-expression data.

Key properties
--------------
1. No SelectKBest pre-filtering.
2. All genes enter the model.
3. Gene importance is learned end-to-end.
4. Sparse gene attention encourages a compact gene signature.
5. Correlation regularization encourages co-expressed genes to behave
   coherently.
6. Prototype-based classification separates AD and Control.
7. Gene selection occurs independently inside every CV training fold.
8. The independent test set is NEVER used for gene selection.
9. Nothing is saved to disk.

Input
-----
D:/Dementia2/codes/ml_data/integrated_gene_expression.csv

Output
------
Printed to console only.

===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)

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


SEED = 42

TEST_SIZE = 0.20

N_SPLITS = 5

EPOCHS = 100

LEARNING_RATE = 0.005

BATCH_SIZE = 64

TEMPERATURE = 0.5


# -----------------------------------------------------------------------------
# Sparse gene selection
# -----------------------------------------------------------------------------

SPARSITY_LAMBDA = 1e-3


# -----------------------------------------------------------------------------
# Correlation regularization
# -----------------------------------------------------------------------------

CORRELATION_LAMBDA = 1e-3


# -----------------------------------------------------------------------------
# Prototype separation
# -----------------------------------------------------------------------------

SEPARATION_LAMBDA = 1e-2


# -----------------------------------------------------------------------------
# Number of final genes
# -----------------------------------------------------------------------------

N_TOP_GENES = 10


# -----------------------------------------------------------------------------
# Attention threshold
# -----------------------------------------------------------------------------

GENE_THRESHOLD = 0.50


# -----------------------------------------------------------------------------
# Maximum number of genes used to construct correlation matrix
# -----------------------------------------------------------------------------

MAX_CORR_GENES = 3000


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 18 - SGAPC")
print("SPARSE GENE-ATTENTION PROTOTYPE CLASSIFIER")
print("=" * 80)

print()

print(f"Device                  : {DEVICE}")
print(f"Input                   : {INFILE}")
print(f"Seed                    : {SEED}")
print(f"Test size               : {TEST_SIZE}")
print(f"CV folds                : {N_SPLITS}")
print(f"Epochs                  : {EPOCHS}")
print(f"Learning rate           : {LEARNING_RATE}")
print(f"Batch size              : {BATCH_SIZE}")
print(f"Sparsity lambda         : {SPARSITY_LAMBDA}")
print(f"Correlation lambda      : {CORRELATION_LAMBDA}")
print(f"Separation lambda       : {SEPARATION_LAMBDA}")
print(f"Final genes             : {N_TOP_GENES}")


# =============================================================================
# LOAD DATA
# =============================================================================

print()
print("=" * 80)
print("LOADING DATA")
print("=" * 80)


if not os.path.exists(INFILE):

    raise FileNotFoundError(f"\nInput file not found:\n{INFILE}")


try:

    df = pd.read_csv(
        INFILE,
        engine="c",
        low_memory=False,
    )

except (pd.errors.ParserError, OSError) as first_error:

    print()
    print("WARNING: Standard CSV parser failed.")
    print("Trying robust Python CSV parser...")

    try:

        df = pd.read_csv(
            INFILE,
            engine="python",
            on_bad_lines="warn",
        )

    except Exception as second_error:

        raise RuntimeError(
            "\nUnable to read the integrated expression file.\n\n"
            f"File:\n{INFILE}\n\n"
            f"First parser error:\n{first_error}\n\n"
            f"Python parser error:\n{second_error}\n"
        )


print()

print(f"Loaded dataframe: " f"{df.shape[0]:,} samples × {df.shape[1]:,} columns")


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

    raise ValueError("No gene-expression columns were detected.")


print()
print(f"Genes detected: {len(genes):,}")


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

print()
print("Converting gene-expression values to numeric...")


df[genes] = df[genes].apply(pd.to_numeric, errors="coerce")


valid_gene_mask = df[genes].notna().any(axis=0).values


genes = [gene for gene, valid in zip(genes, valid_gene_mask) if valid]


df[genes] = df[genes].fillna(0.0)


# =============================================================================
# LABEL
# =============================================================================

df["label"] = pd.to_numeric(
    df["label"],
    errors="raise",
).astype(int)


y = df["label"].values


unique_labels = sorted(np.unique(y))


if not set(unique_labels).issubset({0, 1}):

    raise ValueError("Label column must contain only 0 and 1.")


if len(unique_labels) != 2:

    raise ValueError("Both Alzheimer (1) and Control (0) classes are required.")


# =============================================================================
# CREATE EXPRESSION MATRIX
# =============================================================================

X = df[genes].values.astype(np.float32)


print()
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print()

print(f"Samples                 : {len(df):,}")
print(f"Genes                   : {len(genes):,}")
print(f"Alzheimer               : {(y == 1).sum():,}")
print(f"Control                 : {(y == 0).sum():,}")
print(f"Datasets                : " f"{df['dataset'].nunique():,}")


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

print()
print("=" * 80)
print("CREATING INDEPENDENT TEST SET")
print("=" * 80)


indices = np.arange(len(df))


strata = df["dataset"].astype(str) + "_" + df["label"].astype(str)


try:

    tr, te = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=strata,
    )

    split_method = "dataset + diagnosis stratification"


except ValueError:

    print("WARNING: dataset + diagnosis " "stratification failed.")

    print("Using diagnosis-only stratification.")

    tr, te = train_test_split(
        indices,
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

print(f"Split method            : {split_method}")
print(f"Training samples        : {len(tr):,}")
print(f"Independent test        : {len(te):,}")


# =============================================================================
# CORRELATION GRAPH
# =============================================================================


def build_gene_correlation_matrix(
    X_train,
    max_genes=MAX_CORR_GENES,
):
    """
    Construct a gene-gene correlation graph using ONLY the training fold.
    """

    n_samples, n_genes = X_train.shape

    if n_genes <= max_genes:

        selected_indices = np.arange(n_genes)

    else:

        variances = np.var(
            X_train,
            axis=0,
        )

        selected_indices = np.argsort(variances)[-max_genes:]

    X_graph = X_train[:, selected_indices]

    corr = np.corrcoef(
        X_graph,
        rowvar=False,
    )

    corr = np.nan_to_num(
        corr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    corr = np.abs(corr)

    np.fill_diagonal(
        corr,
        0.0,
    )

    degree = np.sum(
        corr,
        axis=1,
    )

    degree = np.maximum(
        degree,
        1e-8,
    )

    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))

    adjacency = D_inv_sqrt @ corr @ D_inv_sqrt

    return (selected_indices, adjacency.astype(np.float32))


# =============================================================================
# CUSTOM NEURAL NETWORK
# =============================================================================


class SGAPCNetwork(nn.Module):
    """
    Sparse Gene-Attention Prototype Classifier.
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

        self.gene_logits = nn.Parameter(torch.zeros(num_features))

        self.prototypes = nn.Parameter(torch.randn(num_classes, num_features) * 0.02)

    def gene_attention(self):

        return torch.sigmoid(self.gene_logits)

    def forward(self, X):

        attention = self.gene_attention()

        X_weighted = X * attention

        prototypes_weighted = self.prototypes * attention.unsqueeze(0)

        X_norm = F.normalize(
            X_weighted,
            p=2,
            dim=1,
        )

        prototype_norm = F.normalize(
            prototypes_weighted,
            p=2,
            dim=1,
        )

        similarity = torch.mm(
            X_norm,
            prototype_norm.T,
        )

        logits = similarity / self.temperature

        return (
            logits,
            attention,
        )


# =============================================================================
# SKLEARN WRAPPER
# =============================================================================


class SGAPCClassifier(
    BaseEstimator,
    ClassifierMixin,
):
    """
    Scikit-learn compatible implementation of SGAPC.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        epochs=100,
        lr=0.005,
        batch_size=64,
        temperature=0.5,
        sparsity_lambda=1e-3,
        correlation_lambda=1e-3,
        separation_lambda=1e-2,
        random_state=42,
        correlation_indices=None,
        correlation_matrix=None,
    ):

        self.epochs = epochs

        self.lr = lr

        self.batch_size = batch_size

        self.temperature = temperature

        self.sparsity_lambda = sparsity_lambda

        self.correlation_lambda = correlation_lambda

        self.separation_lambda = separation_lambda

        self.random_state = random_state

        self.correlation_indices = correlation_indices

        self.correlation_matrix = correlation_matrix

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.int64,
        )

        np.random.seed(self.random_state)

        torch.manual_seed(self.random_state)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(self.random_state)

        self.classes_ = np.array(sorted(np.unique(y)))

        if len(self.classes_) != 2:

            raise ValueError("SGAPC requires exactly two classes.")

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)

        if self.correlation_indices is not None and self.correlation_matrix is not None:

            graph_indices = np.asarray(self.correlation_indices)

            graph_matrix = np.asarray(self.correlation_matrix)

        else:

            (
                graph_indices,
                graph_matrix,
            ) = build_gene_correlation_matrix(X_scaled)

        self.correlation_indices_ = graph_indices

        self.correlation_matrix_ = graph_matrix

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32,
            device=DEVICE,
        )

        y_tensor = torch.tensor(
            y,
            dtype=torch.long,
            device=DEVICE,
        )

        self.model_ = SGAPCNetwork(
            num_features=X.shape[1],
            num_classes=2,
            temperature=self.temperature,
        ).to(DEVICE)

        with torch.no_grad():

            for class_index, class_value in enumerate(self.classes_):

                class_mask = y_tensor == int(class_value)

                if torch.any(class_mask):

                    class_mean = X_tensor[class_mask].mean(dim=0)

                    self.model_.prototypes[class_index].copy_(class_mean)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
        )

        criterion = nn.CrossEntropyLoss()

        graph_tensor = torch.tensor(
            self.correlation_matrix_,
            dtype=torch.float32,
            device=DEVICE,
        )

        graph_indices_tensor = torch.tensor(
            self.correlation_indices_,
            dtype=torch.long,
            device=DEVICE,
        )

        self.model_.train()

        n_samples = X_scaled.shape[0]

        for epoch in range(self.epochs):

            permutation = torch.randperm(
                n_samples,
                device=DEVICE,
            )

            epoch_loss = 0.0

            for start in range(
                0,
                n_samples,
                self.batch_size,
            ):

                batch_indices = permutation[start : start + self.batch_size]

                batch_X = X_tensor[batch_indices]

                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()

                logits, attention = self.model_(batch_X)

                classification_loss = criterion(
                    logits,
                    batch_y,
                )

                sparse_loss = torch.mean(attention)

                graph_attention = attention[graph_indices_tensor]

                difference = graph_attention.unsqueeze(0) - graph_attention.unsqueeze(1)

                correlation_loss = torch.mean(graph_tensor * difference.pow(2))

                normalized_prototypes = F.normalize(
                    self.model_.prototypes,
                    p=2,
                    dim=1,
                )

                prototype_similarity = torch.sum(
                    normalized_prototypes[0] * normalized_prototypes[1]
                )

                separation_loss = prototype_similarity.pow(2)

                total_loss = (
                    classification_loss
                    + self.sparsity_lambda * sparse_loss
                    + self.correlation_lambda * correlation_loss
                    + self.separation_lambda * separation_loss
                )

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

                epoch_loss += float(total_loss.detach().cpu().item())

            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:

                with torch.no_grad():

                    self.model_.eval()

                    train_logits, _ = self.model_(X_tensor)

                    train_probability = (
                        F.softmax(
                            train_logits,
                            dim=1,
                        )[:, 1]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    try:

                        train_auc = roc_auc_score(
                            y,
                            train_probability,
                        )

                    except Exception:

                        train_auc = np.nan

                    current_attention = (
                        self.model_.gene_attention().detach().cpu().numpy()
                    )

                    n_active = int(np.sum(current_attention >= GENE_THRESHOLD))

                    self.model_.train()

                print(
                    f"      Epoch "
                    f"{epoch + 1:3d}/"
                    f"{self.epochs} | "
                    f"Loss="
                    f"{epoch_loss:.4f} | "
                    f"Train AUC="
                    f"{train_auc:.4f} | "
                    f"Active genes="
                    f"{n_active:,}"
                )

        self.model_.eval()

        with torch.no_grad():

            attention = self.model_.gene_attention().detach().cpu().numpy()

        self.gene_importance_ = attention

        self.feature_importances_ = attention

        return self

    def predict_proba(self, X):

        check_X = np.asarray(
            X,
            dtype=np.float32,
        )

        X_scaled = self.scaler_.transform(check_X).astype(np.float32)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32,
            device=DEVICE,
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

    def predict(self, X):

        probabilities = self.predict_proba(X)

        return self.classes_[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]

    def get_gene_importance(self):

        check = getattr(
            self,
            "gene_importance_",
            None,
        )

        if check is None:

            raise RuntimeError("Model must be fitted first.")

        return check


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

print()
print("=" * 80)
print("STARTING NESTED GENE SELECTION")
print("=" * 80)

print()

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED,
)


# =============================================================================
# STORAGE
# =============================================================================

fold_gene_importance = []

fold_auc = []

fold_accuracy = []

fold_f1 = []

fold_selected_genes = []


# =============================================================================
# CV LOOP
# =============================================================================

for fold, (
    train_idx,
    valid_idx,
) in enumerate(
    cv.split(
        Xtr,
        ytr,
    ),
    start=1,
):

    print()
    print()
    print("=" * 80)
    print(f"CV FOLD " f"{fold}/{N_SPLITS}")
    print("=" * 80)

    X_train_fold = Xtr[train_idx]

    y_train_fold = ytr[train_idx]

    X_valid_fold = Xtr[valid_idx]

    y_valid_fold = ytr[valid_idx]

    print()
    print("Building gene correlation graph " "from training fold...")

    (
        correlation_indices,
        correlation_matrix,
    ) = build_gene_correlation_matrix(
        X_train_fold,
        max_genes=MAX_CORR_GENES,
    )

    print(f"Correlation graph genes: " f"{len(correlation_indices):,}")

    clf = SGAPCClassifier(
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        sparsity_lambda=SPARSITY_LAMBDA,
        correlation_lambda=CORRELATION_LAMBDA,
        separation_lambda=SEPARATION_LAMBDA,
        random_state=SEED + fold,
        correlation_indices=correlation_indices,
        correlation_matrix=correlation_matrix,
    )

    print()
    print("Training SGAPC...")

    clf.fit(
        X_train_fold,
        y_train_fold,
    )

    # -------------------------------------------------------------------------
    # Validation predictions and metrics
    # -------------------------------------------------------------------------

    validation_probability = clf.predict_proba(X_valid_fold)[:, 1]

    validation_prediction = clf.predict(X_valid_fold)

    validation_auc = roc_auc_score(
        y_valid_fold,
        validation_probability,
    )

    validation_accuracy = accuracy_score(
        y_valid_fold,
        validation_prediction,
    )

    validation_f1 = f1_score(
        y_valid_fold,
        validation_prediction,
    )

    fold_auc.append(validation_auc)

    fold_accuracy.append(validation_accuracy)

    fold_f1.append(validation_f1)

    print()
    print(f"Validation ROC-AUC  : " f"{validation_auc:.4f}")
    print(f"Validation Accuracy : " f"{validation_accuracy * 100:.2f}%")
    print(f"Validation F1-score : " f"{validation_f1:.4f}")

    importance = clf.get_gene_importance()

    fold_gene_importance.append(importance)

    ranking = np.argsort(importance)[::-1]

    top_fold_genes = [genes[i] for i in ranking[:N_TOP_GENES]]

    fold_selected_genes.append(top_fold_genes)

    print()
    print(f"TOP {N_TOP_GENES} GENES - " f"FOLD {fold}")

    print("-" * 80)

    for rank, index in enumerate(
        ranking[:N_TOP_GENES],
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{genes[index]:<25} "
            f"attention="
            f"{importance[index]:.6f}"
        )


# =============================================================================
# CV PERFORMANCE
# =============================================================================

print()
print()
print("=" * 80)
print("CROSS-VALIDATION PERFORMANCE")
print("=" * 80)

print()


fold_auc = np.asarray(
    fold_auc,
    dtype=float,
)

fold_accuracy = np.asarray(
    fold_accuracy,
    dtype=float,
)

fold_f1 = np.asarray(
    fold_f1,
    dtype=float,
)


for i, (auc, acc, f1) in enumerate(
    zip(fold_auc, fold_accuracy, fold_f1),
    start=1,
):

    print(
        f"Fold {i}: "
        f"ROC-AUC = {auc:.4f} | "
        f"Accuracy = {acc * 100:.2f}% | "
        f"F1 = {f1:.4f}"
    )


print()

print(f"Mean CV ROC-AUC   : " f"{np.mean(fold_auc):.4f}")
print(f"Std  CV ROC-AUC   : " f"{np.std(fold_auc):.4f}")
print(f"Mean CV Accuracy  : " f"{np.mean(fold_accuracy) * 100:.2f}%")
print(f"Std  CV Accuracy  : " f"{np.std(fold_accuracy) * 100:.2f}%")
print(f"Mean CV F1-score  : " f"{np.mean(fold_f1):.4f}")
print(f"Std  CV F1-score  : " f"{np.std(fold_f1):.4f}")


# =============================================================================
# CONSENSUS GENE IMPORTANCE
# =============================================================================

print()
print("=" * 80)
print("BUILDING GENE CONSENSUS")
print("=" * 80)


importance_matrix = np.vstack(fold_gene_importance)


mean_importance = np.mean(
    importance_matrix,
    axis=0,
)


std_importance = np.std(
    importance_matrix,
    axis=0,
)


selection_frequency = np.mean(
    importance_matrix >= GENE_THRESHOLD,
    axis=0,
)


top10_frequency = np.zeros(
    len(genes),
    dtype=float,
)


for fold_genes in fold_selected_genes:

    for gene in fold_genes:

        gene_index = genes.index(gene)

        top10_frequency[gene_index] += 1


top10_frequency /= N_SPLITS


rank_matrix = (
    np.argsort(
        np.argsort(
            -importance_matrix,
            axis=1,
        ),
        axis=1,
    )
    + 1
)


mean_rank = np.mean(
    rank_matrix,
    axis=0,
)


rank_score = 1.0 - (mean_rank - 1) / max(
    len(genes) - 1,
    1,
)


positive_importance = np.maximum(
    mean_importance,
    0,
)


max_imp = np.max(positive_importance)


if max_imp > 0:

    importance_score = positive_importance / max_imp

else:

    importance_score = np.zeros_like(positive_importance)


consensus_score = (
    0.40 * importance_score
    + 0.30 * rank_score
    + 0.20 * selection_frequency
    + 0.10 * top10_frequency
)


consensus_df = pd.DataFrame(
    {
        "gene": genes,
        "mean_attention": mean_importance,
        "std_attention": std_importance,
        "selection_frequency": selection_frequency,
        "top10_frequency": top10_frequency,
        "mean_rank": mean_rank,
        "rank_score": rank_score,
        "importance_score": importance_score,
        "consensus_score": consensus_score,
    }
)


consensus_df = consensus_df.sort_values(
    [
        "consensus_score",
        "selection_frequency",
        "mean_attention",
    ],
    ascending=False,
).reset_index(drop=True)


consensus_df["consensus_rank"] = np.arange(len(consensus_df)) + 1


top10 = consensus_df.head(N_TOP_GENES).copy()


print()
print()
print("=" * 80)
print("FINAL CONSENSUS TOP-10 GENES")
print("=" * 80)

print()


for _, row in top10.iterrows():

    print(
        f"{int(row['consensus_rank']):2d}. "
        f"{row['gene']:<25} "
        f"score="
        f"{row['consensus_score']:.6f} | "
        f"attention="
        f"{row['mean_attention']:.6f} | "
        f"CV support="
        f"{row['selection_frequency']:.2%} | "
        f"Top-10="
        f"{row['top10_frequency']:.2%}"
    )


# =============================================================================
# FINAL TRAINING MODEL
# =============================================================================

print()
print()
print("=" * 80)
print("TRAINING FINAL MODEL ON TRAINING DATA")
print("=" * 80)


(
    final_corr_indices,
    final_corr_matrix,
) = build_gene_correlation_matrix(
    Xtr,
    max_genes=MAX_CORR_GENES,
)


final_model = SGAPCClassifier(
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    temperature=TEMPERATURE,
    sparsity_lambda=SPARSITY_LAMBDA,
    correlation_lambda=CORRELATION_LAMBDA,
    separation_lambda=SEPARATION_LAMBDA,
    random_state=SEED,
    correlation_indices=final_corr_indices,
    correlation_matrix=final_corr_matrix,
)


final_model.fit(
    Xtr,
    ytr,
)


# =============================================================================
# INDEPENDENT TEST EVALUATION
# =============================================================================

print()
print("=" * 80)
print("INDEPENDENT TEST EVALUATION")
print("=" * 80)

print()

test_probability = final_model.predict_proba(Xte)[:, 1]

test_prediction = final_model.predict(Xte)


test_auc = roc_auc_score(
    yte,
    test_probability,
)

test_accuracy = accuracy_score(
    yte,
    test_prediction,
)

test_f1 = f1_score(
    yte,
    test_prediction,
)


print(f"Independent Test ROC-AUC  : " f"{test_auc:.4f}")
print(f"Independent Test Accuracy : " f"{test_accuracy * 100:.2f}%")
print(f"Independent Test F1-score : " f"{test_f1:.4f}")


# =============================================================================
# FINAL LEARNED GENE IMPORTANCE
# =============================================================================

final_importance = final_model.get_gene_importance()


final_ranking = np.argsort(final_importance)[::-1]


print()
print("=" * 80)
print("FINAL MODEL LEARNED TOP-10 GENES")
print("=" * 80)

print()


for rank, index in enumerate(
    final_ranking[:N_TOP_GENES],
    start=1,
):

    print(
        f"{rank:2d}. "
        f"{genes[index]:<25} "
        f"attention="
        f"{final_importance[index]:.6f}"
    )


# =============================================================================
# FINAL SUMMARY — CV ACCURACY % / TEST ACCURACY % / F1 / AUC
# =============================================================================

print()
print()
print("=" * 80)
print("STEP 18 COMPLETE — FINAL PERFORMANCE SUMMARY")
print("=" * 80)

print()

print(f"Samples                 : {len(df):,}")
print(f"Genes                   : {len(genes):,}")
print(f"Training samples        : {len(tr):,}")
print(f"Independent test        : {len(te):,}")
print(f"CV folds                : {N_SPLITS}")

print()

print("-" * 80)
print(f"{'Metric':<24}{'CV (mean ± std)':<30}{'Independent Test':<20}")
print("-" * 80)

print(
    f"{'Accuracy (%)':<24}"
    f"{np.mean(fold_accuracy) * 100:6.2f} ± {np.std(fold_accuracy) * 100:4.2f}"
    f"{'':<10}"
    f"{test_accuracy * 100:6.2f} %"
)

print(
    f"{'F1-Score':<24}"
    f"{np.mean(fold_f1):6.4f} ± {np.std(fold_f1):4.4f}"
    f"{'':<10}"
    f"{test_f1:6.4f}"
)

print(
    f"{'ROC-AUC':<24}"
    f"{np.mean(fold_auc):6.4f} ± {np.std(fold_auc):4.4f}"
    f"{'':<10}"
    f"{test_auc:6.4f}"
)

print("-" * 80)
print()

print(f"Final genes             : " f"{N_TOP_GENES}")

print()

print("CONSENSUS TOP-10")
print("-" * 80)

for _, row in top10.iterrows():

    print(f"{int(row['consensus_rank']):2d}. " f"{row['gene']}")

print()

print("No result files were written to disk.")

print()

print("=" * 80)
