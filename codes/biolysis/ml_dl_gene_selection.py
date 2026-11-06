"""
UNSUPERVISED ML APPROACH FOR DEMENTIA GENE IDENTIFICATION
Uses clustering and anomaly detection to identify important genes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ML Libraries
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Define paths
BASE = Path(r"D:\Dementia2\codes\biolysis")
DE = BASE / "step08_DE"
OUT = BASE / "unsupervised_ml_results"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("UNSUPERVISED ML APPROACH FOR DEMENTIA GENE IDENTIFICATION")
print("=" * 80)
print(f"Output directory: {OUT}")

# List of datasets
GSES = [
    "GSE1297",
    "GSE4757",
    "GSE5281",
    "GSE6834",
    "GSE15222",
    "GSE16759",
    "GSE28146",
    "GSE29378",
    "GSE33000",
    "GSE36980",
    "GSE37263",
    "GSE44770",
]

# Step 1: Load data
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

all_data = {}
for gse in GSES:
    file_path = DE / f"{gse}_DE.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        all_data[gse] = df
        print(f"✅ Loaded {gse}: {len(df)} genes")

# Identify columns
sample_df = list(all_data.values())[0]
gene_col = "GeneSymbol"
logfc_col = "logFC"
pval_col = "p_value"

print(f"\nUsing columns: Gene={gene_col}, logFC={logfc_col}, P-value={pval_col}")

# Step 2: Create gene feature matrix
print("\n" + "=" * 80)
print("STEP 2: CREATING GENE FEATURE MATRIX")
print("=" * 80)

# Track gene presence across datasets
gene_presence = {}
for gse, df in all_data.items():
    genes = set(df[gene_col].astype(str).str.upper().str.strip())
    for gene in genes:
        gene_presence[gene] = gene_presence.get(gene, 0) + 1

# Keep genes in at least 3 datasets
selected_genes = [gene for gene, count in gene_presence.items() if count >= 3]
print(f"Genes present in >=3 datasets: {len(selected_genes)}")

# Create feature matrix
feature_matrix = pd.DataFrame(index=selected_genes)

for gse, df in all_data.items():
    df_clean = df.copy()
    df_clean[gene_col] = df_clean[gene_col].astype(str).str.upper().str.strip()
    df_clean = df_clean.drop_duplicates(gene_col)
    df_clean = df_clean.set_index(gene_col)

    # Store logFC
    feature_matrix[f"{gse}_logFC"] = df_clean[logfc_col].reindex(selected_genes)

    # Store -log10(p-value)
    pvals = df_clean[pval_col].reindex(selected_genes)
    feature_matrix[f"{gse}_neglog10p"] = -np.log10(pvals.clip(lower=1e-300))

# Fill NaN with 0 (gene not measured in that dataset)
feature_matrix = feature_matrix.fillna(0)

print(f"Feature matrix shape: {feature_matrix.shape}")

# Step 3: Calculate gene-level statistics
print("\n" + "=" * 80)
print("STEP 3: CALCULATING GENE STATISTICS")
print("=" * 80)

gene_features = []
for gene in feature_matrix.index:
    logfc_cols = [col for col in feature_matrix.columns if "logFC" in col]
    pval_cols = [col for col in feature_matrix.columns if "neglog10p" in col]

    logfc_values = feature_matrix.loc[gene, logfc_cols].values
    p_values = feature_matrix.loc[gene, pval_cols].values

    # Only consider genes that were actually measured
    n_measured = np.sum(feature_matrix.loc[gene, logfc_cols] != 0)

    if n_measured >= 3:
        gene_features.append(
            {
                "Gene": gene,
                "mean_logFC": np.mean(logfc_values[logfc_values != 0]),
                "std_logFC": np.std(logfc_values[logfc_values != 0]),
                "max_abs_logFC": np.max(np.abs(logfc_values)),
                "mean_p": np.mean(p_values[p_values != 0]),
                "max_p": np.max(p_values[p_values != 0]),
                "n_measured": n_measured,
                "consistency": np.mean(np.sign(logfc_values[logfc_values != 0])),
            }
        )

gene_stats = pd.DataFrame(gene_features)

# Step 4: Create composite features
print("\n" + "=" * 80)
print("STEP 4: CREATING COMPOSITE SCORES")
print("=" * 80)

# Calculate various scores
gene_stats["effect_score"] = (
    np.abs(gene_stats["mean_logFC"])
    * gene_stats["max_p"]
    * gene_stats["n_measured"]
    / 12
)

gene_stats["consistency_score"] = (
    np.abs(gene_stats["consistency"]) * gene_stats["max_p"]
)

gene_stats["combined_score"] = (
    0.4 * gene_stats["effect_score"]
    + 0.3 * gene_stats["consistency_score"]
    + 0.3 * gene_stats["max_abs_logFC"]
)

print(f"Analyzing {len(gene_stats)} genes")

# Step 5: Unsupervised learning - Anomaly Detection
print("\n" + "=" * 80)
print("STEP 5: ANOMALY DETECTION (FINDING IMPORTANT GENES)")
print("=" * 80)

# Prepare features for unsupervised learning
X = (
    gene_stats[
        ["mean_logFC", "max_abs_logFC", "mean_p", "max_p", "n_measured", "consistency"]
    ]
    .fillna(0)
    .values
)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Multiple anomaly detection methods
anomaly_models = {
    "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
    "Elliptic Envelope": EllipticEnvelope(contamination=0.1, random_state=42),
    "OneClassSVM": OneClassSVM(nu=0.1, kernel="rbf", gamma="auto"),
    "LOF": LocalOutlierFactor(contamination=0.1, novelty=True),
}

anomaly_scores = pd.DataFrame(index=gene_stats["Gene"])

for name, model in anomaly_models.items():
    print(f"\nTraining {name}...")
    try:
        if name == "LOF":
            model.fit(X_scaled)
            scores = model.score_samples(X_scaled)
            predictions = model.predict(X_scaled)
        else:
            model.fit(X_scaled)
            predictions = model.predict(X_scaled)
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_scaled)
            elif hasattr(model, "score_samples"):
                scores = model.score_samples(X_scaled)
            else:
                scores = -model.offset_ * np.ones(len(X_scaled))

        # Convert to scores (higher = more anomalous/important)
        anomaly_scores[f"{name}_score"] = -scores  # Invert so higher = more important
        anomaly_scores[f"{name}_prediction"] = predictions

        n_anomalies = np.sum(predictions == -1)
        print(f"  Found {n_anomalies} anomalies (important genes)")

    except Exception as e:
        print(f"  Error: {e}")

# Step 6: Clustering
print("\n" + "=" * 80)
print("STEP 6: CLUSTERING ANALYSIS")
print("=" * 80)

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
gene_stats["cluster"] = clusters

# Identify cluster with highest effect scores
cluster_effects = gene_stats.groupby("cluster")["effect_score"].mean()
top_cluster = cluster_effects.idxmax()
print(f"Top cluster (most important): {top_cluster}")
print(f"Genes in top cluster: {sum(clusters == top_cluster)}")

# Step 7: PCA for dimensionality reduction
print("\n" + "=" * 80)
print("STEP 7: PCA VISUALIZATION")
print("=" * 80)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 8: Rank genes by combined importance
print("\n" + "=" * 80)
print("STEP 8: RANKING GENES BY IMPORTANCE")
print("=" * 80)

# Combine anomaly scores
if len(anomaly_scores.columns) > 0:
    score_cols = [col for col in anomaly_scores.columns if "score" in col]
    anomaly_scores["combined_anomaly"] = anomaly_scores[score_cols].mean(axis=1)
else:
    anomaly_scores["combined_anomaly"] = 0

# Combine all scores
gene_stats["anomaly_score"] = anomaly_scores["combined_anomaly"].values
gene_stats["final_score"] = (
    0.4 * gene_stats["effect_score"]
    + 0.3 * gene_stats["consistency_score"]
    + 0.3 * gene_stats["anomaly_score"]
)

# Sort by final score
gene_stats = gene_stats.sort_values("final_score", ascending=False)

# Step 9: Identify top 10 genes
print("\n" + "=" * 80)
print("TOP 10 DEMENTIA-ASSOCIATED GENES")
print("=" * 80)

top_genes = gene_stats.head(10)

# Known dementia gene annotations
known_dementia_genes = {
    "GFAP": "Astrocyte activation (neuroinflammation)",
    "GAP43": "Neuronal growth/plasticity",
    "SYT1": "Synaptic function",
    "CALM1": "Calcium signaling",
    "CXCR4": "Neuroinflammation",
    "TGFB1": "Neuroinflammation",
    "ITGB1": "Cell adhesion",
    "VSNL1": "Calcium signaling in neurons",
    "MDH1": "Energy metabolism",
    "NDUFA4": "Mitochondrial function",
    "BEX1": "Neuronal function",
    "SLC12A5": "GABAergic neurotransmission",
    "AEBP1": "Inflammatory response",
    "IER3": "Apoptosis/stress response",
    "ACADL": "Fatty acid metabolism",
    "DECR1": "Fatty acid metabolism",
    "HEXIM1": "Cell cycle regulation",
    "HIST1H2AJ": "Epigenetic regulation",
    "TCEB3B": "Transcription regulation",
}

for i, (idx, row) in enumerate(top_genes.iterrows(), 1):
    gene = row["Gene"]
    known = "✅" if gene in known_dementia_genes else "?"
    function = known_dementia_genes.get(gene, "Function not annotated in our database")
    print(f"{i:2d}. {gene:15s} | Score: {row['final_score']:.4f} | {known} {function}")

# Step 10: Create visualizations
print("\n" + "=" * 80)
print("STEP 10: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: PCA with top genes highlighted
ax1 = axes[0, 0]
scatter = ax1.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=gene_stats["final_score"],
    cmap="viridis",
    alpha=0.6,
    s=20,
)
ax1.set_title("PCA - Gene Importance Score")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
plt.colorbar(scatter, ax=ax1, label="Importance Score")

# Highlight top genes
top_indices = gene_stats.head(10).index
for idx in top_indices:
    ax1.annotate(
        gene_stats.loc[idx, "Gene"],
        (X_pca[idx, 0], X_pca[idx, 1]),
        fontsize=8,
        alpha=0.8,
    )

# Plot 2: Top genes barplot
ax2 = axes[0, 1]
top10 = gene_stats.head(10)
ax2.barh(top10["Gene"], top10["final_score"], color="coral")
ax2.set_xlabel("Final Score")
ax2.set_title("Top 10 Genes by Importance Score")

# Plot 3: Effect score vs Consistency
ax3 = axes[0, 2]
ax3.scatter(
    gene_stats["consistency"],
    gene_stats["effect_score"],
    c=gene_stats["final_score"],
    cmap="viridis",
    alpha=0.5,
    s=10,
)
ax3.set_xlabel("Consistency")
ax3.set_ylabel("Effect Score")
ax3.set_title("Gene Distribution")
# Highlight top genes
for idx in top_indices:
    ax3.annotate(
        gene_stats.loc[idx, "Gene"],
        (gene_stats.loc[idx, "consistency"], gene_stats.loc[idx, "effect_score"]),
        fontsize=8,
        alpha=0.8,
    )

# Plot 4: Feature importance from anomaly detection
ax4 = axes[1, 0]
feature_importance = np.abs(X_scaled[top_indices].mean(axis=0))
feature_names = [
    "mean_logFC",
    "max_abs_logFC",
    "mean_p",
    "max_p",
    "n_measured",
    "consistency",
]
ax4.barh(feature_names, feature_importance, color="steelblue")
ax4.set_xlabel("Feature Importance")
ax4.set_title("Most Important Features")

# Plot 5: Score distribution
ax5 = axes[1, 1]
ax5.hist(gene_stats["final_score"], bins=50, color="green", alpha=0.7)
ax5.axvline(
    gene_stats["final_score"].quantile(0.95),
    color="red",
    linestyle="--",
    label="95th percentile",
)
ax5.set_xlabel("Final Score")
ax5.set_ylabel("Frequency")
ax5.set_title("Distribution of Gene Importance Scores")
ax5.legend()

# Plot 6: Dataset coverage
ax6 = axes[1, 2]
coverage = gene_stats["n_measured"].value_counts().sort_index()
ax6.bar(coverage.index, coverage.values, color="purple")
ax6.set_xlabel("Number of Datasets Gene is Present In")
ax6.set_ylabel("Number of Genes")
ax6.set_title("Gene Coverage Across Datasets")

plt.tight_layout()
plt.savefig(OUT / "unsupervised_ml_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Visualization saved")

# Step 11: Save results
print("\n" + "=" * 80)
print("STEP 11: SAVING RESULTS")
print("=" * 80)

gene_stats.to_csv(OUT / "gene_importance_scores.csv", index=False)
top_genes.to_csv(OUT / "top_10_genes_unsupervised.csv", index=False)

# Create final report
report_file = OUT / "unsupervised_analysis_report.txt"
with open(report_file, "w") as f:
    f.write("=" * 80 + "\n")
    f.write("UNSUPERVISED ML ANALYSIS - FINAL REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Genes Analyzed: {len(gene_stats)}\n\n")

    f.write("=" * 80 + "\n")
    f.write("TOP 10 GENES ASSOCIATED WITH DEMENTIA\n")
    f.write("=" * 80 + "\n\n")

    for i, (idx, row) in enumerate(top_genes.iterrows(), 1):
        f.write(f"{i:2d}. {row['Gene']:15s} | Score: {row['final_score']:.4f} | ")
        f.write(f"Effect: {row['effect_score']:.3f} | ")
        f.write(f"Consistency: {row['consistency']:.3f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("BIOLOGICAL INTERPRETATION\n")
    f.write("=" * 80 + "\n\n")

    for gene in top_genes["Gene"]:
        if gene in known_dementia_genes:
            f.write(f"  - {gene}: {known_dementia_genes[gene]}\n")
        else:
            f.write(f"  - {gene}: Potentially novel dementia-associated gene\n")

print(f"✅ Report saved: {report_file}")

print("\n" + "=" * 80)
print("UNSUPERVISED ML ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nAll results saved to: {OUT}")
print("=" * 80)
