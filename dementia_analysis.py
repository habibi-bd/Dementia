import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import gzip
import warnings

warnings.filterwarnings("ignore")

# For handling GEO data
from sklearn.preprocessing import StandardScaler


class DementiaGEOAnalyzer:
    def __init__(self, base_dir="D:/Dementia2/data"):
        """
        Initialize the analyzer with directory containing GEO data files

        Parameters:
        -----------
        base_dir : str
            Base directory containing the data (default: D:/Dementia2/data)
        """
        self.base_dir = base_dir
        self.gpl_dir = os.path.join(base_dir, "gpl")
        self.matrix_dir = os.path.join(base_dir, "matrix")
        self.gpl_annotations = {}
        self.gse_data = {}
        self.results = {}

        # Create directories if they don't exist
        os.makedirs(self.gpl_dir, exist_ok=True)
        os.makedirs(self.matrix_dir, exist_ok=True)

    def load_annotation(self, gpl_id):
        """
        Load GPL annotation file for probe-to-gene mapping

        Parameters:
        -----------
        gpl_id : str
            GPL platform ID (e.g., 'GPL570')
        """
        annotation_files = {
            "GPL96": ["GPL96_annotation.csv", "GPL96.txt", "GPL96.soft", "GPL96.annot"],
            "GPL570": ["GPL570_annotation.csv", "GPL570.txt", "GPL570.soft"],
            "GPL2700": ["GPL2700.txt", "GPL2700.annot.gz"],
            "GPL4372": ["GPL4372_annotation.csv", "GPL4372.annot.gz"],
            "GPL4757": ["GPL4757.txt", "GPL4757.annot.gz"],
            "GPL5175": ["GPL5175.txt", "GPL5175.annot.gz", "GPL5175 (2).txt"],
            "GPL6244": ["GPL6244_annotation.csv", "GPL6244.annot.gz"],
            "GPL6255": ["GPL6255.txt", "GPL6255.annot.gz"],
            "GPL6947": ["GPL6947_annotation.csv", "GPL6947.annot.gz"],
            "GPL8757": ["GPL8757.soft"],
        }

        if gpl_id not in annotation_files:
            print(f"Warning: No annotation file found for {gpl_id}")
            return None

        annot_df = None
        for file_pattern in annotation_files[gpl_id]:
            # Search in gpl directory
            for file in os.listdir(self.gpl_dir):
                if file_pattern in file and file.endswith(
                    (".txt", ".csv", ".soft", ".gz", "")
                ):
                    file_path = os.path.join(self.gpl_dir, file)
                    try:
                        if file.endswith(".gz"):
                            with gzip.open(file_path, "rt") as f:
                                if "annot" in file or "annotation" in file:
                                    annot_df = pd.read_csv(f, sep="\t", header=None)
                                else:
                                    annot_df = pd.read_csv(f, sep="\t")
                        elif file.endswith(".csv"):
                            annot_df = pd.read_csv(file_path)
                        elif file.endswith(".txt") or file.endswith(".soft"):
                            if "annot" in file or "annotation" in file:
                                annot_df = pd.read_csv(file_path, sep="\t", header=None)
                            else:
                                # Try to read with different delimiters
                                try:
                                    annot_df = pd.read_csv(file_path, sep="\t")
                                except:
                                    annot_df = pd.read_csv(file_path, sep=",")
                        # For files without extension (like GPL96.annot)
                        elif "annot" in file:
                            annot_df = pd.read_csv(file_path, sep="\t", header=None)
                        break
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
                        continue

        if annot_df is not None:
            # Try to identify columns for probe ID and gene symbol
            probe_col = None
            gene_col = None

            # Look for common column names
            for col in annot_df.columns:
                col_lower = str(col).lower()
                if "probe" in col_lower or "id" in col_lower or "probe_id" in col_lower:
                    probe_col = col
                if (
                    "gene" in col_lower
                    or "symbol" in col_lower
                    or "gene_symbol" in col_lower
                ):
                    gene_col = col

            # If columns not found, try to guess based on position
            if probe_col is None and len(annot_df.columns) >= 1:
                probe_col = annot_df.columns[0]
            if gene_col is None and len(annot_df.columns) >= 2:
                gene_col = annot_df.columns[1]

            if probe_col is not None and gene_col is not None:
                self.gpl_annotations[gpl_id] = annot_df[[probe_col, gene_col]].copy()
                self.gpl_annotations[gpl_id].columns = ["Probe_ID", "Gene_Symbol"]
                print(
                    f"Loaded annotation for {gpl_id}: {len(self.gpl_annotations[gpl_id])} probes"
                )
            else:
                print(f"Could not find probe and gene columns in {gpl_id} annotation")
                self.gpl_annotations[gpl_id] = annot_df
        else:
            print(f"No annotation file found for {gpl_id}")

        return self.gpl_annotations.get(gpl_id)

    def load_gse_data(self, gse_id):
        """
        Load GSE series matrix file from matrix directory

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID (e.g., 'GSE1297')
        """
        series_files = []
        for file in os.listdir(self.matrix_dir):
            if f"{gse_id}_series_matrix.txt" in file:
                series_files.append(file)

        if not series_files:
            print(f"No series matrix file found for {gse_id} in {self.matrix_dir}")
            return None

        # Prefer uncompressed file if available
        series_file = None
        for f in series_files:
            if not f.endswith(".gz"):
                series_file = f
                break
        if series_file is None:
            series_file = series_files[0]

        file_path = os.path.join(self.matrix_dir, series_file)

        try:
            if file_path.endswith(".gz"):
                with gzip.open(file_path, "rt") as f:
                    data = self._parse_series_matrix(f)
            else:
                with open(file_path, "r") as f:
                    data = self._parse_series_matrix(f)

            self.gse_data[gse_id] = data
            print(
                f"Loaded {gse_id}: {data['expression'].shape[0]} genes, {data['expression'].shape[1]} samples"
            )
            return data
        except Exception as e:
            print(f"Error loading {gse_id}: {e}")
            return None

    def _parse_series_matrix(self, file_handle):
        """
        Parse GEO series matrix file
        """
        lines = file_handle.readlines()

        # Find the expression data section
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith("!series_matrix_table_begin"):
                data_start = i + 1
                break

        if data_start is None:
            raise ValueError("Could not find expression data in series matrix file")

        # Parse header
        header_line = lines[data_start].strip().split("\t")

        # Parse expression data
        expression_data = []
        for line in lines[data_start + 1 :]:
            if line.startswith("!series_matrix_table_end"):
                break
            if line.strip():
                values = line.strip().split("\t")
                if len(values) == len(header_line):
                    expression_data.append(values)

        # Convert to DataFrame
        df = pd.DataFrame(expression_data, columns=header_line)
        df.set_index(header_line[0], inplace=True)

        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Extract sample metadata from header
        metadata = {}
        for line in lines:
            if line.startswith("!Sample_title"):
                sample_titles = line.strip().split("\t")[1:]
                metadata["sample_titles"] = sample_titles
            elif line.startswith("!Sample_characteristics_ch1"):
                characteristics = line.strip().split("\t")[1:]
                metadata["characteristics"] = characteristics

        return {"expression": df, "metadata": metadata, "header": header_line}

    def map_probes_to_genes(self, gse_id, gpl_id):
        """
        Map probe IDs to gene symbols for a given dataset

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        gpl_id : str
            GPL platform ID
        """
        if gse_id not in self.gse_data:
            print(f"Dataset {gse_id} not loaded")
            return None

        if gpl_id not in self.gpl_annotations:
            print(f"Annotation for {gpl_id} not loaded")
            return None

        expr_df = self.gse_data[gse_id]["expression"]
        annot_df = self.gpl_annotations[gpl_id]

        # Handle case where annot_df has different structure
        if "Probe_ID" not in annot_df.columns:
            # Try to use first column as probe ID
            annot_df = annot_df.copy()
            annot_df.columns = ["Probe_ID"] + list(annot_df.columns[1:])
            if "Gene_Symbol" not in annot_df.columns:
                annot_df["Gene_Symbol"] = (
                    annot_df.iloc[:, 1] if len(annot_df.columns) > 1 else ""
                )

        # Merge expression data with annotation
        merged_df = expr_df.merge(
            annot_df, left_index=True, right_on="Probe_ID", how="inner"
        )

        # Group by gene symbol and take mean expression for multiple probes
        if "Gene_Symbol" in merged_df.columns:
            # Remove empty gene symbols
            merged_df = merged_df[merged_df["Gene_Symbol"].notna()]
            merged_df = merged_df[merged_df["Gene_Symbol"] != ""]

            # Group by gene symbol
            gene_expr = merged_df.groupby("Gene_Symbol").mean()

            return gene_expr
        else:
            print(f"No Gene_Symbol column in annotation for {gpl_id}")
            return merged_df

    def perform_differential_expression(
        self,
        gse_id,
        group1_samples,
        group2_samples,
        gpl_id=None,
        log2_transform=True,
        normalize=True,
    ):
        """
        Perform differential expression analysis between two groups

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        group1_samples : list
            Sample IDs for group 1 (e.g., dementia)
        group2_samples : list
            Sample IDs for group 2 (e.g., control)
        gpl_id : str
            GPL platform ID for annotation
        log2_transform : bool
            Whether to log2 transform the data
        normalize : bool
            Whether to normalize the data

        Returns:
        --------
        dict : Differential expression results with log2FC, p-values, adjusted p-values
        """
        if gse_id not in self.gse_data:
            print(f"Dataset {gse_id} not loaded")
            return None

        expr_df = self.gse_data[gse_id]["expression"]

        # Map probes to genes if annotation is provided
        if gpl_id is not None and gpl_id in self.gpl_annotations:
            expr_df = self.map_probes_to_genes(gse_id, gpl_id)
            if expr_df is None:
                return None

        # Select samples
        available_samples = [
            s for s in group1_samples + group2_samples if s in expr_df.columns
        ]
        if len(available_samples) < len(group1_samples) + len(group2_samples):
            print(
                f"Warning: Not all samples found. Using {len(available_samples)} samples"
            )

        # Filter to available samples
        expr_sel = expr_df[available_samples]

        # Identify group labels
        group_labels = []
        for sample in available_samples:
            if sample in group1_samples:
                group_labels.append("group1")
            else:
                group_labels.append("group2")

        # Log2 transform if requested
        if log2_transform:
            expr_sel = np.log2(expr_sel + 1)

        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            expr_sel = pd.DataFrame(
                scaler.fit_transform(expr_sel.T).T,
                index=expr_sel.index,
                columns=expr_sel.columns,
            )

        # Perform t-test for each gene
        results = []
        for gene in expr_sel.index:
            group1_expr = expr_sel.loc[
                gene,
                [
                    s
                    for i, s in enumerate(available_samples)
                    if group_labels[i] == "group1"
                ],
            ].values
            group2_expr = expr_sel.loc[
                gene,
                [
                    s
                    for i, s in enumerate(available_samples)
                    if group_labels[i] == "group2"
                ],
            ].values

            # Remove NaN values
            group1_expr = group1_expr[~np.isnan(group1_expr)]
            group2_expr = group2_expr[~np.isnan(group2_expr)]

            if len(group1_expr) > 1 and len(group2_expr) > 1:
                # Calculate fold change
                fc = np.mean(group1_expr) - np.mean(group2_expr)

                # Perform t-test
                t_stat, p_val = stats.ttest_ind(
                    group1_expr, group2_expr, equal_var=False
                )

                results.append(
                    {
                        "Gene": gene,
                        "log2FC": fc,
                        "p_value": p_val,
                        "group1_mean": np.mean(group1_expr),
                        "group2_mean": np.mean(group2_expr),
                    }
                )

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Calculate adjusted p-values (FDR)
        if len(results_df) > 0:
            results_df["adj_p_value"] = multipletests(
                results_df["p_value"], method="fdr_bh"
            )[1]

            # Determine regulation direction
            results_df["regulation"] = "NS"
            results_df.loc[
                (results_df["log2FC"] > 1) & (results_df["adj_p_value"] < 0.05),
                "regulation",
            ] = "Upregulated"
            results_df.loc[
                (results_df["log2FC"] < -1) & (results_df["adj_p_value"] < 0.05),
                "regulation",
            ] = "Downregulated"

            # Sort by absolute log2FC and p-value
            results_df["abs_log2FC"] = np.abs(results_df["log2FC"])
            results_df = results_df.sort_values(
                ["abs_log2FC", "p_value"], ascending=[False, True]
            )

            self.results[gse_id] = results_df
            return results_df
        else:
            print("No results generated")
            return None

    def get_top_genes(self, gse_id, n=10, by="log2FC"):
        """
        Get top N genes based on differential expression

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        n : int
            Number of top genes to return
        by : str
            Criteria for ranking ('log2FC', 'p_value', 'adj_p_value')
        """
        if gse_id not in self.results:
            print(f"No results for {gse_id}")
            return None

        results_df = self.results[gse_id]

        # Filter significant genes
        sig_genes = results_df[results_df["adj_p_value"] < 0.05]

        if len(sig_genes) == 0:
            print(f"No significant genes found for {gse_id}")
            # Return top genes by log2FC regardless of significance
            sig_genes = results_df

        # Sort by specified criteria
        if by == "log2FC":
            sig_genes = sig_genes.sort_values("abs_log2FC", ascending=False)
        elif by == "p_value":
            sig_genes = sig_genes.sort_values("p_value", ascending=True)
        else:
            sig_genes = sig_genes.sort_values("adj_p_value", ascending=True)

        return sig_genes.head(n)

    def create_volcano_plot(
        self,
        gse_id,
        title=None,
        output_dir="D:/Dementia2/outputs",
        p_value_threshold=0.05,
        fc_threshold=1,
    ):
        """
        Create volcano plot for differential expression results

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        title : str
            Plot title
        output_dir : str
            Output directory for saving plots
        p_value_threshold : float
            P-value threshold for significance
        fc_threshold : float
            Log2 fold change threshold for significance
        """
        if gse_id not in self.results:
            print(f"No results for {gse_id}")
            return None

        results_df = self.results[gse_id]

        if len(results_df) == 0:
            print("No results to plot")
            return None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create figure with high resolution
        plt.figure(figsize=(12, 10), dpi=300)

        # Define color scheme
        colors = {"Upregulated": "#e74c3c", "Downregulated": "#3498db", "NS": "#bdc3c7"}

        # Filter based on thresholds
        results_df["-log10_p"] = -np.log10(results_df["p_value"])

        # Plot all points
        for reg in ["Upregulated", "Downregulated", "NS"]:
            subset = results_df[results_df["regulation"] == reg]
            if len(subset) > 0:
                plt.scatter(
                    subset["log2FC"],
                    subset["-log10_p"],
                    c=colors[reg],
                    label=reg,
                    alpha=0.6,
                    s=20,
                )

        # Add threshold lines
        plt.axhline(
            y=-np.log10(p_value_threshold), color="black", linestyle="--", alpha=0.5
        )
        plt.axvline(x=fc_threshold, color="black", linestyle="--", alpha=0.5)
        plt.axvline(x=-fc_threshold, color="black", linestyle="--", alpha=0.5)

        # Add labels and title
        plt.xlabel("log2 Fold Change", fontsize=14)
        plt.ylabel("-log10 p-value", fontsize=14)

        # Add gene names for top upregulated and downregulated genes
        top_up = results_df[results_df["regulation"] == "Upregulated"].head(10)
        top_down = results_df[results_df["regulation"] == "Downregulated"].head(10)

        for _, row in top_up.iterrows():
            plt.annotate(
                row["Gene"],
                (row["log2FC"], row["-log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        for _, row in top_down.iterrows():
            plt.annotate(
                row["Gene"],
                (row["log2FC"], row["-log10_p"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        # Set title
        if title is None:
            title = f"Differential Expression Analysis - {gse_id}"
        plt.title(title, fontsize=16)

        # Add legend
        plt.legend(loc="upper left")

        # Add statistics
        n_sig = len(results_df[results_df["adj_p_value"] < 0.05])
        n_up = len(results_df[results_df["regulation"] == "Upregulated"])
        n_down = len(results_df[results_df["regulation"] == "Downregulated"])

        stats_text = f"Significant genes (FDR < 0.05): {n_sig}\nUpregulated: {n_up}\nDownregulated: {n_down}"
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        output_file = os.path.join(output_dir, f"{gse_id}_volcano.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Volcano plot saved to {output_file}")

        plt.show()
        plt.close()

        return output_file

    def create_heatmap(self, gse_id, n_genes=50, output_dir="D:/Dementia2/outputs"):
        """
        Create heatmap for top differentially expressed genes

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        n_genes : int
            Number of top genes to include
        output_dir : str
            Output directory for saving plots
        """
        if gse_id not in self.results:
            print(f"No results for {gse_id}")
            return None

        if gse_id not in self.gse_data:
            print(f"Expression data for {gse_id} not loaded")
            return None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        results_df = self.results[gse_id]
        expr_df = self.gse_data[gse_id]["expression"]

        # Get top genes
        top_genes = results_df.head(n_genes)["Gene"].tolist()

        # Filter expression data for these genes
        expr_subset = expr_df.loc[top_genes]

        # Normalize for heatmap
        scaler = StandardScaler()
        expr_norm = pd.DataFrame(
            scaler.fit_transform(expr_subset.T).T,
            index=expr_subset.index,
            columns=expr_subset.columns,
        )

        # Create heatmap
        plt.figure(figsize=(12, 10), dpi=300)

        sns.heatmap(
            expr_norm,
            cmap="RdBu_r",
            center=0,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "Z-score"},
        )

        plt.title(
            f"Top {n_genes} Differentially Expressed Genes - {gse_id}", fontsize=16
        )
        plt.xlabel("Samples", fontsize=14)
        plt.ylabel("Genes", fontsize=14)

        plt.tight_layout()

        output_file = os.path.join(output_dir, f"{gse_id}_heatmap.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {output_file}")

        plt.show()
        plt.close()

        return output_file

    def run_complete_analysis(
        self,
        gse_id,
        group1_samples,
        group2_samples,
        gpl_id=None,
        output_dir="D:/Dementia2/outputs",
    ):
        """
        Run complete analysis pipeline for a dataset

        Parameters:
        -----------
        gse_id : str
            GSE dataset ID
        group1_samples : list
            Sample IDs for group 1
        group2_samples : list
            Sample IDs for group 2
        gpl_id : str
            GPL platform ID
        output_dir : str
            Output directory for saving results
        """
        print(f"Starting analysis for {gse_id}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load data
        print("Loading GSE data...")
        self.load_gse_data(gse_id)

        if gpl_id:
            print(f"Loading annotation for {gpl_id}...")
            self.load_annotation(gpl_id)

        # Perform differential expression
        print("Performing differential expression analysis...")
        results = self.perform_differential_expression(
            gse_id, group1_samples, group2_samples, gpl_id
        )

        if results is None or len(results) == 0:
            print("Analysis failed or no results")
            return None

        # Get top genes
        print("\nTop 10 differentially expressed genes:")
        top_genes = self.get_top_genes(gse_id, n=10)
        if top_genes is not None:
            print(
                top_genes[
                    ["Gene", "log2FC", "p_value", "adj_p_value", "regulation"]
                ].to_string()
            )

        # Create volcano plot
        print("\nCreating volcano plot...")
        self.create_volcano_plot(gse_id, output_dir=output_dir)

        # Create heatmap
        print("Creating heatmap...")
        self.create_heatmap(gse_id, output_dir=output_dir)

        # Save results to CSV
        results_file = os.path.join(output_dir, f"{gse_id}_results.csv")
        results.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")

        # Save top genes to CSV
        if top_genes is not None:
            top_file = os.path.join(output_dir, f"{gse_id}_top10_genes.csv")
            top_genes.to_csv(top_file, index=False)
            print(f"Top 10 genes saved to {top_file}")

        return results


# Function to get sample groups from grouping.csv file
def load_grouping_info(base_dir="D:/Dementia2/data"):
    """
    Load sample grouping information from grouping.csv
    """
    grouping_file = os.path.join("D:/Dementia2", "grouping.csv")
    if os.path.exists(grouping_file):
        df = pd.read_csv(grouping_file)
        print(f"Loaded grouping information from {grouping_file}")
        return df
    else:
        print(f"Grouping file not found at {grouping_file}")
        return None


# Function to get sample IDs from series matrix
def get_sample_ids(gse_id, base_dir="D:/Dementia2/data"):
    """
    Get sample IDs from a series matrix file
    """
    analyzer = DementiaGEOAnalyzer(base_dir=base_dir)
    data = analyzer.load_gse_data(gse_id)
    if data:
        return data["expression"].columns.tolist()
    return []


# Main analysis function
def run_dementia_analysis():
    """
    Main function to run the dementia analysis
    """
    # Initialize analyzer with correct base directory
    analyzer = DementiaGEOAnalyzer(base_dir="D:/Dementia2/data")

    # Load grouping information if available
    grouping_df = load_grouping_info()
    if grouping_df is not None:
        print("\nGrouping information available:")
        print(grouping_df.head())

    # Example: Analyze GSE5281 (Alzheimer's dataset)
    gse_id = "GSE5281"
    gpl_id = "GPL570"  # Platform for GSE5281

    # Get sample IDs from the dataset
    print(f"\nLoading {gse_id} to get sample IDs...")
    sample_ids = get_sample_ids(gse_id)

    if sample_ids:
        print(f"Found {len(sample_ids)} samples in {gse_id}")
        print(f"Sample IDs: {sample_ids[:5]}...")  # Show first 5

        # If you have grouping information, use it
        if grouping_df is not None and gse_id in grouping_df["GSE"].values:
            # Filter grouping for this GSE
            gse_grouping = grouping_df[grouping_df["GSE"] == gse_id]
            disease_samples = gse_grouping[gse_grouping["Group"] == "disease"][
                "Sample"
            ].tolist()
            control_samples = gse_grouping[gse_grouping["Group"] == "control"][
                "Sample"
            ].tolist()
            print(
                f"\nFound {len(disease_samples)} disease samples and {len(control_samples)} control samples"
            )
        else:
            # Manual grouping - you need to identify which samples are disease vs control
            # This is a placeholder - replace with actual sample IDs
            print("\nNo grouping information found for this dataset.")
            print("Please manually specify disease and control samples.")
            print(f"Available samples: {sample_ids}")

            # Example manual grouping (REPLACE WITH ACTUAL SAMPLE IDs)
            # Look at the sample IDs and determine which are disease vs control
            # You can inspect the series matrix file or use the metadata
            disease_samples = sample_ids[: len(sample_ids) // 2]  # Placeholder
            control_samples = sample_ids[len(sample_ids) // 2 :]  # Placeholder
            print(
                f"\nUsing placeholder grouping: {len(disease_samples)} disease, {len(control_samples)} control"
            )
            print("Please replace with actual sample IDs!")

        # Run analysis
        results = analyzer.run_complete_analysis(
            gse_id=gse_id,
            group1_samples=disease_samples,
            group2_samples=control_samples,
            gpl_id=gpl_id,
            output_dir="D:/Dementia2/outputs",
        )

        # Print final top 10 genes
        if results is not None and len(results) > 0:
            print("\n" + "=" * 60)
            print("TOP 10 GENES ASSOCIATED WITH DEMENTIA")
            print("=" * 60)
            top_genes = analyzer.get_top_genes(gse_id, n=10)
            if top_genes is not None:
                # Filter significant genes
                sig_top = top_genes[top_genes["adj_p_value"] < 0.05]
                if len(sig_top) > 0:
                    print(
                        sig_top[
                            ["Gene", "log2FC", "p_value", "adj_p_value", "regulation"]
                        ].to_string(index=False)
                    )
                else:
                    print("No significant genes found. Showing top by log2FC:")
                    print(
                        top_genes[
                            ["Gene", "log2FC", "p_value", "adj_p_value", "regulation"]
                        ].to_string(index=False)
                    )

                # Save to text file
                output_file = "D:/Dementia2/outputs/top_dementia_genes.txt"
                with open(output_file, "w") as f:
                    f.write("TOP 10 GENES ASSOCIATED WITH DEMENTIA\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Analysis of {gse_id} data\n")
                    f.write(f"Analysis performed on: {pd.Timestamp.now()}\n\n")
                    if len(sig_top) > 0:
                        f.write(
                            sig_top[
                                [
                                    "Gene",
                                    "log2FC",
                                    "p_value",
                                    "adj_p_value",
                                    "regulation",
                                ]
                            ].to_string(index=False)
                        )
                    else:
                        f.write(
                            top_genes[
                                [
                                    "Gene",
                                    "log2FC",
                                    "p_value",
                                    "adj_p_value",
                                    "regulation",
                                ]
                            ].to_string(index=False)
                        )
                print(f"\nResults saved to {output_file}")

        return analyzer, results
    else:
        print(f"Could not load {gse_id}")
        return None, None


# Function to inspect dataset and identify sample groups
def inspect_dataset(gse_id, base_dir="D:/Dementia2/data"):
    """
    Inspect a dataset to help identify sample groups
    """
    analyzer = DementiaGEOAnalyzer(base_dir=base_dir)
    data = analyzer.load_gse_data(gse_id)

    if data is None:
        print(f"Could not load {gse_id}")
        return

    print(f"\n=== Inspection of {gse_id} ===")
    print(f"Number of samples: {data['expression'].shape[1]}")
    print(f"Number of probes: {data['expression'].shape[0]}")

    # Show sample IDs
    print("\nSample IDs:")
    for i, sample in enumerate(data["expression"].columns):
        print(f"  {i+1}. {sample}")

    # Show sample titles from metadata
    if "sample_titles" in data["metadata"]:
        print("\nSample Titles:")
        for i, title in enumerate(data["metadata"]["sample_titles"]):
            print(f"  {i+1}. {title}")

    # Show characteristics from metadata
    if "characteristics" in data["metadata"]:
        print("\nSample Characteristics (first 3):")
        for i, char in enumerate(data["metadata"]["characteristics"][:3]):
            print(f"  {i+1}. {char}")

    # Try to identify disease/control groups
    disease_samples = []
    control_samples = []

    if "sample_titles" in data["metadata"]:
        for i, title in enumerate(data["metadata"]["sample_titles"]):
            sample_id = data["expression"].columns[i]
            if (
                "disease" in title.lower()
                or "alzheimer" in title.lower()
                or "ad" in title.lower()
            ):
                if "control" not in title.lower() and "normal" not in title.lower():
                    disease_samples.append(sample_id)
            elif (
                "control" in title.lower()
                or "normal" in title.lower()
                or "healthy" in title.lower()
            ):
                control_samples.append(sample_id)

    if disease_samples or control_samples:
        print("\nAutomatically identified groups:")
        print(f"  Disease samples: {disease_samples}")
        print(f"  Control samples: {control_samples}")
    else:
        print("\nCould not automatically identify groups.")
        print("Please manually inspect the sample titles/characteristics above.")

    return analyzer, data


if __name__ == "__main__":
    # First, inspect a dataset to identify sample groups
    print("Step 1: Inspecting dataset")
    gse_id = "GSE5281"  # Change to your dataset of interest
    analyzer, data = inspect_dataset(gse_id)

    if data:
        print("\n" + "=" * 60)
        print("Step 2: Running differential expression analysis")
        print("=" * 60)

        # You need to manually specify the sample groups based on inspection
        # Replace these with actual sample IDs from the inspection output
        disease_samples = ["GSM118834", "GSM118835", "GSM118836"]  # REPLACE WITH ACTUAL
        control_samples = ["GSM118837", "GSM118838", "GSM118839"]  # REPLACE WITH ACTUAL

        print("\nUsing the following groups:")
        print(f"  Disease samples: {disease_samples}")
        print(f"  Control samples: {control_samples}")
        print("\nIMPORTANT: Replace these with actual sample IDs from your data!")

        # Uncomment to run analysis with correct sample IDs
        analyzer = DementiaGEOAnalyzer(base_dir="D:/Dementia2/data")
        results = analyzer.run_complete_analysis(
            gse_id=gse_id,
            group1_samples=disease_samples,
            group2_samples=control_samples,
            gpl_id="GPL570",  # Change to match your data
            output_dir="D:/Dementia2/outputs",
        )
    else:
        print("Could not inspect dataset. Please check file paths and data.")
