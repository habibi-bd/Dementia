# ============================================================
# STEP 8 - LIMMA DIFFERENTIAL EXPRESSION
# ============================================================

options(stringsAsFactors = FALSE)

BASE <- "D:/Dementia2"

EXPR_DIR <- file.path(
    BASE,
    "codes",
    "biolysis",
    "step07_phenotype"
)

OUT_DIR <- file.path(
    BASE,
    "codes",
    "biolysis",
    "step08_DE"
)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

library(limma)

DATASETS <- c(
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
    "GSE44770"
)

summary_list <- list()

for (GSE in DATASETS) {

    cat("\n")
    cat(paste0(rep("=", 80), collapse = ""))
    cat("\nSTEP 8 | ", GSE, "\n", sep = "")
    cat(paste0(rep("=", 80), collapse = ""))
    cat("\n")

    expr_file <- file.path(
        EXPR_DIR,
        paste0(GSE, "_case_control_expression.csv")
    )

    meta_file <- file.path(
        EXPR_DIR,
        paste0(GSE, "_curated_metadata.csv")
    )

    if (!file.exists(expr_file) || !file.exists(meta_file)) {
        cat("  Missing Step 7 files\n")
        next
    }

    # --------------------------------------------------------
    # Read expression
    # --------------------------------------------------------

    expr_df <- read.csv(
        expr_file,
        check.names = FALSE
    )

    meta <- read.csv(
        meta_file,
        check.names = FALSE
    )

    gene_col <- colnames(expr_df)[1]

    genes <- as.character(expr_df[[gene_col]])

    expression_samples <- colnames(expr_df)[-1]

    # --------------------------------------------------------
    # Numeric expression matrix
    # --------------------------------------------------------

    expr <- as.matrix(
        expr_df[, -1, drop = FALSE]
    )

    mode(expr) <- "numeric"

    rownames(expr) <- genes

    # Remove empty gene names
    keep <- !is.na(rownames(expr)) &
            rownames(expr) != "" &
            rownames(expr) != "NA"

    expr <- expr[keep, , drop = FALSE]

    # --------------------------------------------------------
    # Collapse duplicate genes
    # --------------------------------------------------------

    if (anyDuplicated(rownames(expr))) {

        cat("  Collapsing duplicate genes...\n")

        expr <- avereps(
            expr,
            ID = rownames(expr)
        )
    }

    # --------------------------------------------------------
    # Match metadata
    # --------------------------------------------------------

    matched <- intersect(
        expression_samples,
        meta$sample_id
    )

    if (length(matched) < 6) {
        cat("  SKIPPED: too few matched samples\n")
        next
    }

    expr <- expr[, matched, drop = FALSE]

    meta <- meta[
        match(matched, meta$sample_id),
        ,
        drop = FALSE
    ]

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    meta$label <- as.numeric(meta$label)

    if (!all(meta$label %in% c(0, 1))) {
        cat("  SKIPPED: invalid labels\n")
        next
    }

    n_ad <- sum(meta$label == 1)
    n_control <- sum(meta$label == 0)

    cat("  Genes      : ", nrow(expr), "\n", sep = "")
    cat("  AD samples : ", n_ad, "\n", sep = "")
    cat("  Controls   : ", n_control, "\n", sep = "")

    if (n_ad < 3 || n_control < 3) {
        cat("  SKIPPED: insufficient groups\n")
        next
    }

    # --------------------------------------------------------
    # Remove genes with too many missing values
    # --------------------------------------------------------

    valid_fraction <- rowMeans(
        !is.na(expr)
    )

    expr <- expr[
        valid_fraction >= 0.80,
        ,
        drop = FALSE
    ]

    # --------------------------------------------------------
    # Detect expression scale
    # --------------------------------------------------------

    finite_values <- expr[
        is.finite(expr)
    ]

    data_min <- min(finite_values, na.rm = TRUE)
    data_max <- max(finite_values, na.rm = TRUE)

    # --------------------------------------------------------
    # Log2 transformation
    # --------------------------------------------------------

    transformed <- FALSE

    if (data_min >= 0 && data_max > 50) {

        cat("  Scale      : raw/linear\n")
        cat("  Transform  : log2(x + 1)\n")

        expr <- log2(expr + 1)

        transformed <- TRUE

    } else {

        cat("  Scale      : log-like\n")
        cat("  Transform  : none\n")
    }

    # --------------------------------------------------------
    # Design matrix
    # --------------------------------------------------------

    group <- factor(
        ifelse(
            meta$label == 1,
            "Alzheimer",
            "Control"
        ),
        levels = c("Control", "Alzheimer")
    )

    design <- model.matrix(
        ~ 0 + group
    )

    colnames(design) <- c(
        "Control",
        "Alzheimer"
    )

    # --------------------------------------------------------
    # Remove problematic rows
    # --------------------------------------------------------

    good_rows <- apply(
        expr,
        1,
        function(x) {
            sum(is.finite(x)) >= 4 &&
            sd(x, na.rm = TRUE) > 0
        }
    )

    expr <- expr[
        good_rows,
        ,
        drop = FALSE
    ]

    # --------------------------------------------------------
    # LIMMA
    # --------------------------------------------------------

    fit <- lmFit(
        expr,
        design
    )

    contrast_matrix <- makeContrasts(
        Alzheimer - Control,
        levels = design
    )

    fit2 <- contrasts.fit(
        fit,
        contrast_matrix
    )

    fit2 <- eBayes(
        fit2,
        trend = TRUE,
        robust = TRUE
    )

    # --------------------------------------------------------
    # Extract ALL genes
    # --------------------------------------------------------

    results <- topTable(
        fit2,
        number = Inf,
        adjust.method = "BH",
        sort.by = "P"
    )

    results$GeneSymbol <- rownames(results)

    # --------------------------------------------------------
    # Rename columns
    # --------------------------------------------------------

    results <- results[, c(
        "GeneSymbol",
        "logFC",
        "AveExpr",
        "t",
        "P.Value",
        "adj.P.Val",
        "B"
    )]

    colnames(results) <- c(
        "GeneSymbol",
        "logFC",
        "AveExpr",
        "t_stat",
        "p_value",
        "adj_p_value",
        "B"
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    results$direction <- ifelse(
        results$logFC > 0,
        "UP_IN_ALZHEIMER",
        "DOWN_IN_ALZHEIMER"
    )

    # --------------------------------------------------------
    # Significance
    # --------------------------------------------------------

    results$significant_FDR_0.05 <-
        results$adj_p_value < 0.05

    results$significant_FDR_0.10 <-
        results$adj_p_value < 0.10

    results$abs_logFC_ge_1 <-
        abs(results$logFC) >= 1

    results$DE_strict <-
        results$adj_p_value < 0.05 &
        abs(results$logFC) >= 1

    results$DE_exploratory <-
        results$adj_p_value < 0.10 &
        abs(results$logFC) >= 0.5

    # --------------------------------------------------------
    # Save complete DE table
    # --------------------------------------------------------

    write.csv(
        results,
        file.path(
            OUT_DIR,
            paste0(GSE, "_DE.csv")
        ),
        row.names = FALSE
    )

    # --------------------------------------------------------
    # Strict DE
    # --------------------------------------------------------

    strict <- results[
        results$DE_strict,
        ,
        drop = FALSE
    ]

    write.csv(
        strict,
        file.path(
            OUT_DIR,
            paste0(GSE, "_DE_strict.csv")
        ),
        row.names = FALSE
    )

    # --------------------------------------------------------
    # Top 100
    # --------------------------------------------------------

    top100 <- head(
        results,
        100
    )

    write.csv(
        top100,
        file.path(
            OUT_DIR,
            paste0(GSE, "_top100_DE.csv")
        ),
        row.names = FALSE
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    fdr05 <- sum(
        results$adj_p_value < 0.05,
        na.rm = TRUE
    )

    fdr10 <- sum(
        results$adj_p_value < 0.10,
        na.rm = TRUE
    )

    strict_n <- sum(
        results$DE_strict,
        na.rm = TRUE
    )

    cat("  Tested genes : ", nrow(results), "\n", sep = "")
    cat("  FDR < 0.05   : ", fdr05, "\n", sep = "")
    cat("  FDR < 0.10   : ", fdr10, "\n", sep = "")
    cat("  Strict DE    : ", strict_n, "\n", sep = "")

    summary_list[[length(summary_list) + 1]] <- data.frame(
        GSE = GSE,
        Genes = nrow(results),
        Alzheimer = n_ad,
        Control = n_control,
        Tested_Genes = nrow(results),
        FDR_0.05 = fdr05,
        FDR_0.10 = fdr10,
        Strict_DE = strict_n,
        Log_Transformed = transformed
    )
}

# ============================================================
# SAVE SUMMARY
# ============================================================

if (length(summary_list) > 0) {

    summary_df <- do.call(
        rbind,
        summary_list
    )

    write.csv(
        summary_df,
        file.path(
            OUT_DIR,
            "step08_DE_summary.csv"
        ),
        row.names = FALSE
    )

    cat("\n")
    cat(paste0(rep("=", 80), collapse = ""))
    cat("\nSTEP 8 SUMMARY\n")
    cat(paste0(rep("=", 80), collapse = ""))
    cat("\n")

    print(
        summary_df,
        row.names = FALSE
    )
}