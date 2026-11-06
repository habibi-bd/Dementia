from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

# =============================================================================
# STEP 3
# SAMPLE METADATA + PHENOTYPE VALIDATION
# =============================================================================

BIOLOGY_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BIOLOGY_DIR.parent.parent

STEP02_DIR = BIOLOGY_DIR / "step02_parsed"
OUTPUT_DIR = BIOLOGY_DIR / "step03_metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "GSE1297",
    "GSE4757",
    "GSE5281",
    "GSE6834",
    "GSE15222",
    "GSE16759",
    "GSE26927",
    "GSE28146",
    "GSE29378",
    "GSE33000",
    "GSE36980",
    "GSE37263",
    "GSE44770",
]

# =============================================================================
# PHENOTYPE KEYWORDS
# =============================================================================
# Conservative rules:
#
# CONTROL:
#   normal, control, healthy, non-demented, nondemented,
#   cognitively normal, normal control, healthy control
#
# AD / DEMENTIA:
#   alzheimer, alzheimer's, alzheimers, ad,
#   dementia, alzheimer disease, alzheimer's disease
#
# OTHER DISEASE:
#   epilepsy, parkinson, huntington, schizophrenia,
#   bipolar, stroke, tumor, cancer, etc.
#
# IMPORTANT:
#   We do NOT treat "disease", "patient", "case", etc.
#   as AD automatically.
# =============================================================================

CONTROL_PATTERNS = [
    r"\bhealthy\b",
    r"\bcontrol\b",
    r"\bnormal\b",
    r"\bnon[- ]?demented\b",
    r"\bnondemented\b",
    r"\bcognitively normal\b",
    r"\bnormal control\b",
    r"\bhealthy control\b",
]

AD_PATTERNS = [
    r"\balzheimer(?:'s)?\b",
    r"\balzheimers\b",
    r"\balzheimer disease\b",
    r"\balzheimer's disease\b",
    r"\bdementia\b",
    r"\bad\b",
    r"\bad patient\b",
    r"\bads\b",
]

OTHER_DISEASE_PATTERNS = [
    r"\bparkinson(?:'s)?\b",
    r"\bhuntington(?:'s)?\b",
    r"\bepilepsy\b",
    r"\bepileptic\b",
    r"\bschizophrenia\b",
    r"\bbipolar\b",
    r"\bstroke\b",
    r"\bmultiple sclerosis\b",
    r"\bam yotrophic lateral sclerosis\b",
    r"\bals\b",
    r"\btumou?r\b",
    r"\bcancer\b",
    r"\bcarcinoma\b",
    r"\bdiabetes\b",
]


# =============================================================================
# HELPERS
# =============================================================================


def normalize_text(value):
    """Convert metadata value into searchable normalized text."""
    if pd.isna(value):
        return ""

    text = str(value)

    # Remove common GEO formatting artifacts
    text = text.replace('"', " ")
    text = text.replace("'", "'")

    # Normalize separators
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def matches_any(text, patterns):
    """Return matching regex patterns."""
    matches = []

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            matches.append(pattern)

    return matches


def classify_sample(row):
    """
    Conservative classification using all metadata fields.

    Priority:
        1. Explicit OTHER_DISEASE evidence
        2. Explicit AD/DEMENTIA evidence
        3. Explicit CONTROL evidence
        4. Otherwise UNCERTAIN

    Conflicting evidence -> UNCERTAIN
    """

    # Combine all metadata fields except internal bookkeeping columns.
    metadata_parts = []

    for col, value in row.items():

        if col in {
            "GSE",
            "sample_index",
            "Phenotype",
            "Include",
            "Classification_reason",
            "Matched_terms",
        }:
            continue

        if pd.notna(value):
            metadata_parts.append(str(value))

    combined = normalize_text(" | ".join(metadata_parts))

    ad_matches = matches_any(combined, AD_PATTERNS)
    control_matches = matches_any(combined, CONTROL_PATTERNS)
    other_matches = matches_any(combined, OTHER_DISEASE_PATTERNS)

    # -------------------------------------------------------------------------
    # OTHER DISEASE
    # -------------------------------------------------------------------------
    # If clear non-AD disease evidence exists, do not accidentally classify it
    # as control just because the metadata contains "control".
    if other_matches:

        return pd.Series(
            {
                "Phenotype": "OTHER_DISEASE",
                "Include": "NO",
                "Classification_reason": "Other disease keyword detected",
                "Matched_terms": "; ".join(other_matches),
            }
        )

    # -------------------------------------------------------------------------
    # CONFLICTING AD + CONTROL
    # -------------------------------------------------------------------------
    if ad_matches and control_matches:

        return pd.Series(
            {
                "Phenotype": "UNCERTAIN",
                "Include": "NO",
                "Classification_reason": "Conflicting AD/dementia and control evidence",
                "Matched_terms": "; ".join(ad_matches + control_matches),
            }
        )

    # -------------------------------------------------------------------------
    # AD / DEMENTIA
    # -------------------------------------------------------------------------
    if ad_matches:

        return pd.Series(
            {
                "Phenotype": "AD_DEMENTIA",
                "Include": "YES",
                "Classification_reason": "AD/dementia keyword detected",
                "Matched_terms": "; ".join(ad_matches),
            }
        )

    # -------------------------------------------------------------------------
    # CONTROL
    # -------------------------------------------------------------------------
    if control_matches:

        return pd.Series(
            {
                "Phenotype": "CONTROL",
                "Include": "YES",
                "Classification_reason": "Control/healthy/normal keyword detected",
                "Matched_terms": "; ".join(control_matches),
            }
        )

    # -------------------------------------------------------------------------
    # UNCERTAIN
    # -------------------------------------------------------------------------
    return pd.Series(
        {
            "Phenotype": "UNCERTAIN",
            "Include": "NO",
            "Classification_reason": "No sufficiently specific phenotype evidence",
            "Matched_terms": "",
        }
    )


# =============================================================================
# LOCATE STEP 2 FILES
# =============================================================================


def find_dataset_files(dataset):
    """
    Find Step 2 metadata/expression files.

    We deliberately inspect filenames rather than assuming one exact naming
    convention because Step 2 may have produced different metadata filenames.
    """

    files = list(STEP02_DIR.glob("*"))

    dataset_files = [f for f in files if dataset.lower() in f.name.lower()]

    return dataset_files


def identify_metadata_file(dataset):
    """
    Identify metadata CSV generated by Step 2.
    """

    candidates = []

    for f in find_dataset_files(dataset):

        name = f.name.lower()

        if f.suffix.lower() != ".csv":
            continue

        if "metadata" in name or "meta" in name or "sample" in name:
            candidates.append(f)

    # Prefer explicit metadata names
    candidates = sorted(
        candidates,
        key=lambda x: (0 if "metadata" in x.name.lower() else 1, len(x.name)),
    )

    return candidates[0] if candidates else None


# =============================================================================
# PROCESS ONE DATASET
# =============================================================================


def process_dataset(dataset):

    print()
    print("=" * 80)
    print(dataset)
    print("=" * 80)

    metadata_file = identify_metadata_file(dataset)

    if metadata_file is None:

        print("ERROR: Metadata file not found in Step 2 output.")

        return None

    print(f"Metadata: {metadata_file.name}")

    try:
        metadata = pd.read_csv(metadata_file, dtype=str, keep_default_na=False)
    except Exception as e:

        print(f"ERROR reading metadata: {e}")

        return None

    if metadata.empty:

        print("ERROR: Metadata table is empty.")

        return None

    # -------------------------------------------------------------------------
    # Clean column names
    # -------------------------------------------------------------------------

    metadata.columns = [str(c).strip().replace('"', "") for c in metadata.columns]

    # -------------------------------------------------------------------------
    # Identify GSM column
    # -------------------------------------------------------------------------

    gsm_col = None

    preferred = [
        "GSM",
        "geo_accession",
        "!Sample_geo_accession",
        "sample_id",
        "sample",
    ]

    for candidate in preferred:

        for col in metadata.columns:

            if col.lower() == candidate.lower():
                gsm_col = col
                break

        if gsm_col is not None:
            break

    # Fallback: search for GSM-like column
    if gsm_col is None:

        for col in metadata.columns:

            values = metadata[col].astype(str)

            if values.str.match(r"^GSM\d+$", case=False).mean() > 0.5:

                gsm_col = col
                break

    if gsm_col is None:

        print("WARNING: Could not identify GSM column.")

        metadata.insert(
            0, "GSM", [f"{dataset}_SAMPLE_{i+1}" for i in range(len(metadata))]
        )

        gsm_col = "GSM"

    # -------------------------------------------------------------------------
    # Add dataset identifier
    # -------------------------------------------------------------------------

    metadata.insert(0, "GSE", dataset)

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    classification = metadata.apply(classify_sample, axis=1)

    metadata = pd.concat([metadata, classification], axis=1)

    # -------------------------------------------------------------------------
    # Sample index
    # -------------------------------------------------------------------------

    metadata.insert(2, "sample_index", np.arange(len(metadata)))

    # -------------------------------------------------------------------------
    # Reorder important columns
    # -------------------------------------------------------------------------

    priority_columns = [
        "GSE",
        gsm_col,
        "sample_index",
        "Phenotype",
        "Include",
        "Classification_reason",
        "Matched_terms",
    ]

    remaining_columns = [c for c in metadata.columns if c not in priority_columns]

    metadata = metadata[priority_columns + remaining_columns]

    # -------------------------------------------------------------------------
    # Save dataset-specific metadata
    # -------------------------------------------------------------------------

    output_file = OUTPUT_DIR / f"{dataset}_sample_metadata_validated.csv"

    metadata.to_csv(output_file, index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    counts = metadata["Phenotype"].value_counts().to_dict()

    control_n = counts.get("CONTROL", 0)
    ad_n = counts.get("AD_DEMENTIA", 0)
    other_n = counts.get("OTHER_DISEASE", 0)
    uncertain_n = counts.get("UNCERTAIN", 0)

    included_n = int((metadata["Include"] == "YES").sum())

    print(f"Samples:          {len(metadata)}")
    print(f"CONTROL:          {control_n}")
    print(f"AD/DEMENTIA:      {ad_n}")
    print(f"OTHER DISEASE:    {other_n}")
    print(f"UNCERTAIN:        {uncertain_n}")
    print(f"Included:         {included_n}")
    print(f"Saved:            {output_file.name}")

    return {
        "GSE": dataset,
        "Total_Samples": len(metadata),
        "CONTROL": control_n,
        "AD_DEMENTIA": ad_n,
        "OTHER_DISEASE": other_n,
        "UNCERTAIN": uncertain_n,
        "Included": included_n,
        "Metadata_File": str(output_file),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 80)
    print("STEP 3 — SAMPLE METADATA + PHENOTYPE VALIDATION")
    print("=" * 80)

    print()
    print(f"Input : {STEP02_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    results = []

    for dataset in DATASETS:

        result = process_dataset(dataset)

        if result is not None:
            results.append(result)

    # =========================================================================
    # DATASET SUMMARY
    # =========================================================================

    summary = pd.DataFrame(results)

    if not summary.empty:

        summary_file = OUTPUT_DIR / "dataset_sample_summary.csv"

        summary.to_csv(summary_file, index=False)

    # =========================================================================
    # COMBINED SAMPLE MANIFEST
    # =========================================================================

    metadata_files = sorted(OUTPUT_DIR.glob("*_sample_metadata_validated.csv"))

    all_metadata = []

    for file in metadata_files:

        try:
            df = pd.read_csv(file, dtype=str)

            all_metadata.append(df)

        except Exception as e:

            print(f"WARNING: Could not read {file.name}: {e}")

    if all_metadata:

        combined = pd.concat(all_metadata, ignore_index=True)

        combined_file = OUTPUT_DIR / "all_sample_metadata.csv"

        combined.to_csv(combined_file, index=False)

        # =========================================================================
        # INCLUDED SAMPLE MANIFEST
        # =========================================================================

        included = combined[combined["Include"].astype(str).str.upper() == "YES"].copy()

        included_file = OUTPUT_DIR / "included_samples.csv"

        included.to_csv(included_file, index=False)

        # =========================================================================
        # PHENOTYPE SUMMARY
        # =========================================================================

        phenotype_summary = (
            combined.groupby(["GSE", "Phenotype"]).size().reset_index(name="N")
        )

        phenotype_file = OUTPUT_DIR / "phenotype_summary.csv"

        phenotype_summary.to_csv(phenotype_file, index=False)

        # =========================================================================
        # DATASET VALIDATION STATUS
        # =========================================================================

        validation_rows = []

        for gse in DATASETS:

            sub = combined[combined["GSE"].astype(str) == gse]

            if sub.empty:

                validation_rows.append(
                    {"GSE": gse, "Status": "FAILED", "Reason": "No metadata processed"}
                )

                continue

            control_n = (sub["Phenotype"] == "CONTROL").sum()

            ad_n = (sub["Phenotype"] == "AD_DEMENTIA").sum()

            uncertain_n = (sub["Phenotype"] == "UNCERTAIN").sum()

            if control_n > 0 and ad_n > 0:

                status = "READY_FOR_PHENOTYPE_REVIEW"

            elif control_n == 0 or ad_n == 0:

                status = "INCOMPLETE"

            else:

                status = "REVIEW"

            validation_rows.append(
                {
                    "GSE": gse,
                    "CONTROL": int(control_n),
                    "AD_DEMENTIA": int(ad_n),
                    "UNCERTAIN": int(uncertain_n),
                    "Status": status,
                }
            )

        validation = pd.DataFrame(validation_rows)

        validation_file = OUTPUT_DIR / "dataset_phenotype_validation_status.csv"

        validation.to_csv(validation_file, index=False)

    else:

        print()
        print("ERROR: No metadata files were successfully combined.")

    # =========================================================================
    # JSON RUN REPORT
    # =========================================================================

    report = {
        "step": 3,
        "description": "Sample metadata and phenotype validation",
        "datasets_requested": len(DATASETS),
        "datasets_processed": len(results),
        "output_directory": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "step03_run_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print()
    print("=" * 80)
    print("STEP 3 SUMMARY")
    print("=" * 80)

    print(f"Datasets processed: {len(results)}")
    print(f"Datasets requested: {len(DATASETS)}")

    if not summary.empty:

        print()
        print(
            summary[
                [
                    "GSE",
                    "Total_Samples",
                    "CONTROL",
                    "AD_DEMENTIA",
                    "OTHER_DISEASE",
                    "UNCERTAIN",
                    "Included",
                ]
            ].to_string(index=False)
        )

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("Important:")
    print("  UNCERTAIN samples are NOT included.")
    print("  Phenotype assignments must be reviewed before DE analysis.")

    print()
    print("=" * 80)
    print("STEP 3 FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
