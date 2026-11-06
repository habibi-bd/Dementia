import os, warnings, json

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    cohen_kappa_score,
    roc_curve,
    precision_recall_curve,
)

BASE = r"D:\Dementia2"
INFILE = os.path.join(BASE, "codes", "ml_data", "integrated_gene_expression.csv")
OUT = os.path.join(BASE, "codes", "ml_results")
os.makedirs(OUT, exist_ok=True)

SEED, TEST_SIZE, K, N_SPLITS = 42, 0.20, 500, 5


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n = len(y)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0
    npv = tn / (tn + fn) if tn + fn else 0
    return {
        "N": n,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": acc,
        "Error_rate": 1 - acc,
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "Sensitivity_Recall_TPR": sens,
        "Specificity_TNR": spec,
        "Precision_PPV": ppv,
        "NPV": npv,
        "F1": f1_score(y, pred, zero_division=0),
        "F2": (5 * ppv * sens / (4 * ppv + sens)) if 4 * ppv + sens else 0,
        "MCC": matthews_corrcoef(y, pred),
        "Cohen_kappa": cohen_kappa_score(y, pred),
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
        "Brier_score": brier_score_loss(y, p),
        "LR_positive": sens / (1 - spec) if 1 - spec > 1e-12 else np.inf,
        "LR_negative": (1 - sens) / spec if spec > 1e-12 else np.inf,
        "Diagnostic_odds_ratio": (
            (sens / (1 - sens)) / ((1 - spec) / spec)
            if sens < 1 and spec > 0 and spec < 1
            else np.inf
        ),
    }


print("=" * 72)
print("STEP 17 - COMPREHENSIVE ML + DL BENCHMARK")
print("=" * 72)

df = pd.read_csv(INFILE)
meta = ["sample_id", "condition", "label", "dataset", "platform"]
genes = [c for c in df.columns if c not in meta]
df[genes] = df[genes].apply(pd.to_numeric, errors="coerce").fillna(0)
df["label"] = pd.to_numeric(df["label"]).astype(int)

X = df[genes].values.astype("float32")
y = df.label.values
strata = df.dataset.astype(str) + "_" + df.label.astype(str)

try:
    tr, te = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=strata
    )
except ValueError:
    tr, te = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

print(f"Samples : {len(df):,}")
print(f"Genes   : {len(genes):,}")
print(f"AD      : {(y==1).sum():,}")
print(f"Control : {(y==0).sum():,}")
print(f"Train   : {len(tr):,}")
print(f"Test    : {len(te):,}")

pd.DataFrame(
    {
        "sample_id": df.iloc[te].sample_id.values,
        "dataset": df.iloc[te].dataset.values,
        "label": yte,
    }
).to_csv(os.path.join(OUT, "test_set_samples.csv"), index=False)

models = {
    "LogisticRegression": Pipeline(
        [
            ("select", SelectKBest(f_classif, k=min(K, Xtr.shape[1]))),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "SVM_RBF": Pipeline(
        [
            ("select", SelectKBest(f_classif, k=min(K, Xtr.shape[1]))),
            ("scale", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    ),
    "RandomForest": Pipeline(
        [
            ("select", SelectKBest(f_classif, k=min(K, Xtr.shape[1]))),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                    max_features="sqrt",
                ),
            ),
        ]
    ),
}

try:
    from xgboost import XGBClassifier

    models["XGBoost"] = Pipeline(
        [
            ("select", SelectKBest(f_classif, k=min(K, Xtr.shape[1]))),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
except Exception:
    print("XGBoost unavailable")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
rows = []
predout = pd.DataFrame(
    {
        "sample_id": df.iloc[te].sample_id.values,
        "dataset": df.iloc[te].dataset.values,
        "true_label": yte,
    }
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "neg_log_loss": "neg_log_loss",
}

for name, m in models.items():
    print("\n" + "-" * 60)
    print(name)
    cvres = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    z = metrics(yte, p)

    row = {"Model": name}
    for k, v in z.items():
        row["Test_" + k] = v
    for k in scoring:
        a = cvres["test_" + k]
        if k == "neg_log_loss":
            a = -a
        row["CV_" + k + "_mean"] = a.mean()
        row["CV_" + k + "_SD"] = a.std()
        row["CV_" + k + "_min"] = a.min()
        row["CV_" + k + "_max"] = a.max()
    rows.append(row)
    predout[name + "_prob"] = p
    predout[name + "_pred"] = (p >= 0.5).astype(int)
    joblib.dump(m, os.path.join(OUT, name + ".joblib"))

    # final selected genes
    sel = m.named_steps["select"]
    ix = sel.get_support()
    pd.DataFrame(
        {
            "gene": np.array(genes)[ix],
            "F_score": sel.scores_[ix],
            "p_value": sel.pvalues_[ix],
        }
    ).sort_values("F_score", ascending=False).to_csv(
        os.path.join(OUT, name + "_selected_genes.csv"), index=False
    )

    fpr, tpr, thr = roc_curve(yte, p)
    pd.DataFrame({"threshold": thr, "TPR": tpr, "FPR": fpr}).to_csv(
        os.path.join(OUT, name + "_roc_curve.csv"), index=False
    )
    prec, rec, thr2 = precision_recall_curve(yte, p)
    pd.DataFrame(
        {"threshold": np.r_[thr2, np.nan], "precision": prec, "recall": rec}
    ).to_csv(os.path.join(OUT, name + "_pr_curve.csv"), index=False)

    print(
        f"CV ROC-AUC       {row['CV_roc_auc_mean']:.4f} +/- {row['CV_roc_auc_SD']:.4f}"
    )
    print(f"TEST Accuracy    {z['Accuracy']:.4f}")
    print(f"TEST BalancedAcc {z['Balanced_accuracy']:.4f}")
    print(f"TEST Precision   {z['Precision_PPV']:.4f}")
    print(f"TEST Recall      {z['Sensitivity_Recall_TPR']:.4f}")
    print(f"TEST Specificity {z['Specificity_TNR']:.4f}")
    print(f"TEST F1          {z['F1']:.4f}")
    print(f"TEST F2          {z['F2']:.4f}")
    print(f"TEST MCC         {z['MCC']:.4f}")
    print(f"TEST ROC-AUC     {z['ROC_AUC']:.4f}")
    print(f"TEST PR-AUC      {z['PR_AUC']:.4f}")

# ---------------- threshold analysis ----------------
threshold_rows = []
for name in models:
    p = predout[name + "_prob"].values
    for t in np.arange(0.10, 0.91, 0.05):
        z = metrics(yte, p, float(t))
        threshold_rows.append({"Model": name, "Threshold": t, **z})
pd.DataFrame(threshold_rows).to_csv(
    os.path.join(OUT, "threshold_analysis.csv"), index=False
)

# ---------------- per-dataset evaluation ----------------
per = []
for name in list(models):
    for ds, g in predout.groupby("dataset"):
        yy = g.true_label.values
        pp = g[name + "_prob"].values
        if len(np.unique(yy)) < 2:
            continue
        z = metrics(yy, pp)
        per.append({"Model": name, "Dataset": ds, **z})
pd.DataFrame(per).to_csv(os.path.join(OUT, "per_dataset_test_metrics.csv"), index=False)

# ---------------- PyTorch MLP ----------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # leakage-safe train/validation split
    a, b = train_test_split(
        np.arange(len(Xtr)), test_size=0.20, random_state=SEED, stratify=ytr
    )
    sel = SelectKBest(f_classif, k=min(K, Xtr.shape[1]))
    A = sel.fit_transform(Xtr[a], ytr[a])
    B = sel.transform(Xtr[b])
    T = sel.transform(Xte)
    scaler = StandardScaler()
    A = scaler.fit_transform(A).astype("float32")
    B = scaler.transform(B).astype("float32")
    T = scaler.transform(T).astype("float32")

    class MLP(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n, 256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MLP(A.shape[1]).to(device)
    pos = (ytr[a] == 0).sum() / max(1, (ytr[a] == 1).sum())
    lossfn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos], device=device, dtype=torch.float32)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(A), torch.tensor(ytr[a]).float()),
        batch_size=64,
        shuffle=True,
    )

    best = None
    bestv = np.inf
    wait = 0
    for epoch in range(200):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossfn(net(xb), yb)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossfn(
                net(torch.tensor(B).to(device)), torch.tensor(ytr[b]).float().to(device)
            ).item()
        if vl < bestv:
            bestv = vl
            wait = 0
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break

    net.load_state_dict(best)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.tensor(T).to(device))).cpu().numpy()
    z = metrics(yte, p)
    row = {"Model": "PyTorch_MLP"}
    for k, v in z.items():
        row["Test_" + k] = v
    row.update({f"CV_{k}_mean": np.nan for k in scoring})
    row.update({f"CV_{k}_SD": np.nan for k in scoring})
    row.update({f"CV_{k}_min": np.nan for k in scoring})
    row.update({f"CV_{k}_max": np.nan for k in scoring})
    rows.append(row)
    predout["PyTorch_MLP_prob"] = p
    predout["PyTorch_MLP_pred"] = (p >= 0.5).astype(int)

    torch.save(
        {
            "state_dict": net.state_dict(),
            "selected_genes": np.array(genes)[sel.get_support()].tolist(),
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        },
        os.path.join(OUT, "PyTorch_MLP.pt"),
    )
    print("\n" + "-" * 60)
    print(f"PyTorch MLP [{device}]")
    for k in [
        "Accuracy",
        "Balanced_accuracy",
        "Precision_PPV",
        "Sensitivity_Recall_TPR",
        "Specificity_TNR",
        "F1",
        "F2",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
        "Log_loss",
        "Brier_score",
    ]:
        print(f"TEST {k:<22} {z[k]:.4f}")
except Exception as e:
    print("\nPyTorch MLP skipped:", type(e).__name__, e)

# ---------------- save everything ----------------
res = pd.DataFrame(rows).sort_values("Test_ROC_AUC", ascending=False)
res.to_csv(os.path.join(OUT, "model_results_comprehensive.csv"), index=False)
predout.to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)

# compact ranking
rank = res[
    [
        "Model",
        "Test_ROC_AUC",
        "Test_PR_AUC",
        "Test_Accuracy",
        "Test_Balanced_accuracy",
        "Test_Precision_PPV",
        "Test_Sensitivity_Recall_TPR",
        "Test_Specificity_TNR",
        "Test_F1",
        "Test_MCC",
    ]
].copy()
rank.to_csv(os.path.join(OUT, "model_ranking.csv"), index=False)

print("\n" + "=" * 72)
print("FINAL MODEL RANKING")
print("=" * 72)
print(rank.to_string(index=False))

print("\nOUTPUT:", OUT)
print("Comprehensive:", os.path.join(OUT, "model_results_comprehensive.csv"))
print("Ranking      :", os.path.join(OUT, "model_ranking.csv"))
print("Predictions  :", os.path.join(OUT, "test_predictions.csv"))
print("Thresholds   :", os.path.join(OUT, "threshold_analysis.csv"))
print("Per-dataset  :", os.path.join(OUT, "per_dataset_test_metrics.csv"))
print("=" * 72)
print("STEP 17 COMPLETE")
print("=" * 72)
