import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
    roc_curve, auc,
)
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BG, SURFACE, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED         = "#e6edf3", "#8b949e"
ACCENT, GREEN       = "#58a6ff", "#3fb950"
ORANGE, RED, PURPLE = "#d29922", "#f85149", "#bc8cff"

CLASS_COLORS = {
    "Enzyme":               ACCENT,
    "Transporter":          GREEN,
    "Transcription_Factor": PURPLE,
    "Structural_Protein":   ORANGE,
}

plt.rcParams.update({
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BORDER, "figure.facecolor": BG,
    "axes.facecolor": SURFACE, "font.family": "monospace",
})


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"✗ Feature matrix not found: {DATA_PATH}")
        print("  Run src/build_features.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} proteins × {df.shape[1]-2} features")
    print(df["label"].value_counts().to_string())

    feat_cols = [c for c in df.columns if c not in ("label", "entry_id")]
    X = df[feat_cols].fillna(0).values
    y = df["label"].values
    return X, y, feat_cols, df


def train_and_evaluate(X, y, feat_cols):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    models = {
        "Random Forest": Pipeline([
            ("rf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2,
                n_jobs=-1, random_state=42,
            ))
        ]),
        "SVM": Pipeline([
            ("sc", StandardScaler()),
            ("svc", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42,
            ))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n🔬 Training {name}...")

        fit_X_tr = X_tr if name == "Random Forest" else X_tr_sc
        fit_X_te = X_te if name == "Random Forest" else X_te_sc

        model.fit(fit_X_tr, y_tr)
        y_pred  = model.predict(fit_X_te)
        y_proba = model.predict_proba(fit_X_te)

        prec  = precision_score(y_te, y_pred, average="macro", zero_division=0)
        rec   = recall_score(y_te, y_pred, average="macro")
        f1    = f1_score(y_te, y_pred, average="macro")
        rocauc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")

        cv_scores = cross_val_score(
            model, fit_X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1
        )

        print(f"   Precision {prec:.4f}  Recall {rec:.4f}  "
              f"F1 {f1:.4f}  ROC-AUC {rocauc:.4f}")
        print(f"   CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        results[name] = dict(
            model=model, y_pred=y_pred, y_proba=y_proba,
            precision=prec, recall=rec, f1=f1, roc_auc=rocauc,
            cv_scores=cv_scores,
            cm=confusion_matrix(y_te, y_pred),
            report=classification_report(y_te, y_pred, target_names=le.classes_),
        )

    print("\n📋 Classification Reports")
    for name, r in results.items():
        print(f"\n── {name} ──\n{r['report']}")

    return results, X_tr, X_te, y_tr, y_te, le, scaler


def plot_dashboard(results, X_tr, X_te, y_tr, y_te, le, scaler, feat_cols):

    fig = plt.figure(figsize=(28, 22), facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 5, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.05, right=0.97,
        top=0.93,  bottom=0.04,
    )

    def ax_style(ax, title, sub=None):
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_title(title, color=TEXT, fontsize=10.5, fontweight="bold",
                     pad=8, loc="left")
        if sub:
            ax.set_title(sub, color=MUTED, fontsize=8, pad=8, loc="right")

    fig.text(0.5, 0.963,
             "PROTEIN FUNCTION CLASSIFIER  ·  UniProt Swiss-Prot",
             ha="center", color=TEXT, fontsize=20, fontweight="bold",
             fontfamily="monospace")
    fig.text(0.5, 0.947,
             "Enzyme · Transporter · Transcription Factor · Structural Protein  "
             "—  Random Forest vs SVM  —  45 Sequence Features",
             ha="center", color=MUTED, fontsize=9, fontfamily="monospace")

    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(BG)
    for sp in ax0.spines.values(): sp.set_visible(False)
    ax0.set_xticks([]); ax0.set_yticks([])

    cards = [
        ("RF  Precision",  results["Random Forest"]["precision"], ACCENT),
        ("RF  Recall",     results["Random Forest"]["recall"],    GREEN),
        ("RF  ROC-AUC",    results["Random Forest"]["roc_auc"],   PURPLE),
        ("SVM ROC-AUC",    results["SVM"]["roc_auc"],            ORANGE),
        ("SVM F1 (macro)", results["SVM"]["f1"],                 RED),
    ]
    for i, (lbl, val, col) in enumerate(cards):
        x = 0.04 + i * 0.196
        ax0.add_patch(plt.Rectangle((x, 0.08), 0.17, 0.75,
            transform=ax0.transAxes, facecolor=SURFACE,
            edgecolor=col, linewidth=1.5, zorder=2))
        ax0.text(x + 0.085, 0.60, f"{val:.4f}",
            transform=ax0.transAxes, ha="center",
            color=col, fontsize=20, fontweight="bold",
            fontfamily="monospace", zorder=3)
        ax0.text(x + 0.085, 0.22, lbl,
            transform=ax0.transAxes, ha="center",
            color=MUTED, fontsize=8, fontfamily="monospace", zorder=3)

    cmap = LinearSegmentedColormap.from_list("cb", [SURFACE, ACCENT])
    short = ["Enzyme", "Trans.", "TF", "Struct."]

    for ci, mname in enumerate(["Random Forest", "SVM"]):
        ax = fig.add_subplot(gs[1, ci*2 : ci*2+2])
        cm_n = results[mname]["cm"].astype(float)
        cm_n = cm_n / cm_n.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(short, fontsize=8, color=MUTED)
        ax.set_yticklabels(short, fontsize=8, color=MUTED, rotation=45)
        ax.set_xlabel("Predicted", color=MUTED, fontsize=8)
        ax.set_ylabel("True",      color=MUTED, fontsize=8)
        ax_style(ax, f"Confusion Matrix  [{mname}]",
                 f"F1={results[mname]['f1']:.3f}")
        for i in range(4):
            for j in range(4):
                v = cm_n[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=TEXT if v > 0.5 else MUTED,
                        fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.label.set_color(MUTED)

    ax_cmp = fig.add_subplot(gs[1, 4])
    ax_style(ax_cmp, "Model Comparison")
    metrics = ["precision","recall","f1","roc_auc"]
    xlbl    = ["Prec.","Recall","F1","AUC"]
    x = np.arange(4); w = 0.35
    rf_v  = [results["Random Forest"][m] for m in metrics]
    svm_v = [results["SVM"][m]           for m in metrics]
    b1 = ax_cmp.bar(x-w/2, rf_v,  w, color=ACCENT,  alpha=0.85, label="RF")
    b2 = ax_cmp.bar(x+w/2, svm_v, w, color=ORANGE, alpha=0.85, label="SVM")
    ax_cmp.set_xticks(x); ax_cmp.set_xticklabels(xlbl, fontsize=8, color=MUTED)
    ax_cmp.set_ylim(0, 1.15); ax_cmp.set_ylabel("Score", color=MUTED, fontsize=8)
    for b in list(b1)+list(b2):
        h = b.get_height()
        ax_cmp.text(b.get_x()+b.get_width()/2, h+0.01,
                    f"{h:.2f}", ha="center", color=TEXT, fontsize=7)
    ax_cmp.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
    ax_cmp.grid(axis="y", color=BORDER, lw=0.5, alpha=0.5)

    ax_tsne = fig.add_subplot(gs[2, :3])
    X_all   = np.vstack([X_tr, X_te])
    y_all   = np.hstack([y_tr, y_te])
    X_all_sc = scaler.transform(X_all)
    idx = np.random.RandomState(42).choice(len(X_all_sc),
                                           min(800, len(X_all_sc)), replace=False)
    print("⚙  Computing t-SNE…")
    emb = TSNE(n_components=2, perplexity=35, max_iter=1000,
               random_state=42, init="pca").fit_transform(X_all_sc[idx])

    for ci, cls in enumerate(le.classes_):
        mask = y_all[idx] == ci
        ax_tsne.scatter(emb[mask,0], emb[mask,1],
                        c=CLASS_COLORS.get(cls, ACCENT),
                        alpha=0.65, s=16, label=cls, edgecolors="none")
    ax_style(ax_tsne, "t-SNE Projection", f"n={len(idx)} sampled")
    ax_tsne.set_xlabel("t-SNE 1", color=MUTED, fontsize=8)
    ax_tsne.set_ylabel("t-SNE 2", color=MUTED, fontsize=8)
    ax_tsne.legend(fontsize=8, facecolor=SURFACE, edgecolor=BORDER,
                   labelcolor=TEXT, markerscale=1.5)
    ax_tsne.grid(color=BORDER, lw=0.4, alpha=0.4)

    ax_fi = fig.add_subplot(gs[2, 3:])
    ax_style(ax_fi, "Top Feature Importances", "(Random Forest)")
    rf_imp = results["Random Forest"]["model"].named_steps["rf"].feature_importances_
    top_idx = np.argsort(rf_imp)[-15:]
    top_vals = rf_imp[top_idx]
    top_names = [feat_cols[i].replace("aa_","").replace("dp_","dp:") for i in top_idx]
    ax_fi.barh(range(15), top_vals, color=ACCENT, alpha=0.85)
    ax_fi.set_yticks(range(15))
    ax_fi.set_yticklabels(top_names, fontsize=8, color=MUTED)
    ax_fi.set_xlabel("Importance", color=MUTED, fontsize=8)
    ax_fi.grid(axis="x", color=BORDER, lw=0.4, alpha=0.5)
    for i, v in enumerate(top_vals):
        ax_fi.text(v+0.0002, i, f"{v:.4f}", va="center", color=MUTED, fontsize=6.5)

    curve_colors = [ACCENT, GREEN, PURPLE, RED]
    y_te_bin = label_binarize(y_te, classes=range(len(le.classes_)))

    for ci, (mname, mcol) in enumerate(zip(["Random Forest","SVM"],[ACCENT,ORANGE])):
        ax_roc = fig.add_subplot(gs[3, ci*2 : ci*2+2])
        ax_style(ax_roc, f"ROC Curves  [{mname}]",
                 f"macro AUC={results[mname]['roc_auc']:.3f}")
        y_proba = results[mname]["y_proba"]
        for i, (cls, col) in enumerate(zip(le.classes_, curve_colors)):
            fpr, tpr, _ = roc_curve(y_te_bin[:,i], y_proba[:,i])
            a = auc(fpr, tpr)
            short_cls = cls.replace("Transcription_","TF_").replace("_Protein","")
            ax_roc.plot(fpr, tpr, color=col, lw=1.8, alpha=0.9,
                        label=f"{short_cls} ({a:.2f})")
        ax_roc.plot([0,1],[0,1],"--", color=BORDER, lw=1)
        ax_roc.set_xlabel("FPR", color=MUTED, fontsize=8)
        ax_roc.set_ylabel("TPR", color=MUTED, fontsize=8)
        ax_roc.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER,
                      labelcolor=TEXT, loc="lower right")
        ax_roc.grid(color=BORDER, lw=0.4, alpha=0.4)
        ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.05)

    ax_cv = fig.add_subplot(gs[3, 4])
    ax_style(ax_cv, "5-Fold CV  F1 Score")
    for i, (mname, col) in enumerate(zip(["Random Forest","SVM"],[ACCENT,ORANGE])):
        sc = results[mname]["cv_scores"]
        ax_cv.scatter(np.random.normal(i,0.06,len(sc)), sc,
                      color=col, alpha=0.8, s=40, zorder=3)
        ax_cv.boxplot(sc, positions=[i], widths=0.35, patch_artist=True,
                      boxprops=dict(facecolor=col, alpha=0.25, color=col),
                      medianprops=dict(color=TEXT, lw=2),
                      whiskerprops=dict(color=MUTED),
                      capprops=dict(color=MUTED),
                      flierprops=dict(color=MUTED))
    ax_cv.set_xticks([0,1]); ax_cv.set_xticklabels(["RF","SVM"],
                                                     fontsize=9, color=MUTED)
    ax_cv.set_ylabel("F1", color=MUTED, fontsize=8)
    ax_cv.set_ylim(0.3, 1.05)
    ax_cv.grid(axis="y", color=BORDER, lw=0.4, alpha=0.5)

    out_path = os.path.join(OUT_DIR, "results_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n✅ Dashboard → {out_path}")


def save_summary(results):
    lines = [
        "PROTEIN FUNCTION CLASSIFIER — RESULTS SUMMARY",
        "UniProt Swiss-Prot · Real sequences",
        "=" * 60, "",
    ]
    for name, r in results.items():
        lines += [
            f"Model: {name}",
            f"  Precision (macro): {r['precision']:.4f}",
            f"  Recall    (macro): {r['recall']:.4f}",
            f"  F1        (macro): {r['f1']:.4f}",
            f"  ROC-AUC   (macro): {r['roc_auc']:.4f}",
            f"  CV F1 (5-fold):    {r['cv_scores'].mean():.4f} "
            f"± {r['cv_scores'].std():.4f}",
            "", r["report"], "-" * 60,
        ]
    path = os.path.join(OUT_DIR, "results_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"✓ Summary → {path}")


if __name__ == "__main__":
    X, y, feat_cols, df = load_data()
    results, X_tr, X_te, y_tr, y_te, le, scaler = train_and_evaluate(X, y, feat_cols)
    plot_dashboard(results, X_tr, X_te, y_tr, y_te, le, scaler, feat_cols)
    save_summary(results)
