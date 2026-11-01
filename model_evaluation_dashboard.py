
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

METRICS_FILE = BASE_DIR / "exports" / "model_evaluation_metrics.csv"
PREDICTIONS_FILE = BASE_DIR / "exports" / "model_evaluation_predictions.csv"

OUTPUT_DIR = BASE_DIR / "exports" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "flowforge_model_evaluation_dashboard.png"


# --------------------------------------------------
# 2. LOAD EVALUATION DATA
# --------------------------------------------------

if not METRICS_FILE.exists():
    raise FileNotFoundError(
        f"Metrics file not found: {METRICS_FILE}"
    )

if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        f"Predictions file not found: {PREDICTIONS_FILE}"
    )

metrics = pd.read_csv(METRICS_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)

required_metrics = [
    "Model",
    "Precision",
    "Recall",
    "F1_Score",
    "Execution_Time_Seconds",
    "Anomalies_Detected"
]

required_predictions = [
    "True_Anomaly",
    "Ensemble_Prediction"
]

missing_metrics = [
    col for col in required_metrics
    if col not in metrics.columns
]

missing_predictions = [
    col for col in required_predictions
    if col not in predictions.columns
]

if missing_metrics:
    raise ValueError(
        f"Missing columns in metrics CSV: {missing_metrics}"
    )

if missing_predictions:
    raise ValueError(
        f"Missing columns in predictions CSV: {missing_predictions}"
    )


# --------------------------------------------------
# 3. CALCULATE KPI VALUES
# --------------------------------------------------

best_f1_row = metrics.loc[metrics["F1_Score"].idxmax()]
best_precision_row = metrics.loc[metrics["Precision"].idxmax()]
best_recall_row = metrics.loc[metrics["Recall"].idxmax()]

best_f1 = best_f1_row["F1_Score"]
best_precision = best_precision_row["Precision"]
best_recall = best_recall_row["Recall"]

best_f1_model = best_f1_row["Model"]
best_precision_model = best_precision_row["Model"]
best_recall_model = best_recall_row["Model"]

injected_anomalies = int(predictions["True_Anomaly"].sum())

total_test_records = len(predictions)
total_dataset_records = 541909
training_records = total_dataset_records - total_test_records
training_sample = 10000


# --------------------------------------------------
# 4. CALCULATE ENSEMBLE CONFUSION MATRIX
# --------------------------------------------------

y_true = predictions["True_Anomaly"].astype(int)
y_pred = predictions["Ensemble_Prediction"].astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

confusion_values = np.array([
    [tn, fp],
    [fn, tp]
])


# --------------------------------------------------
# 5. DASHBOARD THEME
# --------------------------------------------------

NAVY = "#101B2D"
PANEL = "#19283D"
TEAL = "#21C7A8"
WHITE = "#F4F7FB"
GREY = "#AAB7C8"
RED = "#F26B6B"
YELLOW = "#F4C76B"

plt.rcParams["font.family"] = "DejaVu Sans"

fig = plt.figure(
    figsize=(18, 12),
    facecolor=NAVY
)

gs = GridSpec(
    3,
    4,
    figure=fig,
    height_ratios=[0.75, 2.1, 1.8],
    hspace=0.38,
    wspace=0.28
)


# --------------------------------------------------
# 6. DASHBOARD TITLE
# --------------------------------------------------

fig.text(
    0.05,
    0.965,
    "FLOWFORGE",
    fontsize=15,
    fontweight="bold",
    color=TEAL
)

fig.text(
    0.05,
    0.925,
    "MODEL EVALUATION DASHBOARD",
    fontsize=23,
    fontweight="bold",
    color=WHITE
)

fig.text(
    0.95,
    0.945,
    "RETAIL ETL | ANOMALY DETECTION",
    fontsize=10,
    color=GREY,
    ha="right"
)


# --------------------------------------------------
# 7. KPI CARDS
# --------------------------------------------------

kpis = [
    (
        "BEST F1 SCORE",
        f"{best_f1:.4f}",
        best_f1_model,
        TEAL
    ),
    (
        "BEST PRECISION",
        f"{best_precision:.4f}",
        best_precision_model,
        "#64A8FF"
    ),
    (
        "BEST RECALL",
        f"{best_recall:.4f}",
        best_recall_model,
        YELLOW
    ),
    (
        "INJECTED ANOMALIES",
        f"{injected_anomalies:,}",
        "Known test anomalies",
        RED
    )
]

for i, (label, value, description, accent) in enumerate(kpis):

    ax = fig.add_subplot(gs[0, i])

    ax.set_facecolor(PANEL)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        0.07,
        0.78,
        label,
        transform=ax.transAxes,
        fontsize=10,
        color=GREY,
        fontweight="bold"
    )

    ax.text(
        0.07,
        0.40,
        value,
        transform=ax.transAxes,
        fontsize=25,
        color=accent,
        fontweight="bold"
    )

    ax.text(
        0.07,
        0.13,
        description,
        transform=ax.transAxes,
        fontsize=8,
        color=WHITE
    )


# --------------------------------------------------
# 8. MODEL F1 SCORE COMPARISON
# --------------------------------------------------

ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor(PANEL)

model_names = metrics["Model"].tolist()
f1_scores = metrics["F1_Score"].astype(float).tolist()

bars = ax1.barh(
    model_names,
    f1_scores,
    color=[
        TEAL if name == best_f1_model else "#54718F"
        for name in model_names
    ],
    height=0.55
)

ax1.set_title(
    "F1 SCORE BY MODEL",
    color=WHITE,
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=15
)

ax1.set_xlim(0, 1)
ax1.set_xlabel("F1 Score", color=GREY)

ax1.tick_params(colors=WHITE, labelsize=9)

ax1.grid(
    axis="x",
    color=GREY,
    alpha=0.15
)

ax1.set_axisbelow(True)

for bar, score in zip(bars, f1_scores):
    ax1.text(
        score + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{score:.4f}",
        color=WHITE,
        va="center",
        fontsize=9
    )

for spine in ax1.spines.values():
    spine.set_visible(False)


# --------------------------------------------------
# 9. PRECISION VS RECALL
# --------------------------------------------------

ax2 = fig.add_subplot(gs[1, 2:])
ax2.set_facecolor(PANEL)

x = np.arange(len(metrics))
width = 0.35

precision_values = metrics["Precision"].astype(float).values
recall_values = metrics["Recall"].astype(float).values

ax2.bar(
    x - width / 2,
    precision_values,
    width,
    label="Precision",
    color="#64A8FF"
)

ax2.bar(
    x + width / 2,
    recall_values,
    width,
    label="Recall",
    color=TEAL
)

ax2.set_title(
    "PRECISION VS RECALL",
    color=WHITE,
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=15
)

ax2.set_xticks(x)
ax2.set_xticklabels(
    model_names,
    rotation=15,
    ha="right",
    fontsize=8,
    color=WHITE
)

ax2.set_ylim(0, 1)
ax2.set_ylabel("Score", color=GREY)

ax2.tick_params(axis="y", colors=WHITE)

ax2.grid(
    axis="y",
    color=GREY,
    alpha=0.15
)

ax2.set_axisbelow(True)
ax2.legend(
    facecolor=PANEL,
    edgecolor=PANEL,
    labelcolor=WHITE,
    fontsize=9
)

for spine in ax2.spines.values():
    spine.set_visible(False)


# --------------------------------------------------
# 10. ENSEMBLE CONFUSION MATRIX
# --------------------------------------------------

ax3 = fig.add_subplot(gs[2, :2])
ax3.set_facecolor(PANEL)

im = ax3.imshow(
    confusion_values,
    cmap="GnBu"
)

ax3.set_title(
    "ENSEMBLE CONFUSION MATRIX",
    color=WHITE,
    fontsize=13,
    fontweight="bold",
    loc="left",
    pad=15
)

ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])

ax3.set_xticklabels(
    ["Predicted Normal", "Predicted Anomaly"],
    color=WHITE,
    fontsize=9
)

ax3.set_yticklabels(
    ["Actual Normal", "Actual Anomaly"],
    color=WHITE,
    fontsize=9
)

for i in range(2):
    for j in range(2):
        ax3.text(
            j,
            i,
            f"{confusion_values[i, j]:,}",
            ha="center",
            va="center",
            color="black",
            fontsize=13,
            fontweight="bold"
        )

for spine in ax3.spines.values():
    spine.set_visible(False)


# --------------------------------------------------
# 11. EXECUTION TIME COMPARISON
# --------------------------------------------------

ax4 = fig.add_subplot(gs[2, 2])
ax4.set_facecolor(PANEL)

individual_models = metrics[
    metrics["Model"] != "Majority Voting Ensemble"
]

runtime_names = individual_models["Model"].tolist()
runtime_values = individual_models[
    "Execution_Time_Seconds"
].astype(float).tolist()

runtime_bars = ax4.barh(
    runtime_names,
    runtime_values,
    color="#54718F",
    height=0.55
)

ax4.set_title(
    "MODEL EXECUTION TIME",
    color=WHITE,
    fontsize=12,
    fontweight="bold",
    loc="left",
    pad=15
)

ax4.set_xlabel("Seconds", color=GREY)
ax4.tick_params(colors=WHITE, labelsize=8)

ax4.grid(
    axis="x",
    color=GREY,
    alpha=0.15
)

ax4.set_axisbelow(True)

for bar, value in zip(runtime_bars, runtime_values):
    ax4.text(
        value + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}s",
        color=WHITE,
        va="center",
        fontsize=8
    )

for spine in ax4.spines.values():
    spine.set_visible(False)


# --------------------------------------------------
# 12. DATASET SUMMARY AND KEY FINDINGS
# --------------------------------------------------

ax5 = fig.add_subplot(gs[2, 3])
ax5.set_facecolor(PANEL)

ax5.set_xticks([])
ax5.set_yticks([])

for spine in ax5.spines.values():
    spine.set_visible(False)

ax5.text(
    0.07,
    0.91,
    "EVALUATION SUMMARY",
    transform=ax5.transAxes,
    color=WHITE,
    fontsize=11,
    fontweight="bold"
)

summary_items = [
    ("Total records", f"{total_dataset_records:,}"),
    ("Training records", f"{training_records:,}"),
    ("Testing records", f"{total_test_records:,}"),
    ("Training sample", f"{training_sample:,}"),
    ("Ensemble TP", f"{tp:,}"),
    ("Ensemble FP", f"{fp:,}")
]

y_position = 0.76

for label, value in summary_items:

    ax5.text(
        0.07,
        y_position,
        label,
        transform=ax5.transAxes,
        color=GREY,
        fontsize=9
    )

    ax5.text(
        0.93,
        y_position,
        value,
        transform=ax5.transAxes,
        color=TEAL,
        fontsize=9,
        fontweight="bold",
        ha="right"
    )

    y_position -= 0.12


# --------------------------------------------------
# 13. FOOTNOTE
# --------------------------------------------------

fig.text(
    0.05,
    0.025,
    "Note: Evaluation uses synthetically injected test anomalies. "
    "Results represent a controlled benchmark, not verified real-world fraud labels.",
    color=GREY,
    fontsize=9
)

fig.text(
    0.95,
    0.025,
    "FLOWFORGE | MODEL PERFORMANCE",
    color=TEAL,
    fontsize=9,
    ha="right"
)


# --------------------------------------------------
# 14. SAVE DASHBOARD
# --------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=220,
    facecolor=NAVY,
    bbox_inches="tight"
)

plt.close(fig)

print("\nDashboard created successfully!")
print(f"Saved to: {OUTPUT_FILE}")
