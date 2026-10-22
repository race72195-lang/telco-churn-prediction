from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, log_loss,
)
 
# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)
 
# Final model: logistic regression. Reasons: results are as good as the complex
# models, it is simple to explain, and its weights show WHY customers leave.
FINAL_MODEL = "Logistic Regression"
RANDOM_STATE = 42
 
sns.set_theme(style="whitegrid", context="notebook")
 
 
def save_fig(name):
    """Save the current figure into the images folder and close it."""
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / name, dpi=150)
    plt.close()
    print(f"  saved images/{name}")
 
 
# ----------------------------------------------------------------------------
# 1. LOAD AND CLEAN DATA
# ----------------------------------------------------------------------------
if not CSV_PATH.exists():
    raise SystemExit(
        f"CSV file not found: {CSV_PATH}\n"
        "Download 'Telco Customer Churn' from Kaggle and put the CSV next to this script."
    )
 
df = pd.read_csv(CSV_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)  # 1 = left, 0 = stayed
 
overall = df["Churn"].mean() * 100
print(f"Customers: {len(df)} | overall churn rate: {overall:.1f}%")
 
# ----------------------------------------------------------------------------
# 2. EXPLORATORY ANALYSIS: churn rate by group
# ----------------------------------------------------------------------------
print("\nDrawing charts...")
df["TenureGroup"] = pd.cut(
    df["tenure"], [-1, 12, 24, 48, 72],
    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
 
 
def churn_bar(ax, column, title, sort=True):
    rate = df.groupby(column, observed=True)["Churn"].mean() * 100
    if sort:
        rate = rate.sort_values(ascending=False)
    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, color="#4C72B0")
    ax.bar_label(ax.containers[0], fmt="%.0f%%", padding=3)
    ax.axhline(overall, ls="--", color="gray", lw=1)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Churn rate, %")
    ax.tick_params(axis="x", rotation=15)
 
 
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
churn_bar(axes[0, 0], "Contract", "Churn by contract type")
churn_bar(axes[0, 1], "TenureGroup", "Churn by customer tenure", sort=False)
churn_bar(axes[1, 0], "InternetService", "Churn by internet service")
churn_bar(axes[1, 1], "PaymentMethod", "Churn by payment method")
fig.suptitle(f"Where do customers leave? (dashed line = average {overall:.0f}%)",
             fontsize=15, fontweight="bold")
save_fig("01_churn_rates_by_group.png")
 
# ----------------------------------------------------------------------------
# 3. PREPARE FEATURES AND SPLIT: train / validation / test
# ----------------------------------------------------------------------------
X = pd.get_dummies(df.drop(columns=["Churn", "TenureGroup"]), drop_first=True)
y = df["Churn"]
 
# test = final exam (used once); validation = practice exam for tuning the threshold
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tmp)
print(f"\ntrain={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
 
 
# ----------------------------------------------------------------------------
# 4. METRICS AND THRESHOLD TUNING
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    """Accuracy, precision, recall, F1, FPR, FNR from the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FPR": fp / (fp + tn),
        "FNR": fn / (tp + fn),
    }
 
 
def best_threshold(y_true, proba):
    """Threshold with the best F1 on the validation set."""
    grid = np.arange(0.05, 0.96, 0.01)
    f1s = [metrics(y_true, (proba >= t).astype(int))["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])
 
 
# ----------------------------------------------------------------------------
# 5. TRAIN AND COMPARE MODELS
# ----------------------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "SGD (log loss)": make_pipeline(
        StandardScaler(), SGDClassifier(loss="log_loss", max_iter=1000, random_state=RANDOM_STATE)),
    "KNN (k=25)": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}
 
print("\nTraining models...")
rows, probas = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]     # P(churn) on validation
    p_test = model.predict_proba(X_test)[:, 1]   # P(churn) on test
    t = best_threshold(y_val, p_val)             # threshold chosen on VALIDATION
    probas[name] = (p_val, p_test, t)
 
    at_default = metrics(y_test, (p_test >= 0.5).astype(int))
    at_best = metrics(y_test, (p_test >= t).astype(int))
    rows.append({
        "model": name,
        "ROC-AUC": roc_auc_score(y_test, p_test),
        "log loss": log_loss(y_test, p_test),
        "best threshold": t,
        "recall @0.5": at_default["recall"],
        "precision @0.5": at_default["precision"],
        "F1 @0.5": at_default["f1"],
        "recall @best": at_best["recall"],
        "precision @best": at_best["precision"],
        "F1 @best": at_best["f1"],
        "FPR @best": at_best["FPR"],
        "FNR @best": at_best["FNR"],
    })
 
results = pd.DataFrame(rows).set_index("model").round(3)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n=== Model comparison on the TEST set (threshold tuned on validation) ===")
print(results.T)
results.to_csv(BASE_DIR / "model_comparison.csv")
 
# ----------------------------------------------------------------------------
# 6. FINAL MODEL: detailed evaluation on the test set
# ----------------------------------------------------------------------------
final = models[FINAL_MODEL]
p_val, p_test, threshold = probas[FINAL_MODEL]
y_pred = (p_test >= threshold).astype(int)
 
print(f"\n=== FINAL MODEL: {FINAL_MODEL} | threshold = {threshold:.2f} ===")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))
 
# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["Stayed", "Churned"],
    cmap="Blues", ax=ax, colorbar=False)
ax.set_title(f"{FINAL_MODEL}: predicted vs actual", fontweight="bold")
ax.grid(False)
save_fig("02_confusion_matrix.png")
 
# Precision / recall / F1 versus threshold (validation set)
grid = np.arange(0.05, 0.96, 0.01)
curve = pd.DataFrame([metrics(y_val, (p_val >= t).astype(int)) for t in grid], index=grid)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve.index, curve["precision"], label="precision")
ax.plot(curve.index, curve["recall"], label="recall")
ax.plot(curve.index, curve["f1"], label="F1", lw=2.5, color="black")
ax.axvline(0.5, ls=":", color="gray", label="default threshold 0.5")
ax.axvline(threshold, ls="--", color="red", label=f"chosen threshold = {threshold:.2f}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Lower threshold = more churners caught, more false alarms", fontweight="bold")
ax.legend()
save_fig("03_threshold_tradeoff.png")
 
# What drives churn? Weights w of the logistic regression (red = more churn)
lr = final.named_steps["logisticregression"]
w = pd.Series(lr.coef_[0], index=X.columns)
print(f"Bias b = {lr.intercept_[0]:.3f}")
top = w.reindex(w.abs().nlargest(10).index).sort_values()
colors = ["#C44E52" if v > 0 else "#55A868" for v in top.values]
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top.index, top.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Top 10 factors: red raises churn risk, green lowers it", fontweight="bold")
ax.set_xlabel("Weight w (features standardized)")
save_fig("04_churn_drivers.png")
 
# ----------------------------------------------------------------------------
# 7. SUMMARY FOR YOUR README / RESUME
# ----------------------------------------------------------------------------
final_metrics = metrics(y_test, y_pred)
auc = roc_auc_score(y_test, p_test)
print("\n" + "=" * 70)
print("SUMMARY (use your own numbers in the resume)")
print(f"  Customers: {len(df)} | final model: {FINAL_MODEL}")
print(f"  ROC-AUC: {auc:.2f} | recall: {final_metrics['recall']:.0%} | "
      f"precision: {final_metrics['precision']:.0%} | threshold: {threshold:.2f}")
print(f"  Resume line: Built a churn prediction model on {len(df):,} telecom customers; "
      f"logistic regression achieved ROC-AUC {auc:.2f} and "
      f"{final_metrics['recall']:.0%} recall on churned customers.")
print("=" * 70)
print("\nDone. Charts are in the 'images' folder; table saved as model_comparison.csv")