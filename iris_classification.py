
# =============================================================
# IRIS FLOWER CLASSIFICATION — CodeAlpha Data Science Internship
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# ── 1. LOAD ───────────────────────────────────────────────────
print("\n📂 STEP 1: Loading Dataset...")
df = pd.read_csv("Iris.csv")
print(df.head())

# ── 2. CLEAN ──────────────────────────────────────────────────
print("\n🧹 STEP 2: Cleaning Dataset...")

# Drop Id — not a feature
df.drop(columns=["Id"], inplace=True)

# Check for duplicates
dupes = df.duplicated().sum()
print(f"   Duplicate rows found: {dupes}")
df.drop_duplicates(inplace=True)

# Check for missing values
print(f"   Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)

# Check data types
print(f"\n   Data types:\n{df.dtypes}")
print(f"   Clean dataset shape: {df.shape}")
print(f"   Class distribution:\n{df['Species'].value_counts()}")

# ── 3. EXPLORE (EDA) ──────────────────────────────────────────
print("\n📊 STEP 3: Exploratory Data Analysis...")
print(df.describe())

# Plot 1: Distribution of each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = df.columns[:-1]  # all columns except Species
for ax, feat in zip(axes.flatten(), features):
    for species, group in df.groupby("Species"):
        ax.hist(group[feat], alpha=0.6, label=species, bins=15)
    ax.set_title(f"Distribution of {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend()
plt.suptitle("Feature Distributions by Species", fontsize=14)
plt.tight_layout()
plt.savefig("iris_distributions.png", dpi=150)
plt.close()
print("   Saved: iris_distributions.png")

# Plot 2: Boxplots — spot outliers
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flatten(), features):
    sns.boxplot(data=df, x="Species", y=feat, palette="Set2", ax=ax)
    ax.set_title(f"{feat} by Species")
    ax.set_xlabel("")
plt.suptitle("Boxplots — Feature vs Species", fontsize=14)
plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=150)
plt.close()
print("   Saved: iris_boxplots.png")

# Plot 3: Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_heatmap.png", dpi=150)
plt.close()
print("   Saved: iris_heatmap.png")

# Plot 4: Pairplot
pair = sns.pairplot(df, hue="Species", diag_kind="kde", height=2.5)
pair.fig.suptitle("Pairplot of All Features", y=1.02)
plt.savefig("iris_pairplot.png", bbox_inches="tight", dpi=150)
plt.close()
print("   Saved: iris_pairplot.png")

# ── 4. PREPROCESS ─────────────────────────────────────────────
print("\n⚙️  STEP 4: Preprocessing...")

X = df.drop(columns=["Species"])   # features
y = df["Species"]                  # target label

# Encode species names to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split: 80% train, 20% test — stratified to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Train samples: {X_train.shape[0]}")
print(f"   Test  samples: {X_test.shape[0]}")

# ── 5. TRAIN & EVALUATE ───────────────────────────────────────
print("\n🤖 STEP 5: Training Models...")

models = {
    "K-Nearest Neighbors"    : KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"          : DecisionTreeClassifier(random_state=42),
    "Random Forest"          : RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine" : SVC(kernel="rbf", random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)          # train
    y_pred = model.predict(X_test)       # predict
    acc = accuracy_score(y_test, y_pred) # evaluate
    results[name] = {"y_pred": y_pred, "accuracy": acc}
    print(f"\n   {name}")
    print(f"   Accuracy: {acc * 100:.2f}%")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    for line in report.split("\n"):
        print("   " + line)

# ── 6. COMPARE MODELS ─────────────────────────────────────────
print("\n📊 STEP 6: Comparing Models...")

names      = list(results.keys())
accuracies = [results[n]["accuracy"] * 100 for n in names]

plt.figure(figsize=(10, 5))
bars = plt.bar(names, accuracies,
               color=["#4CAF50","#2196F3","#FF9800","#9C27B0"],
               edgecolor="black", width=0.5)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5, f"{acc:.1f}%",
             ha="center", va="bottom", fontweight="bold")
plt.ylim(85, 105)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150)
plt.close()
print("   Saved: iris_model_comparison.png")

# ── 7. CONFUSION MATRICES ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm   = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy: {res['accuracy']*100:.1f}%")
    ax.set_xticklabels(le.classes_, rotation=15)
plt.suptitle("Confusion Matrices", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", dpi=150)
plt.close()
print("   Saved: iris_confusion_matrices.png")

# ── 8. FINAL SUMMARY ──────────────────────────────────────────
best_name = max(results, key=lambda n: results[n]["accuracy"])
best_acc  = results[best_name]["accuracy"]

print("\n" + "=" * 55)
print("  FINAL SUMMARY")
print("=" * 55)
for name in names:
    acc  = results[name]["accuracy"]
    star = " ⭐ BEST" if name == best_name else ""
    print(f"  {name:<30} {acc*100:.2f}%{star}")
print(f"\n🏆 Best Model : {best_name} — {best_acc*100:.2f}%")
print("\n✅ Project complete!")
