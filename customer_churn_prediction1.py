import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


# ------------------ LOAD DATASET ------------------

df = pd.read_csv("E:/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ------------------ DATA PREPROCESSING ------------------

# Remove duplicate records
df = df.drop_duplicates()

# Convert TotalCharges into numeric format
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges values with 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove customerID because it is only an identification column
df = df.drop("customerID", axis=1)

# Convert target column: No = 0 and Yes = 1
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nDataset Shape After Preprocessing:")
print(df.shape)


# ------------------ PREPARE DATA FOR MACHINE LEARNING ------------------

# Separate input features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nInput Features:")
print(X.head())

print("\nInput Data Shape:", X.shape)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Scale features for Logistic Regression, KNN and SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print("\n========== LOGISTIC REGRESSION ==========")

lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, zero_division=0)
lr_recall = recall_score(y_test, lr_pred, zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, zero_division=0)

print("Accuracy:", round(lr_accuracy, 4))
print("Precision:", round(lr_precision, 4))
print("Recall:", round(lr_recall, 4))
print("F1 Score:", round(lr_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, zero_division=0))

lr_cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# MODEL 2 — DECISION TREE
# ============================================================

print("\n========== DECISION TREE ==========")

dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    class_weight="balanced",
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)

print("Accuracy:", round(dt_accuracy, 4))
print("Precision:", round(dt_precision, 4))
print("Recall:", round(dt_recall, 4))
print("F1 Score:", round(dt_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, dt_pred, zero_division=0))

dt_cm = confusion_matrix(y_test, dt_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# MODEL 3 — RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST ==========")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, zero_division=0)
rf_recall = recall_score(y_test, rf_pred, zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, zero_division=0)

print("Accuracy:", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1 Score:", round(rf_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Oranges")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# MODEL 4 — K-NEAREST NEIGHBORS (KNN)
# ============================================================

print("\n========== K-NEAREST NEIGHBORS ==========")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)
knn_prob = knn_model.predict_proba(X_test_scaled)[:, 1]

knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, zero_division=0)
knn_recall = recall_score(y_test, knn_pred, zero_division=0)
knn_f1 = f1_score(y_test, knn_pred, zero_division=0)

print("Accuracy:", round(knn_accuracy, 4))
print("Precision:", round(knn_precision, 4))
print("Recall:", round(knn_recall, 4))
print("F1 Score:", round(knn_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, knn_pred, zero_division=0))

knn_cm = confusion_matrix(y_test, knn_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Purples")
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# MODEL 5 — SUPPORT VECTOR MACHINE (SVM)
# ============================================================

print("\n========== SUPPORT VECTOR MACHINE ==========")

svm_model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, zero_division=0)
svm_recall = recall_score(y_test, svm_pred, zero_division=0)
svm_f1 = f1_score(y_test, svm_pred, zero_division=0)

print("Accuracy:", round(svm_accuracy, 4))
print("Precision:", round(svm_precision, 4))
print("Recall:", round(svm_recall, 4))
print("F1 Score:", round(svm_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test, svm_pred, zero_division=0))

svm_cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(svm_cm, annot=True, fmt="d", cmap="Reds")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ------------------ COMPARISON OF FIVE MODELS ------------------

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM"
    ],
    "Accuracy": [
        lr_accuracy, dt_accuracy, rf_accuracy, knn_accuracy, svm_accuracy
    ],
    "Precision": [
        lr_precision, dt_precision, rf_precision, knn_precision, svm_precision
    ],
    "Recall": [
        lr_recall, dt_recall, rf_recall, knn_recall, svm_recall
    ],
    "F1 Score": [
        lr_f1, dt_f1, rf_f1, knn_f1, svm_f1
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))


# Accuracy comparison graph
plt.figure(figsize=(9, 5))
sns.barplot(x="Model", y="Accuracy", data=results)
plt.title("Accuracy Comparison of Five Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# F1 score comparison graph
plt.figure(figsize=(9, 5))
sns.barplot(x="Model", y="F1 Score", data=results)
plt.title("F1 Score Comparison of Five Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ------------------ ROC CURVE AND AUC COMPARISON ------------------

lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_prob)
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dt_prob)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_prob)
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_prob)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_prob)

lr_auc = auc(lr_fpr, lr_tpr)
dt_auc = auc(dt_fpr, dt_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
knn_auc = auc(knn_fpr, knn_tpr)
svm_auc = auc(svm_fpr, svm_tpr)

print("\n========== AUC SCORES ==========")
print("Logistic Regression:", round(lr_auc, 4))
print("Decision Tree:", round(dt_auc, 4))
print("Random Forest:", round(rf_auc, 4))
print("KNN:", round(knn_auc, 4))
print("SVM:", round(svm_auc, 4))

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label="Logistic Regression")
plt.plot(dt_fpr, dt_tpr, label="Decision Tree")
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(knn_fpr, knn_tpr, label="KNN")
plt.plot(svm_fpr, svm_tpr, label="SVM")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# ------------------ BEST MODEL ------------------

best_model_index = results["F1 Score"].idxmax()
best_model_name = results.loc[best_model_index, "Model"]

print("\nBest Model Based on F1 Score:", best_model_name)
print("Accuracy:", round(results.loc[best_model_index, "Accuracy"], 4))
print("Precision:", round(results.loc[best_model_index, "Precision"], 4))
print("Recall:", round(results.loc[best_model_index, "Recall"], 4))
print("F1 Score:", round(results.loc[best_model_index, "F1 Score"], 4))


