
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD DATASETS
# =========================================================

customers = pd.read_csv("customers.csv")
transactions = pd.read_csv("transactions.csv")
complaints = pd.read_csv("complaints.csv")
call_transcripts = pd.read_csv("call_transcripts.csv")
products = pd.read_csv("products.csv")
interactions = pd.read_csv("interactions.csv")


print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Customers:", customers.shape)
print("Transactions:", transactions.shape)
print("Complaints:", complaints.shape)
print("Call Transcripts:", call_transcripts.shape)
print("Products:", products.shape)
print("Interactions:", interactions.shape)


# =========================================================
# 2. CUSTOMER 360 BASE TABLE
# =========================================================

customer_360 = customers.copy()


# =========================================================
# 3. TRANSACTION FEATURES
# =========================================================

transaction_summary = (
    transactions
    .groupby("customer_id")
    .agg(
        transaction_count=("transaction_id", "count"),
        total_transaction_amount=("amount", "sum"),
        avg_transaction_amount=("amount", "mean")
    )
    .reset_index()
)

customer_360 = customer_360.merge(
    transaction_summary,
    on="customer_id",
    how="left"
)


# =========================================================
# 4. COMPLAINT FEATURES
# =========================================================

complaint_summary = (
    complaints
    .groupby("customer_id")
    .agg(
        complaint_count=("complaint_id", "count"),
        avg_resolution_days=("resolution_days", "mean"),
        high_severity_complaints=(
            "severity",
            lambda x: (x == "High").sum()
        ),
        open_complaints=(
            "status",
            lambda x: (x == "Open").sum()
        )
    )
    .reset_index()
)

customer_360 = customer_360.merge(
    complaint_summary,
    on="customer_id",
    how="left"
)


# =========================================================
# 5. CALL FEATURES
# =========================================================

call_summary = (
    call_transcripts
    .groupby("customer_id")
    .agg(
        call_count=("call_id", "count"),
        positive_calls=(
            "sentiment",
            lambda x: (x == "Positive").sum()
        ),
        negative_calls=(
            "sentiment",
            lambda x: (x == "Negative").sum()
        )
    )
    .reset_index()
)

customer_360 = customer_360.merge(
    call_summary,
    on="customer_id",
    how="left"
)


# =========================================================
# 6. PRODUCT FEATURES
# =========================================================

product_summary = (
    products
    .groupby("customer_id")
    .agg(
        total_products=("product_id", "count"),
        active_products=(
            "status",
            lambda x: (x == "Active").sum()
        ),
        closed_products=(
            "status",
            lambda x: (x == "Closed").sum()
        )
    )
    .reset_index()
)

customer_360 = customer_360.merge(
    product_summary,
    on="customer_id",
    how="left"
)


# =========================================================
# 7. INTERACTION FEATURES
# =========================================================

interaction_summary = (
    interactions
    .groupby("customer_id")
    .agg(
        total_interactions=("interaction_id", "count"),
        support_interactions=(
            "interaction_type",
            lambda x: (x == "Support").sum()
        ),
        complaint_interactions=(
            "interaction_type",
            lambda x: (x == "Complaint").sum()
        )
    )
    .reset_index()
)

customer_360 = customer_360.merge(
    interaction_summary,
    on="customer_id",
    how="left"
)


# =========================================================
# 8. HANDLE MISSING VALUES
# =========================================================

numeric_columns = customer_360.select_dtypes(
    include=["number"]
).columns

customer_360[numeric_columns] = (
    customer_360[numeric_columns].fillna(0)
)


# =========================================================
# 9. FEATURE ENGINEERING - RATIOS
# =========================================================

customer_360["negative_call_rate"] = (
    customer_360["negative_calls"] /
    customer_360["call_count"].replace(0, 1)
)

customer_360["positive_call_rate"] = (
    customer_360["positive_calls"] /
    customer_360["call_count"].replace(0, 1)
)

customer_360["open_complaint_rate"] = (
    customer_360["open_complaints"] /
    customer_360["complaint_count"].replace(0, 1)
)

customer_360["closed_product_rate"] = (
    customer_360["closed_products"] /
    customer_360["total_products"].replace(0, 1)
)

customer_360["support_interaction_rate"] = (
    customer_360["support_interactions"] /
    customer_360["total_interactions"].replace(0, 1)
)


# =========================================================
# 10. GENERATE SYNTHETIC CHURN TARGET
# =========================================================

churn_risk = 0.02

churn_risk += (
    customer_360["complaint_count"] * 0.015
)

churn_risk += (
    customer_360["negative_calls"] * 0.020
)

churn_risk += (
    customer_360["negative_call_rate"] * 0.080
)

churn_risk += (
    customer_360["open_complaints"] * 0.025
)

churn_risk += (
    customer_360["open_complaint_rate"] * 0.060
)

churn_risk += (
    customer_360["closed_products"] * 0.020
)

churn_risk += (
    customer_360["closed_product_rate"] * 0.050
)

churn_risk += (
    customer_360["transaction_count"] < 30
).astype(int) * 0.10

churn_risk += (
    customer_360["total_interactions"] < 5
).astype(int) * 0.05

churn_risk = churn_risk.clip(0, 0.80)

np.random.seed(42)

customer_360["churn_probability"] = churn_risk

customer_360["churned"] = (
    np.random.random(
        len(customer_360)
    ) < churn_risk
).astype(int)


# =========================================================
# 11. SAVE CUSTOMER 360
# =========================================================

customer_360.to_csv(
    "customer_360.csv",
    index=False
)

print("\nCustomer 360 dataset created successfully!")

print("\nChurn Distribution:")

print(
    customer_360["churned"].value_counts()
)


# =========================================================
# 12. BASIC EDA
# =========================================================

print("\n==============================")
print("CHURN ANALYSIS")
print("==============================")

print(
    customer_360.groupby("churned")[
        [
            "age",
            "income",
            "complaint_count",
            "negative_calls",
            "transaction_count",
            "closed_product_rate",
            "support_interaction_rate"
        ]
    ].mean()
)


# =========================================================
# 13. CHURN DISTRIBUTION CHART
# =========================================================

plt.figure(figsize=(6, 4))

customer_360["churned"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churned")
plt.ylabel("Number of Customers")

plt.xticks(
    [0, 1],
    ["Not Churned", "Churned"],
    rotation=0
)

plt.tight_layout()
plt.show()


# =========================================================
# 14. AVERAGE COMPLAINTS CHART
# =========================================================

plt.figure(figsize=(6, 4))

customer_360.groupby("churned")[
    "complaint_count"
].mean().plot(
    kind="bar"
)

plt.title("Average Complaints by Churn Status")
plt.xlabel("Churned")
plt.ylabel("Average Complaints")

plt.xticks(
    [0, 1],
    ["Not Churned", "Churned"],
    rotation=0
)

plt.tight_layout()
plt.show()


# =========================================================
# 15. AVERAGE NEGATIVE CALLS CHART
# =========================================================

plt.figure(figsize=(6, 4))

customer_360.groupby("churned")[
    "negative_calls"
].mean().plot(
    kind="bar"
)

plt.title("Average Negative Calls by Churn Status")
plt.xlabel("Churned")
plt.ylabel("Average Negative Calls")

plt.xticks(
    [0, 1],
    ["Not Churned", "Churned"],
    rotation=0
)

plt.tight_layout()
plt.show()


# =========================================================
# 16. ML FEATURES
# =========================================================

X = customer_360[
    [
        "age",
        "income",
        "transaction_count",
        "total_transaction_amount",
        "avg_transaction_amount",
        "complaint_count",
        "avg_resolution_days",
        "high_severity_complaints",
        "open_complaints",
        "call_count",
        "positive_calls",
        "negative_calls",
        "total_products",
        "active_products",
        "closed_products",
        "total_interactions",
        "support_interactions",
        "complaint_interactions",
        "negative_call_rate",
        "positive_call_rate",
        "open_complaint_rate",
        "closed_product_rate",
        "support_interaction_rate"
    ]
]

y = customer_360["churned"]


# =========================================================
# 17. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==============================")
print("TRAIN TEST DATA")
print("==============================")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\ny_train:")
print(y_train.value_counts())

print("\ny_test:")
print(y_test.value_counts())


# =========================================================
# 18. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# =========================================================
# 19. LOGISTIC REGRESSION
# =========================================================

logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = (
    logistic_model.predict(
        X_test_scaled
    )
)

logistic_probabilities = (
    logistic_model.predict_proba(
        X_test_scaled
    )[:, 1]
)


print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        logistic_predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        logistic_probabilities
    )
)


# =========================================================
# 20. THRESHOLD TUNING
# =========================================================

print("\n==============================")
print("THRESHOLD TUNING")
print("==============================")


for threshold in [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50
]:

    threshold_predictions = (
        logistic_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.2f} | "
        f"Recall: {recall:.2f} | "
        f"F1: {f1:.2f}"
    )


# =========================================================
# 21. DECISION TREE
# =========================================================

decision_tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(
    X_train,
    y_train
)

tree_predictions = (
    decision_tree.predict(
        X_test
    )
)

tree_probabilities = (
    decision_tree.predict_proba(
        X_test
    )[:, 1]
)


print("\n==============================")
print("DECISION TREE")
print("==============================")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        tree_predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        tree_predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        tree_predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        tree_probabilities
    )
)


# =========================================================
# 22. RANDOM FOREST
# =========================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    max_depth=8
)

random_forest.fit(
    X_train,
    y_train
)

forest_predictions = (
    random_forest.predict(
        X_test
    )
)

forest_probabilities = (
    random_forest.predict_proba(
        X_test
    )[:, 1]
)


print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        forest_predictions
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        forest_predictions
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        forest_predictions
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        forest_probabilities
    )
)


# =========================================================
# 23. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": random_forest.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        by="importance",
        ascending=False
    )
)


print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(feature_importance)


# =========================================================
# 24. PREDICT CHURN FOR ALL CUSTOMERS
# =========================================================

all_customers_scaled = scaler.transform(
    X
)

all_churn_probabilities = (
    logistic_model.predict_proba(
        all_customers_scaled
    )[:, 1]
)


# =========================================================
# 25. CREATE CHURN PREDICTION TABLE
# =========================================================

churn_predictions = customer_360[
    [
        "customer_id",
        "name",
        "complaint_count",
        "negative_call_rate",
        "open_complaints",
        "transaction_count"
    ]
].copy()


churn_predictions[
    "predicted_churn_probability"
] = all_churn_probabilities


# =========================================================
# 26. ASSIGN RISK LEVEL
# =========================================================

def assign_risk(probability):

    if probability >= 0.60:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    else:
        return "Low"


churn_predictions["risk_level"] = (
    churn_predictions[
        "predicted_churn_probability"
    ].apply(assign_risk)
)


# =========================================================
# 27. SORT BY CHURN PROBABILITY
# =========================================================

churn_predictions = (
    churn_predictions
    .sort_values(
        by="predicted_churn_probability",
        ascending=False
    )
)


# =========================================================
# 28. DISPLAY CUSTOMER RISK
# =========================================================

print("\n==============================")
print("CUSTOMER CHURN RISK")
print("==============================")

print(
    churn_predictions.head(20)
)


print("\nRisk Level Distribution:")

print(
    churn_predictions[
        "risk_level"
    ].value_counts()
)


# =========================================================
# 29. RISK REASON
# =========================================================

def find_risk_reason(row):

    reasons = []

    if row["negative_call_rate"] >= 0.50:
        reasons.append(
            "High negative call rate"
        )

    if row["open_complaints"] >= 2:
        reasons.append(
            "Multiple open complaints"
        )

    if row["complaint_count"] >= 4:
        reasons.append(
            "High complaint count"
        )

    if row["transaction_count"] < 30:
        reasons.append(
            "Low transaction activity"
        )

    if len(reasons) == 0:

        if row["transaction_count"] < 50:

            reasons.append(
                "Relatively low transaction activity"
            )

        elif row["complaint_count"] > 0:

            reasons.append(
                "Customer has complaint history"
            )

        elif row["negative_call_rate"] > 0:

            reasons.append(
                "Customer has negative call interactions"
            )

        else:

            reasons.append(
                "Low observed customer engagement"
            )

    return ", ".join(reasons)


churn_predictions["risk_reason"] = (
    churn_predictions.apply(
        find_risk_reason,
        axis=1
    )
)


# =========================================================
# 30. RECOMMENDED ACTION
# =========================================================

def recommend_action(row):

    if row["risk_level"] == "High":

        if row["open_complaints"] >= 2:

            return (
                "Escalate and resolve open "
                "complaints immediately"
            )

        elif row["negative_call_rate"] >= 0.50:

            return (
                "Priority customer-service "
                "follow-up"
            )

        elif row["transaction_count"] < 30:

            return (
                "Launch customer "
                "re-engagement campaign"
            )

        elif row["complaint_count"] >= 4:

            return (
                "Assign priority support "
                "and retention offer"
            )

        else:

            return (
                "Proactive retention campaign"
            )


    elif row["risk_level"] == "Medium":

        if row["open_complaints"] >= 1:

            return (
                "Follow up on unresolved complaints"
            )

        elif row["negative_call_rate"] >= 0.40:

            return (
                "Customer-service follow-up"
            )

        else:

            return (
                "Monitor customer activity"
            )


    else:

        return (
            "Continue regular customer engagement"
        )


churn_predictions["recommended_action"] = (
    churn_predictions.apply(
        recommend_action,
        axis=1
    )
)


# =========================================================
# 31. SAVE FINAL CHURN REPORT
# =========================================================

churn_predictions.to_csv(
    "customer_churn_predictions.csv",
    index=False
)


# =========================================================
# 32. FINAL CUSTOMER RISK REPORT
# =========================================================

print("\n==============================")
print("FINAL CUSTOMER RISK REPORT")
print("==============================")

print(
    churn_predictions[
        [
            "customer_id",
            "name",
            "predicted_churn_probability",
            "risk_level",
            "risk_reason",
            "recommended_action"
        ]
    ].head(20)
)


# =========================================================
# 33. RISK REASON SUMMARY
# =========================================================

print("\n==============================")
print("RISK REASON SUMMARY")
print("==============================")

print(
    churn_predictions[
        "risk_reason"
    ].value_counts()
)


# =========================================================
# 34. RISK LEVEL DISTRIBUTION CHART
# =========================================================

plt.figure(figsize=(7, 5))

(
    churn_predictions[
        "risk_level"
    ]
    .value_counts()
    .reindex(
        ["Low", "Medium", "High"]
    )
    .plot(
        kind="bar"
    )
)

plt.title(
    "Customer Churn Risk Distribution"
)

plt.xlabel("Risk Level")

plt.ylabel(
    "Number of Customers"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.show()


# =========================================================
# 35. TOP 10 HIGHEST-RISK CUSTOMERS
# =========================================================

top_10_customers = (
    churn_predictions
    .sort_values(
        by="predicted_churn_probability",
        ascending=False
    )
    .head(10)
)


plt.figure(figsize=(10, 5))

plt.bar(
    top_10_customers["customer_id"],
    top_10_customers[
        "predicted_churn_probability"
    ]
)

plt.title(
    "Top 10 Highest-Risk Customers"
)

plt.xlabel(
    "Customer ID"
)

plt.ylabel(
    "Predicted Churn Probability"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()


# =========================================================
# 36. CHURN PROBABILITY DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    churn_predictions[
        "predicted_churn_probability"
    ],
    bins=20
)

plt.title(
    "Churn Probability Distribution"
)

plt.xlabel(
    "Predicted Churn Probability"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# =========================================================
# 37. RISK REASON DISTRIBUTION
# =========================================================

reason_counts = (
    churn_predictions[
        "risk_reason"
    ]
    .value_counts()
    .head(10)
    .sort_values()
)


plt.figure(figsize=(10, 6))

plt.barh(
    reason_counts.index,
    reason_counts.values
)

plt.title(
    "Top Customer Churn Risk Reasons"
)

plt.xlabel(
    "Number of Customers"
)

plt.ylabel(
    "Risk Reason"
)

plt.tight_layout()

plt.show()


# =========================================================
# 38. FEATURE IMPORTANCE CHART
# =========================================================

top_features = (
    feature_importance
    .head(10)
    .sort_values(
        by="importance"
    )
)


plt.figure(figsize=(9, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "Top 10 Feature Importance - Random Forest"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.show()


# =========================================================
# 39. FINAL OUTPUT INFORMATION
# =========================================================

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")

print(
    "\nCustomer 360 file:"
)

print(
    "customer_360.csv"
)

print(
    "\nCustomer churn prediction file:"
)

print(
    "customer_churn_predictions.csv"
)

print(
    "\nTotal customers:",
    len(customer_360)
)

print(
    "\nRisk level distribution:"
)

print(
    churn_predictions[
        "risk_level"
    ].value_counts()
)

print(
    "\nFinal customer churn report saved successfully!"
)

print(
    "\nCustomer 360 file: customer_360.csv"
)

print(
    "Customer churn file: "
    "customer_churn_predictions.csv"
)
