# Customer 360 & Churn Intelligence

> **An AI-powered platform that brings customer data together, predicts churn risk, explains why customers are at risk, and recommends retention actions.**

---

##  What is this project?

Businesses have customer data spread across transactions, complaints, calls, products, and interactions.

This project combines all of that information into a **single Customer 360 view** and uses **Machine Learning** to identify customers who may be at risk of leaving.

The result is a dashboard that helps businesses answer:

* Who is at risk?
* How high is their churn risk?
* Why are they at risk?
* What action can be taken?

---

##  Problem

Customer information is often stored across multiple systems.

Because of this, businesses may find it difficult to:

* Understand the complete customer journey
* Identify customers at risk of churn
* Detect important behavioral patterns
* Prioritize high-risk customers
* Decide what retention action to take

---

##  Our Solution

**Customer 360 & Churn Intelligence** brings different customer data sources together and converts them into actionable insights.

```text
Multiple Customer Data Sources
              ↓
       Data Integration
              ↓
        Customer 360
              ↓
      Feature Engineering
              ↓
      ML Churn Prediction
              ↓
       Risk Classification
              ↓
      Risk Explanation
              ↓
    Retention Recommendation
              ↓
       Interactive Dashboard
```

---

##  Key Features

###  Customer 360

Creates a unified profile for every customer using:

* Customer information
* Transactions
* Complaints
* Calls
* Products
* Interactions

###  Churn Prediction

Uses Machine Learning to estimate the probability that a customer may churn.

###  Risk Classification

Customers are categorized into:

| Probability  | Risk      |
| ------------ | --------- |
| ≥ 60%        | 🔴 High   |
| 40% – 59.99% | 🟡 Medium |
| < 40%        | 🟢 Low    |

###  Explainable Risk

The system identifies factors associated with a customer's risk, such as:

* High negative call rate
* Multiple open complaints
* High complaint count
* Low transaction activity
* Negative customer interactions

###  Retention Recommendations

The system provides an appropriate suggested action based on the customer's risk and behavior.

Examples:

```text
High Risk
→ Priority customer-service follow-up

Open Complaints
→ Escalate and resolve complaints

Low Transaction Activity
→ Launch re-engagement campaign
```

---

##  Dashboard

The project includes an interactive **Streamlit dashboard**.

The dashboard provides:

### Overview

* Total customers
* High-risk customers
* Medium-risk customers
* Low-risk customers
* Estimated revenue at risk

### Customer Risk Explorer

Users can explore customers based on their risk category.

### Top High-Risk Customers

The dashboard highlights customers with the highest predicted churn probability.

### Customer 360 Profile

Search for a customer and view:

* Age
* Region
* Segment
* Income
* Tenure
* Transactions
* Products
* Complaints
* Calls
* Interactions

### AI Churn Prediction

For each customer:

```text
Churn Probability
        ↓
Risk Level
        ↓
Risk Factors
        ↓
Recommended Action
```

### Customer Report

Individual customer information and risk analysis can be downloaded as a CSV report.

---

##  Machine Learning

The project uses **Logistic Regression** for churn-risk prediction.

The model uses customer behavioral and engagement features such as:

```text
Transaction Activity
Complaint History
Call Sentiment Indicators
Product Activity
Customer Interactions
Support Activity
```

The data is split into training and testing sets using a stratified split.

Feature scaling is performed using `StandardScaler`, and class balancing is applied using `class_weight="balanced"`.

The model produces a probability for each customer rather than only a binary prediction.

Example:

```text
Customer ID: C10775

Churn Probability: 82.11%
Risk Level: High
```

---

##  Feature Engineering

Raw customer data is transformed into meaningful behavioral features.

### Transaction Features

```text
transaction_count
total_transaction_amount
avg_transaction_amount
```

### Complaint Features

```text
complaint_count
avg_resolution_days
high_severity_complaints
open_complaints
open_complaint_rate
```

### Call Features

```text
call_count
positive_calls
negative_calls
negative_call_rate
positive_call_rate
```

### Product Features

```text
total_products
active_products
closed_products
closed_product_rate
```

### Interaction Features

```text
total_interactions
support_interactions
complaint_interactions
support_interaction_rate
```

These features create a more complete representation of customer behavior.

---

##  Data

The project uses six main datasets:

| Dataset                | Purpose                |
| ---------------------- | ---------------------- |
| `customers.csv`        | Customer information   |
| `transactions.csv`     | Transaction history    |
| `complaints.csv`       | Complaint history      |
| `call_transcripts.csv` | Customer-service calls |
| `products.csv`         | Product information    |
| `interactions.csv`     | Customer interactions  |

These datasets are combined to create:

```text
customer_360.csv
```

The final Customer 360 dataset contains aggregated behavioral information for each customer.

---

##  Project Structure

```text
customer-360-churn-intelligence/
│
├── customers.csv
├── transactions.csv
├── complaints.csv
├── call_transcripts.csv
├── products.csv
├── interactions.csv
│
├── customer_360.py
├── customer_360.csv
├── customer_churn_predictions.csv
│
├── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Tech Stack

| Technology   | Usage                 |
| ------------ | --------------------- |
| Python       | Core development      |
| Pandas       | Data processing       |
| NumPy        | Numerical operations  |
| Scikit-learn | Machine Learning      |
| Streamlit    | Interactive dashboard |
| Git          | Version control       |
| GitHub       | Project repository    |

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/sahithi-jalaka/customer-360-churn-intelligence.git
```

### 2. Open the project folder

```bash
cd customer-360-churn-intelligence
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run dashboard.py
```

The application will open in your browser.

---

##  How the System Works

The complete workflow is:

```text
Customers
    │
    ├── Transactions
    ├── Complaints
    ├── Calls
    ├── Products
    └── Interactions
            │
            ▼
      Data Aggregation
            │
            ▼
       Customer 360
            │
            ▼
    Feature Engineering
            │
            ▼
     Machine Learning
            │
            ▼
    Churn Probability
            │
            ▼
      Risk Category
            │
            ▼
      Risk Explanation
            │
            ▼
   Retention Recommendation
            │
            ▼
    Business Dashboard
```

---

##  Business Value

The project transforms customer data into actionable intelligence.

Instead of simply showing historical customer information, it helps identify:

**At-risk customers → Why they are at risk → What action can be considered**

This can help businesses focus customer-service and retention efforts on customers requiring greater attention.

---

##  Future Enhancements

The platform can be extended with:

* Real-time customer data
* NLP-based call sentiment analysis
* Automated complaint sentiment detection
* Customer Lifetime Value prediction
* Real-time churn monitoring
* Automated retention campaigns
* Advanced Explainable AI
* Cloud deployment
* CRM integration
* Automated model retraining

---

##  Disclaimer

This project is developed as a **hackathon-oriented customer analytics and churn prediction solution**.

The datasets are used for analytical and demonstration purposes. Churn labels and predictions should be interpreted within the context of the dataset and machine-learning approach.

The predicted churn probability represents a **model-generated risk estimate**, not a guarantee that a customer will churn.

Revenue-at-risk figures are analytical estimates based on the available customer transaction data.

---

##  Author

### Jalaka Sahithi

**Project:** Customer 360 & Churn Intelligence

**Focus:** AI • Machine Learning • Customer Analytics • Churn Prediction

---

<p align="center">

**Customer 360 & Churn Intelligence**

*Turning customer data into actionable retention intelligence.*

Built with  using Python, Machine Learning & Streamlit

</p>
