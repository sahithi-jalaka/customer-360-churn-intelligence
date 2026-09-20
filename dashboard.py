import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer 360 AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main content */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        color: #111827;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.05);
        min-height: 120px;
    }

    .card-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
    }

    .card-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
        margin-top: 8px;
    }

    .risk-high {
        color: #dc2626;
        font-weight: 800;
    }

    .risk-medium {
        color: #d97706;
        font-weight: 800;
    }

    .risk-low {
        color: #16a34a;
        font-weight: 800;
    }

    /* Customer profile */
    .profile-card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* AI prediction box */
    .prediction-card {
        background-color: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding-top: 40px;
        font-size: 13px;
    }

    /* Fix Streamlit metric and text visibility */
    .stMetric,
    .stMetric label,
    .stMetric div,
    .stMetric [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    .stProgress {
        color: #111827 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 Customer 360 & Churn Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered customer analytics, churn prediction and retention intelligence'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

customer_360 = pd.read_csv(
    "customer_360.csv"
)

churn_predictions = pd.read_csv(
    "customer_churn_predictions.csv"
)


# ============================================================
# OVERVIEW CALCULATIONS
# ============================================================

total_customers = len(customer_360)

high_risk = len(
    churn_predictions[
        churn_predictions["risk_level"] == "High"
    ]
)

medium_risk = len(
    churn_predictions[
        churn_predictions["risk_level"] == "Medium"
    ]
)

low_risk = len(
    churn_predictions[
        churn_predictions["risk_level"] == "Low"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 👤 Customer Selection"
)

st.sidebar.markdown(
    "Search for a customer using their ID."
)

customer_id = st.sidebar.text_input(
    "Customer ID",
    placeholder="Example: C10775"
)


# ============================================================
# CUSTOMER SELECTION
# ============================================================

if customer_id:

    customer_match = customer_360[
        customer_360["customer_id"].str.upper()
        == customer_id.upper()
    ]

    if len(customer_match) == 0:

        st.sidebar.error(
            "Customer ID not found"
        )

        selected_customer = (
            customer_360.iloc[0]["name"]
        )

    else:

        selected_customer = (
            customer_match.iloc[0]["name"]
        )

else:

    customer_names = (
        customer_360["name"].tolist()
    )

    selected_customer = st.sidebar.selectbox(
        "Or select a customer",
        customer_names
    )


# ============================================================
# SELECTED CUSTOMER DATA
# ============================================================

customer = customer_360[
    customer_360["name"] == selected_customer
].iloc[0]

prediction = churn_predictions[
    churn_predictions["name"] == selected_customer
].iloc[0]


# ============================================================
# OVERVIEW METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📌 Customer Risk Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">TOTAL CUSTOMERS</div>
            <div class="card-value">{total_customers:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">🔴 HIGH RISK</div>
            <div class="card-value risk-high">{high_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">🟡 MEDIUM RISK</div>
            <div class="card-value risk-medium">{medium_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">🟢 LOW RISK</div>
            <div class="card-value risk-low">{low_risk:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📈 Risk Distribution Summary</div>',
    unsafe_allow_html=True
)

high_percentage = (
    high_risk / total_customers
) * 100

medium_percentage = (
    medium_risk / total_customers
) * 100

low_percentage = (
    low_risk / total_customers
) * 100


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "🔴 High Risk",
        f"{high_percentage:.1f}%"
    )

with col2:
    st.metric(
        "🟡 Medium Risk",
        f"{medium_percentage:.1f}%"
    )

with col3:
    st.metric(
        "🟢 Low Risk",
        f"{low_percentage:.1f}%"
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.markdown(
    '<div class="section-title">💰 Business Impact</div>',
    unsafe_allow_html=True
)

high_risk_customers = churn_predictions[
    churn_predictions["risk_level"] == "High"
]

high_risk_ids = high_risk_customers[
    "customer_id"
].tolist()

high_risk_data = customer_360[
    customer_360["customer_id"].isin(
        high_risk_ids
    )
]

revenue_at_risk = high_risk_data[
    "total_transaction_amount"
].sum()


col1, col2 = st.columns(2)


with col1:
    st.metric(
        "High-Risk Customers",
        f"{high_risk:,}"
    )


with col2:
    st.metric(
        "Estimated Revenue at Risk",
        f"₹{revenue_at_risk:,.0f}"
    )


st.caption(
    "Estimated using the historical transaction amount "
    "of customers classified as high risk."
)


# ============================================================
# RISK DISTRIBUTION CHART
# ============================================================

st.markdown(
    '<div class="section-title">📊 Risk Distribution</div>',
    unsafe_allow_html=True
)

risk_counts = pd.Series(
    {
        "High": high_risk,
        "Medium": medium_risk,
        "Low": low_risk
    }
)

st.bar_chart(
    risk_counts,
    height=300
)


# ============================================================
# CUSTOMER RISK FILTER
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Customer Risk Explorer</div>',
    unsafe_allow_html=True
)

risk_filter = st.selectbox(
    "Filter customers by risk level",
    ["All", "High", "Medium", "Low"]
)


if risk_filter == "All":

    filtered_customers = churn_predictions

else:

    filtered_customers = churn_predictions[
        churn_predictions["risk_level"] == risk_filter
    ]


st.write(
    f"Showing **{len(filtered_customers):,} customers**"
)


filtered_table = filtered_customers[
    [
        "customer_id",
        "name",
        "predicted_churn_probability",
        "risk_level"
    ]
].copy()


filtered_table = filtered_table.sort_values(
    by="predicted_churn_probability",
    ascending=False
)


filtered_table["predicted_churn_probability"] = (
    filtered_table[
        "predicted_churn_probability"
    ] * 100
).round(2).astype(str) + "%"


st.dataframe(
    filtered_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TOP HIGH-RISK CUSTOMERS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Top High-Risk Customers</div>',
    unsafe_allow_html=True
)

top_risk_customers = (
    churn_predictions[
        churn_predictions["risk_level"] == "High"
    ]
    .sort_values(
        by="predicted_churn_probability",
        ascending=False
    )
    .head(10)
)


display_table = top_risk_customers[
    [
        "customer_id",
        "name",
        "predicted_churn_probability",
        "risk_level",
        "risk_reason",
        "recommended_action"
    ]
].copy()


display_table[
    "predicted_churn_probability"
] = (
    display_table[
        "predicted_churn_probability"
    ] * 100
).round(2).astype(str) + "%"


st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CUSTOMER 360
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer 360 Profile</div>',
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.markdown(
    '<div class="profile-card">',
    unsafe_allow_html=True
)

st.write("### Customer Information")


col1, col2, col3 = st.columns(3)


with col1:

    st.write("**Customer ID**")
    st.write(customer["customer_id"])

    st.write("**Segment**")
    st.write(customer["segment"])


with col2:

    st.write("**Age**")
    st.write(int(customer["age"]))

    st.write("**Income**")
    st.write(
        f"₹{int(customer['income']):,}"
    )


with col3:

    st.write("**Region**")
    st.write(customer["region"])

    st.write("**Tenure**")
    st.write(
        f"{int(customer['tenure_months'])} months"
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CUSTOMER ACTIVITY
# ============================================================

st.write("### 📌 Customer Activity")


col1, col2, col3 = st.columns(3)


col1.metric(
    "Transactions",
    int(customer["transaction_count"])
)

col2.metric(
    "Complaints",
    int(customer["complaint_count"])
)

col3.metric(
    "Calls",
    int(customer["call_count"])
)


col1.metric(
    "Products",
    int(customer["total_products"])
)

col2.metric(
    "Interactions",
    int(customer["total_interactions"])
)

col3.metric(
    "Negative Call Rate",
    f"{customer['negative_call_rate']:.2%}"
)


# ============================================================
# AI CHURN PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Churn Prediction</div>',
    unsafe_allow_html=True
)

probability = prediction[
    "predicted_churn_probability"
]

col1, col2 = st.columns([1, 2])


with col1:

    st.write("**Churn Probability**")

    st.markdown(
        f"<h2 style='color:#111827; margin-top:0;'>{probability:.2%}</h2>",
        unsafe_allow_html=True
    )


with col2:

    st.write("**AI Churn Risk**")

    st.progress(
        float(probability)
    )

    st.markdown(
        f"<strong style='color:#111827;'>AI Churn Risk: {probability:.2%}</strong>",
        unsafe_allow_html=True
    )


# ============================================================
# RISK LEVEL
# ============================================================

risk_level = prediction[
    "risk_level"
]


if risk_level == "High":

    st.error(
        "🔴 HIGH RISK — Immediate attention recommended"
    )

elif risk_level == "Medium":

    st.warning(
        "🟡 MEDIUM RISK — Customer should be monitored"
    )

else:

    st.success(
        "🟢 LOW RISK — Continue regular engagement"
    )


# ============================================================
# AI EXPLANATION
# ============================================================

st.write("### 🔍 Why is this customer at risk?")

st.info(
    prediction["risk_reason"]
)


# ============================================================
# CONTRIBUTING FACTORS
# ============================================================

st.write("### 📊 Contributing Factors")


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Negative Call Rate",
    f"{customer['negative_call_rate']:.2%}"
)

col2.metric(
    "Complaints",
    int(customer["complaint_count"])
)

col3.metric(
    "Open Complaints",
    int(customer["open_complaints"])
)

col4.metric(
    "Transactions",
    int(customer["transaction_count"])
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "High Severity Complaints",
    int(customer["high_severity_complaints"])
)

col2.metric(
    "Negative Calls",
    int(customer["negative_calls"])
)

col3.metric(
    "Closed Products",
    int(customer["closed_products"])
)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

st.write("### 💡 Recommended Retention Action")

recommended_action = prediction[
    "recommended_action"
]

st.success(
    recommended_action
)


# ============================================================
# DOWNLOAD CUSTOMER REPORT
# ============================================================

st.write("### 📥 Customer Report")


customer_report = pd.DataFrame(
    {
        "Customer ID": [
            customer["customer_id"]
        ],

        "Name": [
            customer["name"]
        ],

        "Age": [
            customer["age"]
        ],

        "Region": [
            customer["region"]
        ],

        "Segment": [
            customer["segment"]
        ],

        "Income": [
            customer["income"]
        ],

        "Tenure Months": [
            customer["tenure_months"]
        ],

        "Transactions": [
            customer["transaction_count"]
        ],

        "Total Transaction Amount": [
            customer["total_transaction_amount"]
        ],

        "Complaints": [
            customer["complaint_count"]
        ],

        "Open Complaints": [
            customer["open_complaints"]
        ],

        "Calls": [
            customer["call_count"]
        ],

        "Negative Calls": [
            customer["negative_calls"]
        ],

        "Negative Call Rate": [
            customer["negative_call_rate"]
        ],

        "Products": [
            customer["total_products"]
        ],

        "Interactions": [
            customer["total_interactions"]
        ],

        "Churn Probability": [
            probability
        ],

        "Risk Level": [
            risk_level
        ],

        "Risk Reason": [
            prediction["risk_reason"]
        ],

        "Recommended Action": [
            recommended_action
        ]
    }
)


csv_report = customer_report.to_csv(
    index=False
)


st.download_button(
    label="📥 Download Customer Report",
    data=csv_report,
    file_name=(
        f"{customer['customer_id']}_customer_report.csv"
    ),
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer 360 AI • Intelligent Churn Prediction & Retention Analytics
    </div>
    """,
    unsafe_allow_html=True
)