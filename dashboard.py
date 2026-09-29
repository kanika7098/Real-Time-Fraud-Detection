import os
import joblib
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

API_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_URL}/predict"


# ============================================================
# LOAD SAVED FILES
# ============================================================

def load_pickle(filename, default=None):

    path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(path):
        return default

    try:
        return joblib.load(path)
    except Exception:
        return default


model_metrics = load_pickle(
    "model_metrics.pkl",
    {}
)

confusion_matrix_data = load_pickle(
    "confusion_matrix.pkl",
    None
)


# ============================================================
# LOAD SHAP DATA
# ============================================================

shap_file = os.path.join(
    MODELS_DIR,
    "top_15_shap_features.csv"
)

if os.path.exists(shap_file):

    try:
        shap_features = pd.read_csv(
            shap_file
        )
    except Exception:
        shap_features = pd.DataFrame()

else:
    shap_features = pd.DataFrame()


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "step": 100.0,
    "transaction_type": "TRANSFER",
    "amount": 50000.0,
    "oldbalanceOrg": 60000.0,
    "hour": 4,
    "day": 4,
    "current_result": None,
    "analysis_count": 0,
    "high_risk_count": 0,
    "medium_risk_count": 0,
    "low_risk_count": 0,
    "history": []
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# API STATUS
# ============================================================

api_online = False

try:

    response = requests.get(
        API_URL,
        timeout=2
    )

    if response.status_code == 200:
        api_online = True

except Exception:
    api_online = False


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ FraudGuard")

st.subheader(
    "Real-Time Payment Fraud Monitoring Center"
)

if api_online:
    st.success("● SYSTEM ONLINE")
else:
    st.error("● API OFFLINE")

st.divider()


# ============================================================
# CURRENT RESULT
# ============================================================

current_probability = 0.0
current_risk = "—"

if st.session_state.current_result:

    current_probability = (
        st.session_state.current_result[
            "fraud_probability"
        ]
    )

    current_risk = (
        st.session_state.current_result[
            "risk_level"
        ]
    )


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("Monitoring Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.metric(
        "Transactions Analyzed",
        st.session_state.analysis_count
    )


with kpi2:

    st.metric(
        "High Risk Alerts",
        st.session_state.high_risk_count
    )


with kpi3:

    st.metric(
        "Current Fraud Probability",
        f"{current_probability:.2f}%"
    )


with kpi4:

    st.metric(
        "Current Risk",
        current_risk
    )


st.write("")


# ============================================================
# TABS
# ============================================================

monitor_tab, performance_tab, explainability_tab = st.tabs(
    [
        "📡 Monitor",
        "📊 Model Performance",
        "🔎 Explainability"
    ]
)


# ============================================================
# MONITOR TAB
# ============================================================

with monitor_tab:

    left_col, right_col = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # TRANSACTION ANALYSIS
    # ========================================================

    with left_col:

        st.subheader(
            "Transaction Analysis"
        )

        with st.container(border=True):

            st.markdown("#### Quick Test")

            quick1, quick2, quick3 = st.columns(3)


            with quick1:

                if st.button(
                    "Normal",
                    use_container_width=True
                ):

                    st.session_state.step = 100.0
                    st.session_state.transaction_type = "TRANSFER"
                    st.session_state.amount = 50000.0
                    st.session_state.oldbalanceOrg = 60000.0
                    st.session_state.hour = 4
                    st.session_state.day = 4
                    st.session_state.current_result = None

                    st.rerun()


            with quick2:

                if st.button(
                    "Known Fraud",
                    use_container_width=True
                ):

                    st.session_state.step = 303.0
                    st.session_state.transaction_type = "CASH_OUT"
                    st.session_state.amount = 5614246.07
                    st.session_state.oldbalanceOrg = 5614246.07
                    st.session_state.hour = 15
                    st.session_state.day = 12
                    st.session_state.current_result = None

                    st.rerun()


            with quick3:

                if st.button(
                    "Reset",
                    use_container_width=True
                ):

                    st.session_state.step = 100.0
                    st.session_state.transaction_type = "TRANSFER"
                    st.session_state.amount = 50000.0
                    st.session_state.oldbalanceOrg = 60000.0
                    st.session_state.hour = 4
                    st.session_state.day = 4
                    st.session_state.current_result = None

                    st.rerun()


            st.divider()


            st.number_input(
                "Simulation Step",
                min_value=0.0,
                key="step"
            )


            st.selectbox(
                "Transaction Type",
                [
                    "PAYMENT",
                    "TRANSFER",
                    "CASH_OUT",
                    "DEBIT",
                    "CASH_IN"
                ],
                key="transaction_type"
            )


            st.number_input(
                "Transaction Amount",
                min_value=0.0,
                step=1000.0,
                key="amount"
            )


            st.number_input(
                "Sender Balance Before Transaction",
                min_value=0.0,
                step=1000.0,
                key="oldbalanceOrg"
            )


            time1, time2 = st.columns(2)


            with time1:

                st.number_input(
                    "Hour",
                    min_value=0,
                    max_value=23,
                    key="hour"
                )


            with time2:

                st.number_input(
                    "Simulation Day",
                    min_value=0,
                    key="day"
                )


            amount_to_balance_ratio = (
                st.session_state.amount
                /
                (
                    st.session_state.oldbalanceOrg
                    + 1
                )
            )


            st.metric(
                "Amount / Sender Balance",
                f"{amount_to_balance_ratio:.2%}"
            )


            analyze = st.button(
                "🔍 Analyze Transaction",
                type="primary",
                use_container_width=True
            )


    # ========================================================
    # RISK ASSESSMENT
    # ========================================================

    with right_col:

        st.subheader(
            "Current Risk Assessment"
        )

        result = st.session_state.current_result


        if result is None:

            with st.container(border=True):

                st.info(
                    "No transaction analyzed yet. "
                    "Enter transaction details and click "
                    "Analyze Transaction."
                )


        else:

            probability = result[
                "fraud_probability"
            ]

            risk_score = result[
                "risk_score"
            ]

            risk_level = result[
                "risk_level"
            ]


            with st.container(border=True):

                if risk_level == "High":

                    st.error(
                        f"🚨 HIGH RISK\n\n"
                        f"Risk Level: {risk_level}"
                    )

                elif risk_level == "Medium":

                    st.warning(
                        f"⚠️ MEDIUM RISK\n\n"
                        f"Risk Level: {risk_level}"
                    )

                else:

                    st.success(
                        f"✅ LOW RISK\n\n"
                        f"Risk Level: {risk_level}"
                    )


                score1, score2 = st.columns(2)


                with score1:

                    st.metric(
                        "Fraud Probability",
                        f"{probability:.2f}%"
                    )


                with score2:

                    st.metric(
                        "Risk Score",
                        f"{risk_score:.2f} / 100"
                    )


                st.write("Risk Score")

                st.progress(
                    min(
                        max(
                            int(risk_score),
                            0
                        ),
                        100
                    )
                )


                st.markdown(
                    "#### Risk Indicators"
                )


                for reason in result[
                    "risk_reasons"
                ]:

                    st.warning(
                        reason
                    )


                st.markdown(
                    "#### Transaction Summary"
                )


                summary1, summary2 = st.columns(2)


                with summary1:

                    st.metric(
                        "Transaction Type",
                        st.session_state.transaction_type
                    )


                with summary2:

                    st.metric(
                        "Amount",
                        f"{st.session_state.amount:,.2f}"
                    )


    # ========================================================
    # PROCESS PREDICTION
    # ========================================================

    if analyze:

        transaction_data = {

            "step":
                float(
                    st.session_state.step
                ),

            "type":
                st.session_state.transaction_type,

            "amount":
                float(
                    st.session_state.amount
                ),

            "oldbalanceOrg":
                float(
                    st.session_state.oldbalanceOrg
                ),

            "hour":
                int(
                    st.session_state.hour
                ),

            "day":
                int(
                    st.session_state.day
                ),

            "amount_to_balance_ratio":
                float(
                    amount_to_balance_ratio
                )
        }


        try:

            prediction_response = requests.post(
                PREDICT_URL,
                json=transaction_data,
                timeout=10
            )


            if prediction_response.status_code != 200:

                st.error(
                    f"API Error: "
                    f"{prediction_response.status_code}"
                )

                st.code(
                    prediction_response.text
                )

            else:

                prediction = (
                    prediction_response.json()
                )


                st.session_state.current_result = (
                    prediction
                )


                st.session_state.analysis_count += 1


                risk = prediction[
                    "risk_level"
                ]


                if risk == "High":

                    st.session_state.high_risk_count += 1

                elif risk == "Medium":

                    st.session_state.medium_risk_count += 1

                else:

                    st.session_state.low_risk_count += 1


                st.session_state.history.insert(
                    0,
                    {
                        "Transaction Type":
                            st.session_state.transaction_type,

                        "Amount":
                            st.session_state.amount,

                        "Fraud Probability":
                            prediction[
                                "fraud_probability"
                            ],

                        "Risk Score":
                            prediction[
                                "risk_score"
                            ],

                        "Risk Level":
                            prediction[
                                "risk_level"
                            ]
                    }
                )


                st.session_state.history = (
                    st.session_state.history[:10]
                )


                st.rerun()


        except requests.exceptions.ConnectionError:

            st.error(
                "FastAPI is not running."
            )


        except requests.exceptions.Timeout:

            st.error(
                "Prediction request timed out."
            )


        except Exception as error:

            st.error(
                f"Unexpected error: {error}"
            )


    # ========================================================
    # RECENT ANALYSES
    # ========================================================

    st.divider()

    st.subheader(
        "Recent Analyses"
    )


    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )


        history_display = history_df.copy()


        history_display["Amount"] = (
            history_display["Amount"]
            .map(
                lambda x:
                    f"{x:,.2f}"
            )
        )


        history_display[
            "Fraud Probability"
        ] = history_display[
            "Fraud Probability"
        ].map(
            lambda x:
                f"{x:.2f}%"
        )


        history_display[
            "Risk Score"
        ] = history_display[
            "Risk Score"
        ].map(
            lambda x:
                f"{x:.2f}"
        )


        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No transactions analyzed during this session."
        )


# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.subheader(
        "Model Performance"
    )


    if model_metrics:

        m1, m2, m3, m4, m5 = st.columns(5)


        with m1:

            st.metric(
                "Precision",
                f"{model_metrics.get('precision', 0):.2%}"
            )


        with m2:

            st.metric(
                "Recall",
                f"{model_metrics.get('recall', 0):.2%}"
            )


        with m3:

            st.metric(
                "F1 Score",
                f"{model_metrics.get('f1_score', 0):.2%}"
            )


        with m4:

            st.metric(
                "ROC-AUC",
                f"{model_metrics.get('roc_auc', 0):.2%}"
            )


        with m5:

            st.metric(
                "PR-AUC",
                f"{model_metrics.get('pr_auc', 0):.2%}"
            )


    else:

        st.warning(
            "Model metrics were not found."
        )


    st.divider()


    if confusion_matrix_data is not None:

        st.subheader(
            "Confusion Matrix"
        )


        cm_df = pd.DataFrame(
            confusion_matrix_data,
            index=[
                "Actual Legitimate",
                "Actual Fraud"
            ],
            columns=[
                "Predicted Legitimate",
                "Predicted Fraud"
            ]
        )


        st.dataframe(
            cm_df,
            use_container_width=True
        )


        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        ax.imshow(
            confusion_matrix_data
        )


        ax.set_title(
            "Fraud Detection Confusion Matrix"
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )


        ax.set_xticks(
            [0, 1]
        )

        ax.set_yticks(
            [0, 1]
        )


        ax.set_xticklabels(
            [
                "Legitimate",
                "Fraud"
            ]
        )


        ax.set_yticklabels(
            [
                "Legitimate",
                "Fraud"
            ]
        )


        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(
                        confusion_matrix_data[
                            i,
                            j
                        ]
                    ),
                    ha="center",
                    va="center"
                )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


    else:

        st.warning(
            "Confusion matrix was not found."
        )


# ============================================================
# EXPLAINABILITY TAB
# ============================================================

with explainability_tab:

    st.subheader(
        "Fraud Feature Explainability"
    )

    st.caption(
        "Features with the largest mean absolute SHAP values."
    )


    if not shap_features.empty:

        chart_data = (
            shap_features
            .sort_values(
                "Importance",
                ascending=True
            )
        )


        fig, ax = plt.subplots(
            figsize=(9, 6)
        )


        ax.barh(
            chart_data["Feature"],
            chart_data["Importance"]
        )


        ax.set_title(
            "Top Fraud Prediction Features"
        )

        ax.set_xlabel(
            "Mean Absolute SHAP Value"
        )

        ax.set_ylabel(
            "Feature"
        )


        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)


        st.dataframe(
            shap_features,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "SHAP feature importance data was not found."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FraudGuard • Real-Time Payment Fraud Monitoring"
)