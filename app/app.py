from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Real-time-safe ML API for online payment fraud detection",
    version="2.0.0"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_model.pkl"
)


PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)


RISK_POLICY_PATH = os.path.join(
    BASE_DIR,
    "models",
    "risk_policy.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


risk_policy = joblib.load(
    RISK_POLICY_PATH
)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(score):

    if score < risk_policy["low_risk_threshold"]:

        return "Low"

    elif score < risk_policy["high_risk_threshold"]:

        return "Medium"

    else:

        return "High"


# ============================================================
# INPUT SCHEMA
# ============================================================

class Transaction(BaseModel):

    step: float

    type: str

    amount: float

    oldbalanceOrg: float

    hour: int

    day: int

    amount_to_balance_ratio: float


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Real-Time Fraud Detection API is running",
        "status": "active",
        "version": "2.0.0"
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # --------------------------------------------------------
    # Convert request to DataFrame
    # --------------------------------------------------------

    transaction_data = transaction.model_dump()

    transaction_df = pd.DataFrame(
        [transaction_data]
    )


    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    transaction_processed = (
        preprocessor.transform(
            transaction_df
        )
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    fraud_probability = model.predict_proba(
        transaction_processed
    )[0, 1]


    risk_score = (
        fraud_probability * 100
    )


    risk_level = get_risk_level(
        risk_score
    )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    risk_reasons = []


    if transaction.amount > 500000:

        risk_reasons.append(
            "Transaction amount is unusually large"
        )


    if transaction.amount_to_balance_ratio > 0.8:

        risk_reasons.append(
            "Transaction is large relative to sender balance"
        )


    if transaction.amount_to_balance_ratio > 0.95:

        risk_reasons.append(
            "Transaction nearly exhausts the sender balance"
        )


    if transaction.type in [
        "TRANSFER",
        "CASH_OUT"
    ]:

        if transaction.amount_to_balance_ratio > 0.8:

            risk_reasons.append(
                "High-value transfer/cash-out relative to sender balance"
            )


    if not risk_reasons:

        risk_reasons.append(
            "No major risk indicators detected"
        )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "fraud_probability": round(
            fraud_probability * 100,
            2
        ),

        "risk_score": round(
            risk_score,
            2
        ),

        "risk_level": risk_level,

        "risk_reasons": risk_reasons
    }