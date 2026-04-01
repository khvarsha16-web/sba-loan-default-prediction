"""
Project 1 Scoring Template
SBA Loan Default Prediction

Required Output Format:
    - Pandas DataFrame with columns: index, label, probability_0, probability_1
"""

import pandas as pd
import numpy as np
import pickle


def project_1_scoring(data: pd.DataFrame, model_type: str = "sklearn") -> pd.DataFrame:
    """
    Score input dataset using trained model.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset in the same schema as training data (without MIS_Status column)
    model_type : str
        'sklearn' or 'h2o' (this script implements sklearn path)

    Returns
    -------
    pd.DataFrame
        Columns: index, label, probability_0, probability_1
    """

    if model_type != "sklearn":
        raise NotImplementedError("This scoring script currently implements only model_type='sklearn'.")

    # ----------------------------
    # Load sklearn artifacts
    # ----------------------------
    artifacts = pickle.load(open("./artifacts/sklearn_artifacts.pkl", "rb"))

    model = artifacts["model"]
    threshold = artifacts["threshold"]
    feature_columns = artifacts["feature_columns"]
    target_encoder = artifacts["target_encoder"]
    woe_encoder = artifacts["woe_encoder"]

    # ----------------------------
    # Copy + basic schema handling
    # ----------------------------
    X = data.copy()

    # Keep original index column required for output
    if "index" not in X.columns:
        raise ValueError("Input data must contain 'index' column for output formatting.")

    # Drop target if someone accidentally includes it
    if "MIS_Status" in X.columns:
        X = X.drop(columns=["MIS_Status"])

    # ----------------------------
    # Convert numeric fields (same fields used in FE)
    # ----------------------------
    num_cols = [
        "SBA_Appv",
        "GrAppv",
        "DisbursementGross",
        "BalanceGross",
        "CreateJob",
        "RetainedJob",
        "NoEmp",
        "UrbanRural",
        "FranchiseCode",
    ]

    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # ----------------------------
    # Feature Engineering (11)
    # ----------------------------
    # Note: use +1 to avoid divide-by-zero, and log1p for stability.
    X["loan_to_sba_ratio"] = X["SBA_Appv"] / (X["GrAppv"] + 1)
    X["disbursement_to_approval"] = X["DisbursementGross"] / (X["GrAppv"] + 1)
    X["balance_to_disbursement"] = X["BalanceGross"] / (X["DisbursementGross"] + 1)
    X["jobs_created_ratio"] = X["CreateJob"] / (X["NoEmp"] + 1)
    X["jobs_retained_ratio"] = X["RetainedJob"] / (X["NoEmp"] + 1)
    X["total_jobs"] = X["CreateJob"] + X["RetainedJob"]
    X["jobs_per_employee"] = (X["CreateJob"] + X["RetainedJob"]) / (X["NoEmp"] + 1)
    X["is_urban"] = (X["UrbanRural"] == 1).astype(int)
    X["is_franchise"] = (X["FranchiseCode"] > 0).astype(int)
    X["sba_guarantee_diff"] = X["GrAppv"] - X["SBA_Appv"]
    X["log_approval"] = np.log1p(X["GrAppv"])

    # ----------------------------
    # One-hot encode low-cardinality cols
    # ----------------------------
    onehot_cols = ["RevLineCr", "LowDoc"]
    for c in onehot_cols:
        if c not in X.columns:
            X[c] = np.nan

    X_oh = pd.get_dummies(X, columns=onehot_cols, drop_first=True)

    # ----------------------------
    # Align to training feature columns
    # (create any missing cols as 0; then order)
    # ----------------------------
    for col in feature_columns:
        if col not in X_oh.columns:
            X_oh[col] = 0

    X_oh = X_oh[feature_columns]

    # ----------------------------
    # Apply encoders (no fitting here)
    # ----------------------------
    X_te = target_encoder.transform(X_oh)
    X_enc = woe_encoder.transform(X_te)

    # ----------------------------
    # Final safety cleanup for scoring
    # ----------------------------
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    X_enc = X_enc.fillna(0)

    # ----------------------------
    # Predict
    # ----------------------------
    prob1 = model.predict_proba(X_enc)[:, 1]
    prob0 = 1.0 - prob1
    label = (prob1 >= float(threshold)).astype(int)

    # ----------------------------
    # Output format
    # ----------------------------
    results = pd.DataFrame(
        {
            "index": data["index"].values,
            "label": label,
            "probability_0": prob0,
            "probability_1": prob1,
        }
    )

    return results


def validate_scoring_function():
    """
    Minimal validation on holdout file format.
    """
    holdout = pd.read_csv("./data/SBA_loans_project_1_holdout_students_valid_no_labels.csv.zip")
    preds = project_1_scoring(holdout, model_type="sklearn")

    assert isinstance(preds, pd.DataFrame), "Output must be a DataFrame"
    assert list(preds.columns) == ["index", "label", "probability_0", "probability_1"], "Wrong columns"
    assert len(preds) == len(holdout), "Row count mismatch"
    assert preds["label"].isin([0, 1]).all(), "Labels must be 0/1"
    assert np.max(np.abs((preds["probability_0"] + preds["probability_1"]) - 1.0)) < 1e-6, "Probabilities must sum to 1"
    assert preds.isna().sum().sum() == 0, "No NaNs allowed in output"

    print("✓ Scoring validation passed!")
    return preds


if __name__ == "__main__":
    print("Testing scoring function...")
    _ = validate_scoring_function()