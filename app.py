import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")


@st.cache_resource
def load_assets():
    # Load dataset
    df = pd.read_csv("breast_cancer.csv")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    if "diagnosis" in df.columns and df["diagnosis"].dtype == object:
        df["diagnosis"] = df["diagnosis"].apply(lambda v: 1 if v == "M" else 0)

    all_feature_cols = [c for c in df.columns if c != "diagnosis"]

    # Load model (✅ match your file name exactly)
    with open("breast_cancer.pkl", "rb") as f:
        model = pickle.load(f)

    # Determine how many features the model expects
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is None:
        raise ValueError("Model does not have n_features_in_. Re-save model with sklearn >= 1.0")

    # Pick first N columns (must match training order)
    feature_cols = all_feature_cols[:expected_n]

    # Fit scaler on ONLY those feature columns
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)

    stats = df[feature_cols].describe().T
    return model, scaler, feature_cols, stats


def predict_one(model, scaler, feature_cols, input_dict):
    X = np.array([input_dict[c] for c in feature_cols], dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(Xs)[0][1])  # prob of Malignant
    return pred, proba


def show_big_result(pred, proba):
    """Simple user-friendly result UI."""
    if pred == 1:
        st.error("🔴 Cancer Detected (Malignant)")
        st.markdown("**⚠️ Please consult a doctor for confirmation.**")
    else:
        st.success("🟢 No Cancer Detected (Benign)")
        st.markdown("**✅ Looks safe, but regular checkups are still important.**")

    if proba is not None:
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"**Probability of Cancer (Malignant):** `{proba*100:.2f}%`")


st.title("🩺 Breast Cancer Prediction (ML App)")
st.caption("Model: SVM (SVC) + StandardScaler")

# Load assets
try:
    model, scaler, feature_cols, stats = load_assets()
except Exception as e:
    st.error(f"Failed to load assets: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🧾 Manual Input",
    "📤 Upload CSV & Predict",
    "📈 Model Performance (Images)",
    "🧠 Causes • Effects • Prevention"
])

# --- Manual Input ---
with tab1:
    st.subheader("Enter Feature Values")

    st.info(
        f"Your model expects **{len(feature_cols)} features**.\n\n"
        "Prediction: **1 = Malignant (Cancer)**, **0 = Benign (Not Cancer)**"
    )

    cols = st.columns(3)
    user_input = {}

    for i, col_name in enumerate(feature_cols):
        c = cols[i % 3]
        mean_val = float(stats.loc[col_name, "mean"])
        min_val = float(stats.loc[col_name, "min"])
        max_val = float(stats.loc[col_name, "max"])

        with c:
            user_input[col_name] = st.number_input(
                label=col_name,
                value=mean_val,
                min_value=min_val,
                max_value=max_val,
                format="%.6f",
            )

    if st.button("🔮 Predict", type="primary"):
        pred, proba = predict_one(model, scaler, feature_cols, user_input)
        show_big_result(pred, proba)


# --- CSV Upload ---
with tab2:
    st.subheader("Upload a CSV file for Batch Prediction")

    st.write(f"Your model expects **{len(feature_cols)} columns**.")
    st.write("If your CSV has extra columns, the app will ignore them.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        data = pd.read_csv(uploaded)

        if "id" in data.columns:
            data = data.drop("id", axis=1)
        if "diagnosis" in data.columns:
            data = data.drop("diagnosis", axis=1)

        # Keep only expected columns
        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        data = data[feature_cols].copy()

        # Predict
        Xs = scaler.transform(data.values)
        preds = model.predict(Xs).astype(int)

        result = data.copy()
        result["prediction"] = preds
        result["prediction_label"] = result["prediction"].map({1: "Malignant", 0: "Benign"})

        if hasattr(model, "predict_proba"):
            result["malignant_probability"] = model.predict_proba(Xs)[:, 1]

        # Move prediction columns to front (so user can see easily)
        front_cols = ["prediction_label", "prediction"]
        if "malignant_probability" in result.columns:
            front_cols.append("malignant_probability")
        result = result[front_cols + [c for c in result.columns if c not in front_cols]]

        # Summary for simple people
        malignant_count = int((result["prediction"] == 1).sum())
        benign_count = int((result["prediction"] == 0).sum())
        total = len(result)

        st.subheader("📌 Result Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", total)
        c2.metric("🔴 Cancer (Malignant)", malignant_count)
        c3.metric("🟢 No Cancer (Benign)", benign_count)

        if malignant_count > 0:
            st.error(f"⚠️ {malignant_count} samples predicted as **Cancer (Malignant)**")
        else:
            st.success("✅ All samples predicted as **No Cancer (Benign)**")

        st.success("✅ Prediction completed!")
        st.dataframe(result, use_container_width=True)

        csv_out = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_out,
            file_name="predictions.csv",
            mime="text/csv",
        )


# --- Images (JPEG) ---
with tab3:
    st.subheader("📈 Model Performance Images")

    colA, colB = st.columns(2)

    pe_path = "PE_breast_cancer.jpeg"
    roc_path = "roc_breast_cancer.jpeg"

    with colA:
        st.write("**Performance Evaluation (Accuracy & ROC)**")
        if os.path.exists(pe_path):
            st.image(pe_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {pe_path}")

    with colB:
        st.write("**ROC Curve Comparison**")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {roc_path}")


# --- Causes, Effects, Prevention ---
with tab4:
    st.subheader("🧠 Causes / Risk Factors")
    st.markdown("""
- **Age:** Risk increases after 40–50 years.
- **Family history / genetics:** BRCA1/BRCA2 mutation or close relatives with breast cancer.
- **Hormonal factors:** Early menstruation, late menopause, first child late, not having children.
- **Lifestyle:** Obesity, low physical activity, alcohol, smoking.
- **Radiation exposure:** Past radiation treatment on chest.
- **Dense breast tissue:** May increase risk slightly.
""")

    st.subheader("⚠️ Effects / Symptoms")
    st.markdown("""
### Early symptoms
- Breast lump or thickening
- Change in breast shape/size
- Nipple discharge (sometimes blood)
- Skin changes (dimpled / redness)
- Nipple pulling inward

### Advanced stage
- Pain, swelling in lymph nodes
- Weight loss, fatigue

### If it spreads (metastasis)
- **Bones:** bone pain  
- **Lungs:** breathing issues  
- **Liver:** jaundice  
- **Brain:** headache, dizziness  
""")

    st.subheader("✅ Prevention / Risk Reduction")
    st.markdown("""
- Maintain healthy weight
- Exercise regularly (150 min/week recommended)
- Eat healthy diet (fruits, vegetables, whole grains)
- Reduce alcohol & avoid smoking
- Regular checkups and screening (mammograms for 40+ / high-risk)
- High-risk people: consult doctor for genetic testing and extra screening
""")

    st.info("This app is for educational use only. Final confirmation requires doctor tests like mammogram/ultrasound/biopsy.")

st.markdown("---")
st.caption("Note: This app is for educational use and not a medical diagnosis tool.")
