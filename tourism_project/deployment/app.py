# ============================================================
# Streamlit App for Tourism Package Prediction
# ============================================================
# This app loads a trained XGBoost model from Hugging Face,
# collects customer inputs, applies the same preprocessing
# as prep.py, and predicts whether the customer will purchase
# the Wellness Tourism Package.

import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib


# ── Load Model from Hugging Face ───────────────────────────
# Replace <your-hf-username> with your Hugging Face username
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="panda1391/tourism-package-prediction",
        filename="tourism-package-prediction_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# ── App Header ─────────────────────────────────────────────
st.title("🌿 Tourism Package Prediction")
st.write("""
**Visit with Us** — Predict whether a customer is likely to purchase the
Wellness Tourism Package based on their profile and interaction data.
Fill in the customer details below and click **Predict**.
""")
st.divider()

# ── Input Section ─────────────────────────────────────────
st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])

with col2:
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, max_value=100000, value=20000, step=500)
    duration_of_pitch = st.slider("Duration of Pitch (mins)", min_value=1, max_value=60, value=10)
    number_of_followups = st.slider("Number of Follow-ups", min_value=1, max_value=10, value=3)
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    number_of_trips = st.slider("Number of Trips/Year", min_value=1, max_value=20, value=3)

col4, col5 = st.columns(2)
with col4:
    num_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
with col5:
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])

st.divider()

# ── Feature Engineering (must mirror prep.py) ─────────────
def get_age_group(age):
    """Replicate the AgeGroup binning from prep.py"""
    if age <= 25:
        return 0  # Young Adult (18-25)
    elif age <= 35:
        return 1  # Early Career (26-35)
    elif age <= 45:
        return 2  # Mid Career (36-45)
    elif age <= 55:
        return 3  # Senior Professional (46-55)
    else:
        return 4  # Elder (56+)

designation_order = {'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP': 3, 'VP': 4}

# Build base input dictionary (numeric + ordinals)
input_dict = {
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_persons_visiting,
    'NumberOfFollowups': number_of_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'Designation': designation_order[designation],
    'AgeGroup': get_age_group(age),
    # One-hot encoded features (TypeofContact)
    'TypeofContact_Self Enquiry': 1 if type_of_contact == "Self Enquiry" else 0,
    # One-hot encoded features (Occupation) — 'Free Lancer' is base
    'Occupation_Large Business': 1 if occupation == "Large Business" else 0,
    'Occupation_Salaried': 1 if occupation == "Salaried" else 0,
    'Occupation_Small Business': 1 if occupation == "Small Business" else 0,
    # One-hot encoded features (Gender) — 'Female' is base
    'Gender_Male': 1 if gender == "Male" else 0,
    # One-hot encoded features (MaritalStatus) — 'Divorced' is base
    'MaritalStatus_Married': 1 if marital_status == "Married" else 0,
    'MaritalStatus_Single': 1 if marital_status == "Single" else 0,
    # One-hot encoded features (ProductPitched) — 'Basic' is base
    'ProductPitched_Deluxe': 1 if product_pitched == "Deluxe" else 0,
    'ProductPitched_King': 1 if product_pitched == "King" else 0,
    'ProductPitched_Standard': 1 if product_pitched == "Standard" else 0,
    'ProductPitched_Super Deluxe': 1 if product_pitched == "Super Deluxe" else 0,
}

input_df = pd.DataFrame([input_dict])

# Ensure columns match training data order
try:
    # Reorder columns to match training set
    training_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(input_dict.keys())
    # Add any missing columns as 0
    for col in training_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[training_cols]
except Exception:
    pass

# ── Prediction ─────────────────────────────────────────────
if st.button("🔮 Predict", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ **Likely to Purchase** the Tourism Package")
        st.info(f"📊 Confidence: **{probability:.1%}** probability of purchase")
    else:
        st.warning(f"❌ **Unlikely to Purchase** the Tourism Package")
        st.info(f"📊 Confidence: **{1 - probability:.1%}** probability of not purchasing")

    st.caption("*Powered by XGBoost | Visit with Us — Tourism Prediction*")
