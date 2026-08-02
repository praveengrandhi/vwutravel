import streamlit as st
import pandas as pd
import joblib

model_path = "best_tourism_model.joblib"
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction")
st.write("This app predicts whether a customer is likely to purchase the Wellness Tourism Package.")

st.subheader("Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Female", "Male"])
num_persons = st.number_input("Number of Persons Accompanying", min_value=1, max_value=10, value=2)
property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
num_trips = st.number_input("Number of Trips Annually", min_value=0, max_value=20, value=3)
passport = st.selectbox("Holds Valid Passport?", ["Yes", "No"])
own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
num_children = st.number_input("Number of Children Accompanying (under 5)", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
income = st.number_input("Monthly Income (INR)", min_value=1000, max_value=200000, value=25000)

st.subheader("Customer Interaction Details")
satisfaction = st.slider("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)
pitch_duration = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)

passport_val = 1 if passport == "Yes" else 0
own_car_val = 1 if own_car == "Yes" else 0

input_data = pd.DataFrame([{
    'Age': float(age),
    'TypeofContact': typeof_contact,
    'CityTier': int(city_tier),
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': float(num_persons),
    'PreferredPropertyStar': float(property_star),
    'MaritalStatus': marital_status,
    'NumberOfTrips': float(num_trips),
    'Passport': int(passport_val),
    'OwnCar': int(own_car_val),
    'NumberOfChildrenVisiting': float(num_children),
    'Designation': designation,
    'MonthlyIncome': float(income),
    'PitchSatisfactionScore': float(satisfaction),
    'ProductPitched': product_pitched,
    'NumberOfFollowups': float(followups),
    'DurationOfPitch': float(pitch_duration)
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Likely to Purchase! (Probability: {prob:.2%})")
    else:
        st.warning(f"Unlikely to Purchase. (Probability: {prob:.2%})")

