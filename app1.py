import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sqlite3
import bcrypt

# --- Database Functions ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return bcrypt.checkpw(password.encode('utf-8'), result[0])
    return False

# --- Policy Management Database Functions ---

def create_policy_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS policies (
            policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plan_type TEXT NOT NULL,
            is_renewal BOOLEAN NOT NULL,
            age INTEGER,
            sex TEXT,
            bmi REAL,
            children INTEGER,
            smoker TEXT,
            region TEXT,
            -- Removed original Python-style comments from here --
            HighBP INTEGER,
            HighChol INTEGER,
            CholCheck INTEGER,
            PhysActivity INTEGER,
            Fruits INTEGER,
            Veggies INTEGER,
            HvyAlcoholConsump INTEGER,
            AnyHealthcare INTEGER,
            NoDocbcCost INTEGER,
            GenHlth INTEGER,
            MentHlth INTEGER,
            PhysHlth INTEGER,
            DiffWalk INTEGER,
            Education INTEGER,
            Income INTEGER,
            Stroke INTEGER,
            HeartDiseaseorAttack INTEGER,
            -- Removed original Python-style comments from here --
            predicted_expenses REAL,
            predicted_risk TEXT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

def save_policy_data(username, plan_type, is_renewal, age, sex, bmi, children, smoker, region,
                     high_bp, high_chol, chol_check, phys_activity, fruits, veggies, hvy_alcohol_consump,
                     any_healthcare, no_doc_bc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
                     education, income, stroke, heart_disease_attack,
                     predicted_expenses, predicted_risk):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO policies (
            username, plan_type, is_renewal, age, sex, bmi, children, smoker, region,
            HighBP, HighChol, CholCheck, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
            AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
            Education, Income, Stroke, HeartDiseaseorAttack,
            predicted_expenses, predicted_risk
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username, plan_type, is_renewal, age, sex, bmi, children, smoker, region,
        int(high_bp), int(high_chol), int(chol_check), int(phys_activity), int(fruits), int(veggies), int(hvy_alcohol_consump),
        int(any_healthcare), int(no_doc_bc_cost), int(gen_hlth), int(ment_hlth), int(phys_hlth), int(diff_walk),
        int(education), int(income), int(stroke), int(heart_disease_attack),
        predicted_expenses, predicted_risk
    ))
    conn.commit()
    conn.close()
    return c.lastrowid # Return the ID of the newly inserted row

def get_latest_policy_data(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Get the latest policy entry for the user, ordered by timestamp
    c.execute('''
        SELECT
            plan_type, age, sex, bmi, children, smoker, region,
            HighBP, HighChol, CholCheck, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
            AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
            Education, Income, Stroke, HeartDiseaseorAttack,
            predicted_expenses, predicted_risk,
            policy_id -- This is a SQL comment, valid within the string
        FROM policies
        WHERE username = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        # Map boolean-like integers back to booleans for checkboxes
        # Ensure correct type mapping for all fields
        columns = [
            "plan_type", "age", "sex", "bmi", "children", "smoker", "region",
            "HighBP", "HighChol", "CholCheck", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
            "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Education", "Income", "Stroke", "HeartDiseaseorAttack",
            "predicted_expenses", "predicted_risk", "policy_id"
        ]
        policy_data = dict(zip(columns, result))

        # Convert 0/1 integers back to boolean for Streamlit checkboxes
        for key in ["HighBP", "HighChol", "CholCheck", "PhysActivity", "Fruits", "Veggies",
                    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Stroke", "HeartDiseaseorAttack"]:
            policy_data[key] = bool(policy_data[key])
        return policy_data
    return None

def update_policy_data(policy_id, age, sex, bmi, children, smoker, region,
                       high_bp, high_chol, chol_check, phys_activity, fruits, veggies, hvy_alcohol_consump,
                       any_healthcare, no_doc_bc_cost, gen_hlth, ment_hlth, phys_hlth, diff_walk,
                       education, income, stroke, heart_disease_attack,
                       predicted_expenses, predicted_risk):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        UPDATE policies SET
            timestamp = CURRENT_TIMESTAMP, -- Update timestamp on renewal
            age = ?, sex = ?, bmi = ?, children = ?, smoker = ?, region = ?,
            HighBP = ?, HighChol = ?, CholCheck = ?, PhysActivity = ?, Fruits = ?, Veggies = ?,
            HvyAlcoholConsump = ?, AnyHealthcare = ?, NoDocbcCost = ?, GenHlth = ?,
            MentHlth = ?, PhysHlth = ?, DiffWalk = ?, Education = ?, Income = ?,
            Stroke = ?, HeartDiseaseorAttack = ?,
            predicted_expenses = ?, predicted_risk = ?,
            is_renewal = TRUE -- Mark as renewal
        WHERE policy_id = ?
    ''', (
        age, sex, bmi, children, smoker, region,
        int(high_bp), int(high_chol), int(chol_check), int(phys_activity), int(fruits), int(veggies), int(hvy_alcohol_consump),
        int(any_healthcare), int(no_doc_bc_cost), int(gen_hlth), int(ment_hlth), int(phys_hlth), int(diff_walk),
        int(education), int(income), int(stroke), int(heart_disease_attack),
        predicted_expenses, predicted_risk,
        policy_id
    ))
    conn.commit()
    conn.close()

# Initialize the database and policy table
init_db()
create_policy_table() # Ensure the policy table is created

# Load trained models
try:
    lr_regressor = pickle.load(open("charges_model.pkl", "rb"))
    rf_classifier = pickle.load(open("diabetes_model.pkl", "rb"))
    # Load both specific label encoders
    le_gender = pickle.load(open("label_encoder_gender.pkl", "rb"))
    le_smoker = pickle.load(open("label_encoder_smoker.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure 'charges_model.pkl'" \
    ", 'diabetes_model.pkl', 'label_encoder_gender.pkl', and 'label_encoder_smoker.pkl' are in the same directory.")
    st.stop()
except Exception as e: # Catch other potential loading errors like UnpicklingError
    st.error(f"Error loading models or encoders: {e}. Please ensure the .pkl files are not corrupted and were saved correctly.")
    st.stop()


# Define expected features for the charges model (now including 'smoker')
expected_charges_features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']

# Define expected features for the diabetes model (unchanged)
model_features_diabetes = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

# ========== Risk Logic (No change) ==========
def map_age_group(code):
    if code <= 6:
        return "young"
    elif 7 <= code <= 9:
        return "middle"
    else:
        return "senior"

def custom_risk(row):
    age_group = map_age_group(row["Age"])
    if row["HeartDiseaseorAttack"] == 1 or row["Stroke"] == 1:
        return "High"
    elif row["predicted_diabetes"] == 1 and (row["BMI"] > 30 or row["Smoker"] == 1):
        return "High"
    elif age_group == "senior":
        return "High"
    elif row["BMI"] > 30 or row["Smoker"] == 1 or row["HighBP"] == 1 or age_group == "middle":
        return "Medium"
    else:
        return "Low"

# ========== Streamlit UI ==========

st.title("🩺 Medical Insurance Expenses & Health Risk Predictor")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.sidebar.header("Login / Register")
    login_option = st.sidebar.radio("Choose an option", ["Login", "Register"])

    if login_option == "Register":
        with st.sidebar.form("Register Form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_button = st.form_submit_button("Register")

            if register_button:
                if new_password == confirm_password:
                    if new_username and new_password:
                        if add_user(new_username, new_password):
                            st.sidebar.success("Registration successful! Please login.")
                        else:
                            st.sidebar.error("Username already exists. Please choose a different one.")
                    else:
                        st.sidebar.warning("Username and password cannot be empty.")
                else:
                    st.sidebar.error("Passwords do not match.")

    elif login_option == "Login":
        with st.sidebar.form("Login Form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if verify_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.sidebar.success(f"Welcome, {username}!")
                    st.rerun() # Use st.rerun() for reloading the app
                else:
                    st.sidebar.error("Invalid username or password.")
else: # User is logged in
    st.sidebar.success(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        # Reset policy_status and plan_type on logout
        if "policy_status" in st.session_state:
            del st.session_state["policy_status"]
        if "plan_type" in st.session_state:
            del st.session_state["plan_type"]
        if "renewal_data" in st.session_state: # Clear renewal data too
            del st.session_state["renewal_data"]
        st.rerun()

    # --- Policy Status Selection (New or Renewal) ---
    if "policy_status" not in st.session_state:
        st.header("Policy Status")
        policy_choice = st.radio(
            "Is this a new policy or a renewal?",
            ["New Policy", "Renewal"]
        )
        if st.button("Proceed to Plan Selection"):
            st.session_state["policy_status"] = policy_choice
            # Attempt to load data if it's a renewal
            if policy_choice == "Renewal":
                latest_data = get_latest_policy_data(st.session_state["username"])
                if latest_data:
                    st.session_state["renewal_data"] = latest_data
                    st.success("Loaded your latest policy data for renewal.")
                else:
                    st.warning("No previous policy data found. Proceeding as 'New Policy' for calculations.")
                    st.session_state["policy_status"] = "New Policy" # Fallback if no data
            st.rerun()
    else: # Policy status is selected
        # Option to change policy status
        if st.button("Change Policy Status"):
            del st.session_state["policy_status"]
            # Also reset plan_type if policy status is changed
            if "plan_type" in st.session_state:
                del st.session_state["plan_type"]
            if "renewal_data" in st.session_state: # Clear renewal data too
                del st.session_state["renewal_data"]
            st.rerun()

        # --- Existing Plan Type Selection now nested here ---
        if "plan_type" not in st.session_state:
            st.header("Choose Plan Type")
            plan_choice = st.radio(
                "Select the type of insurance plan you want to predict:",
                ["Individual", "Family"]
            )
            if st.button("Proceed to Details"):
                st.session_state["plan_type"] = plan_choice
                st.rerun()
        else: # Both policy status and plan type are selected
            # Option to change plan type (moved here)
            if st.button("Change Plan Type"):
                del st.session_state["plan_type"]
                st.rerun()

            # --- INDIVIDUAL PLAN PREDICTION ---
            if st.session_state["plan_type"] == "Individual":
                st.subheader("👤 Individual Plan Prediction")
                st.sidebar.header("Enter your details:")

                # Get default values for inputs, pre-filling if renewal_data exists
                default_data = st.session_state.get("renewal_data", {})

                age = st.sidebar.number_input("Age", min_value=18, max_value=100,
                                              value=default_data.get("age", 30))
                bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0,
                                              value=default_data.get("bmi", 25.0))
                smoker_options = ["no", "yes"]
                smoker_index = smoker_options.index(default_data.get("smoker", "no"))
                smoker_input = st.sidebar.selectbox("Do you smoke?", smoker_options,
                                                    index=smoker_index)

                gender_options = ["female", "male"]
                gender_index = gender_options.index(default_data.get("sex", "female"))
                gender_input = st.sidebar.selectbox("Gender", gender_options,
                                                    index=gender_index)
                children = st.sidebar.slider("Number of children", 0, 5,
                                             value=default_data.get("children", 0))
                region_options = ["southwest", "southeast", "northwest", "northeast"]
                region_index = region_options.index(default_data.get("region", "southwest"))
                region = st.sidebar.selectbox("Region", region_options,
                                              index=region_index)

                st.sidebar.subheader("Additional Health Details (for Risk Prediction):")

                high_bp = st.sidebar.checkbox("High Blood Pressure?",
                                              value=default_data.get("HighBP", False))
                high_chol = st.sidebar.checkbox("High Cholesterol?",
                                                value=default_data.get("HighChol", False))
                chol_check = st.sidebar.checkbox("Cholesterol Check in last 5 years?",
                                                 value=default_data.get("CholCheck", True))
                phys_activity = st.sidebar.checkbox("Physical Activity in last 30 days?",
                                                    value=default_data.get("PhysActivity", True))
                fruits = st.sidebar.checkbox("Consume Fruits daily?",
                                             value=default_data.get("Fruits", True))
                veggies = st.sidebar.checkbox("Consume Vegetables daily?",
                                              value=default_data.get("Veggies", True))
                hvy_alcohol_consump = st.sidebar.checkbox("Heavy Alcohol Consumption (men >=14 drinks/week, women >=7 drinks/week)?",
                                                        value=default_data.get("HvyAlcoholConsump", False))
                any_healthcare = st.sidebar.checkbox("Have any healthcare coverage?",
                                                     value=default_data.get("AnyHealthcare", True))
                no_doc_bc_cost = st.sidebar.checkbox("Could not see doctor due to cost?",
                                                     value=default_data.get("NoDocbcCost", False))
                diff_walk = st.sidebar.checkbox("Have serious difficulty walking or climbing stairs?",
                                                value=default_data.get("DiffWalk", False))
                stroke = st.sidebar.checkbox("Ever had a Stroke?",
                                             value=default_data.get("Stroke", False))
                heart_disease_attack = st.sidebar.checkbox("Ever had Heart Disease or Attack?",
                                                            value=default_data.get("HeartDiseaseorAttack", False))

                gen_hlth_options = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
                gen_hlth_value = default_data.get("GenHlth", 3)
                gen_hlth_index = list(gen_hlth_options.keys()).index(gen_hlth_value)
                gen_hlth = st.sidebar.selectbox("General Health",
                                                options=list(gen_hlth_options.keys()),
                                                format_func=lambda x: gen_hlth_options[x],
                                                index=gen_hlth_index)

                ment_hlth = st.sidebar.slider("Days of poor mental health in last 30 days", 0, 30,
                                            value=default_data.get("MentHlth", 0))
                phys_hlth = st.sidebar.slider("Days of poor physical health in last 30 days", 0, 30,
                                            value=default_data.get("PhysHlth", 0))

                edu_options = {
                    1: "Never attended school / Kindergarten", 2: "Grades 1-8 (Elementary)",
                    3: "Grades 9-11 (Some High School)", 4: "High School Graduate",
                    5: "Some College / Technical School", 6: "College Graduate or more"
                }
                edu_value = default_data.get("Education", 4)
                edu_index = list(edu_options.keys()).index(edu_value)
                education = st.sidebar.selectbox("Education Level",
                                                options=list(edu_options.keys()),
                                                format_func=lambda x: edu_options[x],
                                                index=edu_index)

                income_options = {
                    1: "< $10,000", 2: "$10,000 - $14,999", 3: "$15,000 - $19,999",
                    4: "$20,000 - $24,999", 5: "$25,000 - $34,999", 6: "$35,000 - $49,999",
                    7: "$50,000 - $74,999", 8: "$75,000 or more"
                }
                income_value = default_data.get("Income", 5)
                income_index = list(income_options.keys()).index(income_value)
                income = st.sidebar.selectbox("Income Level (Annual Household Income)",
                                            options=list(income_options.keys()),
                                            format_func=lambda x: income_options[x],
                                            index=income_index)

                # Prepare input for charges model
                input_dict = {
                    "age": age,
                    "bmi": bmi,
                    "children": children,
                    "sex": le_gender.transform([gender_input])[0],
                    "smoker": le_smoker.transform([smoker_input])[0],
                    "region_northwest": 0,
                    "region_southeast": 0,
                    "region_southwest": 0
                }

                if region == "northwest":
                    input_dict['region_northwest'] = 1
                elif region == "southeast":
                    input_dict['region_southeast'] = 1
                elif region == "southwest":
                    input_dict['region_southwest'] = 1

                X_charge = pd.DataFrame([input_dict])
                X_charge = X_charge[expected_charges_features]

                predicted_expense = lr_regressor.predict(X_charge)[0]

                # Prepare input for diabetes + risk model
                age_code = pd.cut([age], bins=[0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 200],
                                labels=[1,2,3,4,5,6,7,8,9,10,11,12,13]).astype(float)[0]

                X_risk = pd.DataFrame([{
                        "Age": age_code,
                        "BMI": bmi,
                        "Smoker": 1.0 if smoker_input == "yes" else 0.0,
                        "Sex": 1.0 if gender_input == "male" else 0.0,
                        "HighBP": 1.0 if high_bp else 0.0,
                        "HighChol": 1.0 if high_chol else 0.0,
                        "CholCheck": 1.0 if chol_check else 0.0,
                        "PhysActivity": 1.0 if phys_activity else 0.0,
                        "Fruits": 1.0 if fruits else 0.0,
                        "Veggies": 1.0 if veggies else 0.0,
                        "HvyAlcoholConsump": 1.0 if hvy_alcohol_consump else 0.0,
                        "AnyHealthcare": 1.0 if any_healthcare else 0.0,
                        "NoDocbcCost": 1.0 if no_doc_bc_cost else 0.0,
                        "DiffWalk": 1.0 if diff_walk else 0.0,
                        "Stroke": 1.0 if stroke else 0.0,
                        "HeartDiseaseorAttack": 1.0 if heart_disease_attack else 0.0,
                        "GenHlth": float(gen_hlth),
                        "MentHlth": float(ment_hlth),
                        "PhysHlth": float(phys_hlth),
                        "Education": float(education),
                        "Income": float(income)
                    }])

                X_risk = X_risk[model_features_diabetes]

                X_risk["predicted_diabetes"] = rf_classifier.predict(X_risk)
                X_risk["risk_level"] = X_risk.apply(custom_risk, axis=1)
                predicted_risk = X_risk['risk_level'].iloc[0] # Capture predicted_risk

                st.subheader("📊 Prediction Results:")
                st.markdown(f"💰 **Estimated Insurance Expenses:** ₹ {predicted_expense:,.2f}")
                st.markdown(f"⚠️ **Estimated Health Risk:** `{predicted_risk}`")

                # --- Save/Update Policy Data for Individual ---
                if st.button("Save/Update Policy Data"):
                    if st.session_state["policy_status"] == "New Policy":
                        save_policy_data(
                            username=st.session_state["username"],
                            plan_type=st.session_state["plan_type"],
                            is_renewal=False, # New policy
                            age=age, sex=gender_input, bmi=bmi, children=children, smoker=smoker_input, region=region,
                            high_bp=high_bp, high_chol=high_chol, chol_check=chol_check, phys_activity=phys_activity,
                            fruits=fruits, veggies=veggies, hvy_alcohol_consump=hvy_alcohol_consump,
                            any_healthcare=any_healthcare, no_doc_bc_cost=no_doc_bc_cost, gen_hlth=gen_hlth,
                            ment_hlth=ment_hlth, phys_hlth=phys_hlth, diff_walk=diff_walk,
                            education=education, income=income, stroke=stroke, heart_disease_attack=heart_disease_attack,
                            predicted_expenses=predicted_expense, predicted_risk=predicted_risk
                        )
                        st.success("New policy data saved successfully!")
                    elif st.session_state["policy_status"] == "Renewal":
                        if "renewal_data" in st.session_state and st.session_state["renewal_data"]:
                            policy_id_to_update = st.session_state["renewal_data"]["policy_id"]
                            update_policy_data(
                                policy_id=policy_id_to_update,
                                age=age, sex=gender_input, bmi=bmi, children=children, smoker=smoker_input, region=region,
                                high_bp=high_bp, high_chol=high_chol, chol_check=chol_check, phys_activity=phys_activity,
                                fruits=fruits, veggies=veggies, hvy_alcohol_consump=hvy_alcohol_consump,
                                any_healthcare=any_healthcare, no_doc_bc_cost=no_doc_bc_cost, gen_hlth=gen_hlth,
                                ment_hlth=ment_hlth, phys_hlth=phys_hlth, diff_walk=diff_walk,
                                education=education, income=income, stroke=stroke, heart_disease_attack=heart_disease_attack,
                                predicted_expenses=predicted_expense, predicted_risk=predicted_risk
                            )
                            st.success(f"Policy ID {policy_id_to_update} updated successfully!")
                        else:
                            st.error("Cannot update: No existing policy data found for renewal.")


            # --- FAMILY PLAN PREDICTION ---
            elif st.session_state["plan_type"] == "Family":
                st.subheader("👨‍👩‍👧‍👦 Family Plan Prediction")

                num_family_members = st.slider("Number of Family Members (including yourself)", 1, 10, 2)

                family_data = []
                total_predicted_expense = 0.0
                family_risk_levels = [] # To store risk level for each member

                regions_list = ["southwest", "southeast", "northwest", "northeast"]

                st.write("---") # Separator

                for i in range(num_family_members):
                    st.markdown(f"**Member {i+1} Details:**")
                    col1, col2 = st.columns(2)

                    # For family, we don't pre-fill each member from a single "renewal_data"
                    # If you want to handle family renewals with pre-fill, it gets more complex
                    # (e.g., storing each family member's past data). For now, we treat family
                    # renewal as recalculating for potentially new family members.
                    # You could modify this to load a list of family members if stored previously.

                    with col1:
                        age_member = st.number_input(f"Age (Member {i+1})", min_value=0, max_value=100, value=30, key=f"age_{i}")
                        smoker_member = st.selectbox(f"Smoker? (Member {i+1})", ["no", "yes"], key=f"smoker_{i}")
                        bmi_member = st.number_input(f"BMI (Member {i+1})", min_value=10.0, max_value=50.0, value=25.0, key=f"bmi_{i}")
                    with col2:
                        gender_member = st.selectbox(f"Gender (Member {i+1})", ["female", "male"], key=f"gender_{i}")
                        children_member = st.slider(f"Children (Member {i+1})", 0, 5, 0, key=f"children_{i}")
                        region_member = st.selectbox(f"Region (Member {i+1})", regions_list, key=f"region_{i}")

                    # Simplified/default health features for family members to prevent overly long forms
                    high_bp_member = False
                    high_chol_member = False
                    chol_check_member = True
                    phys_activity_member = True
                    fruits_member = True
                    veggies_member = True
                    hvy_alcohol_consump_member = False
                    any_healthcare_member = True
                    no_doc_bc_cost_member = False
                    diff_walk_member = False
                    stroke_member = False
                    heart_disease_attack_member = False
                    gen_hlth_member = 3
                    ment_hlth_member = 0
                    phys_hlth_member = 0
                    education_member = 4
                    income_member = 5

                    family_data.append({
                        "age": age_member,
                        "bmi": bmi_member,
                        "children": children_member,
                        "sex": gender_member,
                        "smoker": smoker_member,
                        "region": region_member,
                        "HighBP": high_bp_member,
                        "HighChol": high_chol_member,
                        "CholCheck": chol_check_member,
                        "PhysActivity": phys_activity_member,
                        "Fruits": fruits_member,
                        "Veggies": veggies_member,
                        "HvyAlcoholConsump": hvy_alcohol_consump_member,
                        "AnyHealthcare": any_healthcare_member,
                        "NoDocbcCost": no_doc_bc_cost_member,
                        "GenHlth": gen_hlth_member,
                        "MentHlth": ment_hlth_member,
                        "PhysHlth": phys_hlth_member,
                        "DiffWalk": diff_walk_member,
                        "Education": education_member,
                        "Income": income_member,
                        "Stroke": stroke_member,
                        "HeartDiseaseorAttack": heart_disease_attack_member
                    })
                    st.write("---")

                if st.button("Predict Family Expenses and Risks"):
                    family_records_to_save = [] # Collect data for saving/updating
                    current_family_predicted_risks = [] # For overall summary

                    for member_num, member in enumerate(family_data):
                        st.markdown(f"### Results for Member {member_num + 1}:")

                        # Prepare input for charges model for this member
                        input_dict_member_charge = {
                            "age": member["age"],
                            "bmi": member["bmi"],
                            "children": member["children"],
                            "sex": le_gender.transform([member["sex"]])[0],
                            "smoker": le_smoker.transform([member["smoker"]])[0],
                            "region_northwest": 0,
                            "region_southeast": 0,
                            "region_southwest": 0
                        }
                        if member["region"] == "northwest":
                            input_dict_member_charge['region_northwest'] = 1
                        elif member["region"] == "southeast":
                            input_dict_member_charge['region_southeast'] = 1
                        elif member["region"] == "southwest":
                            input_dict_member_charge['region_southwest'] = 1

                        X_charge_member = pd.DataFrame([input_dict_member_charge])
                        X_charge_member = X_charge_member[expected_charges_features]
                        predicted_expense_member = lr_regressor.predict(X_charge_member)[0]
                        total_predicted_expense += predicted_expense_member

                        st.markdown(f"💰 **Estimated Insurance Expenses:** ₹ {predicted_expense_member:,.2f}")

                        # Prepare input for diabetes + risk model for this member
                        age_code_member = pd.cut([member["age"]], bins=[0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 200],
                                              labels=[1,2,3,4,5,6,7,8,9,10,11,12,13]).astype(float)[0]

                        X_risk_member = pd.DataFrame([{
                                "Age": age_code_member,
                                "BMI": member["bmi"],
                                "Smoker": 1.0 if member["smoker"] == "yes" else 0.0,
                                "Sex": 1.0 if member["sex"] == "male" else 0.0,
                                "HighBP": 1.0 if member["HighBP"] else 0.0,
                                "HighChol": 1.0 if member["HighChol"] else 0.0,
                                "CholCheck": 1.0 if member["CholCheck"] else 0.0,
                                "PhysActivity": 1.0 if member["PhysActivity"] else 0.0,
                                "Fruits": 1.0 if member["Fruits"] else 0.0,
                                "Veggies": 1.0 if member["Veggies"] else 0.0,
                                "HvyAlcoholConsump": 1.0 if member["HvyAlcoholConsump"] else 0.0,
                                "AnyHealthcare": 1.0 if member["AnyHealthcare"] else 0.0,
                                "NoDocbcCost": 1.0 if member["NoDocbcCost"] else 0.0,
                                "DiffWalk": 1.0 if member["DiffWalk"] else 0.0,
                                "Stroke": 1.0 if member["Stroke"] else 0.0,
                                "HeartDiseaseorAttack": 1.0 if member["HeartDiseaseorAttack"] else 0.0,
                                "GenHlth": float(member["GenHlth"]),
                                "MentHlth": float(member["MentHlth"]),
                                "PhysHlth": float(member["PhysHlth"]),
                                "Education": float(member["Education"]),
                                "Income": float(member["Income"])
                            }])

                        X_risk_member = X_risk_member[model_features_diabetes]
                        X_risk_member["predicted_diabetes"] = rf_classifier.predict(X_risk_member)
                        X_risk_member["risk_level"] = X_risk_member.apply(custom_risk, axis=1)
                        risk_level_member = X_risk_member['risk_level'].iloc[0]
                        current_family_predicted_risks.append(risk_level_member)

                        st.markdown(f"⚠️ **Estimated Health Risk:** `{risk_level_member}`")
                        st.write("") # Add a small space

                        # Prepare data for saving for this member
                        family_records_to_save.append({
                            "age": member["age"], "sex": member["sex"], "bmi": member["bmi"],
                            "children": member["children"], "smoker": member["smoker"], "region": member["region"],
                            "HighBP": member["HighBP"], "HighChol": member["HighChol"], "CholCheck": member["CholCheck"],
                            "PhysActivity": member["PhysActivity"], "Fruits": member["Fruits"], "Veggies": member["Veggies"],
                            "HvyAlcoholConsump": member["HvyAlcoholConsump"], "AnyHealthcare": member["AnyHealthcare"],
                            "NoDocbcCost": member["NoDocbcCost"], "GenHlth": member["GenHlth"],
                            "MentHlth": member["MentHlth"], "PhysHlth": member["PhysHlth"], "DiffWalk": member["DiffWalk"],
                            "Education": member["Education"], "Income": member["Income"], "Stroke": member["Stroke"],
                            "HeartDiseaseorAttack": member["HeartDiseaseorAttack"],
                            "predicted_expenses": predicted_expense_member,
                            "predicted_risk": risk_level_member
                        })

                    st.subheader("--- Overall Family Summary ---")
                    st.markdown(f"💰 **Total Estimated Family Insurance Expenses:** ₹ {total_predicted_expense:,.2f}")
                    st.markdown(f"⚠️ **Family Health Risk Overview:**")
                    unique_risks = ", ".join(sorted(list(set(current_family_predicted_risks)), key=lambda x: {"Low":1, "Medium":2, "High":3}[x]))
                    st.markdown(f"The family members have risk levels of: `{unique_risks}`.")

                    # --- Save/Update Policy Data for Family ---
                    if st.button("Save Family Policy Data"):
                        if family_records_to_save:
                            # Save the first member's data, marking it as part of a family plan
                            # This is a compromise: we're saving one aggregated record for the family,
                            # using the primary member's demographic details, but the total expenses
                            # and aggregated risk for the whole family.
                            main_member_data = family_records_to_save[0]
                            main_member_data["predicted_expenses"] = total_predicted_expense
                            main_member_data["predicted_risk"] = ", ".join(sorted(list(set(current_family_predicted_risks)), key=lambda x: {"Low":1, "Medium":2, "High":3}[x])) # Concatenate risks

                            save_policy_data(
                                username=st.session_state["username"],
                                plan_type="Family", # Explicitly set as Family
                                is_renewal=(st.session_state["policy_status"] == "Renewal"), # Based on overall choice
                                age=main_member_data["age"], sex=main_member_data["sex"], bmi=main_member_data["bmi"],
                                children=main_member_data["children"], smoker=main_member_data["smoker"], region=main_member_data["region"],
                                high_bp=main_member_data["HighBP"], high_chol=main_member_data["HighChol"], chol_check=main_member_data["CholCheck"],
                                phys_activity=main_member_data["PhysActivity"], fruits=main_member_data["Fruits"], veggies=main_member_data["Veggies"],
                                hvy_alcohol_consump=main_member_data["HvyAlcoholConsump"], any_healthcare=main_member_data["AnyHealthcare"],
                                no_doc_bc_cost=main_member_data["NoDocbcCost"], gen_hlth=main_member_data["GenHlth"],
                                ment_hlth=main_member_data["MentHlth"], phys_hlth=main_member_data["PhysHlth"], diff_walk=main_member_data["DiffWalk"],
                                education=main_member_data["Education"], income=main_member_data["Income"], stroke=main_member_data["Stroke"],
                                heart_disease_attack=main_member_data["HeartDiseaseorAttack"],
                                predicted_expenses=main_member_data["predicted_expenses"],
                                predicted_risk=main_member_data["predicted_risk"]
                            )
                            st.success("Family policy summary saved successfully!")
                        else:
                            st.warning("No family data to save.")