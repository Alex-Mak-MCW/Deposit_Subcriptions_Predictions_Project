# codebase for streamlit deployment
import streamlit as st
import pickle
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Load machine models built earlier
with open('../../Model/DT_Model_Deploy.pkl', 'rb') as file:
    decision_tree_pipeline = pickle.load(file)
    decision_tree_model = decision_tree_pipeline.named_steps['classifier']

with open('../../Model/RF_Model_Deploy.pkl', 'rb') as file:
    random_forest_pipeline = pickle.load(file)
    random_forest_model = random_forest_pipeline.named_steps['classifier']

with open('../../Model/XGB_Model_Deploy.pkl', 'rb') as file:
    xgboost_pipeline = pickle.load(file)
    xgboost_model = xgboost_pipeline.named_steps['classifier']

# Helper function to make predictions
def make_prediction(model, user_input):
    return model.predict([user_input])

# Define separate input forms for each model
def user_input_form_decision_tree():

    # Calculate the current day of the year for days_in_year
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # Assign inputs
    age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=0)
    balance = st.number_input("What is you bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=1)
    duration = st.slider("How long was the Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=2)
    campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=3)
    pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=4)
    poutcome = st.selectbox("What is the outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=5)  # Default value is 0
    days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=6)
    
    # input handling
    age = float(age)
    balance = float(balance)
    duration = float(duration)
    duration*=60
    campaign = float(campaign)
    days_in_year = float(days_in_year)

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    # return input values in array
    return [age, balance, duration, campaign, pdays, poutcome, days_in_year]

def user_input_form_random_forest():

    # Get current date
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # Assign inputs
    age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=10)
    balance = st.number_input("What is your bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=11)
    duration = st.slider("How long was the duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=12)
    campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=13)
    pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=14)
    poutcome = st.selectbox("What is the outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=15)  # Default value is 0
    days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=16)
    
    # input handling
    age = float(age)
    balance = float(balance)
    duration = float(duration)
    duration*=60
    campaign = float(campaign)
    days_in_year = float(days_in_year)

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    # return input values in array
    return [age, balance, duration, campaign, pdays, poutcome, days_in_year]

def user_input_form_xgboost():

    # Get the current date
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # Mapping dictionary for Yes/No to 1/0
    yes_no_mapping = {"Yes": 1, "No": 0}

    # input handling
    housing = yes_no_mapping[st.selectbox("Do you have any housing loans?", ["No", "Yes"], index=0, key=20)]
    loan = yes_no_mapping[st.selectbox("Do you have any personal loans?", ["No", "Yes"], index=0, key=21)]

    duration = st.slider("Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=22)
    pdays = st.slider("How many days ago was your last contacted by us?", min_value=0, max_value=1000, value=5, key=23)
    poutcome = st.selectbox("Outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=24)  # Default value is 0

    marital_married = yes_no_mapping[st.selectbox("Are you married?", ["No", "Yes"], index=0, key=25)]
    job_blue_collar = yes_no_mapping[st.selectbox("Do you work as a blue collar job?", ["No", "Yes"], index=0, key=26)]
    job_housemaid = yes_no_mapping[st.selectbox("Do you work as a housemaid?", ["No", "Yes"], index=0, key=27)]
    days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=28)
    
    # input handling
    duration = float(duration)
    duration*=60
    days_in_year = float(days_in_year)

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    # return input values in array
    return [housing, loan, duration, pdays, poutcome, marital_married, job_blue_collar, job_housemaid, days_in_year]

# Prediction function to handle success/failure messages
# def display_prediction(prediction):
#     if prediction[0] == 1:
#         msg = 'The marketing campaign will: \nSucceed! (Output=1)'
#         st.success(msg)
#     elif prediction[0] == 0:
#         msg = 'The marketing campaign will: \nFail! (Output=0)'
#         st.success(msg)

def display_prediction(prediction):
    col1, col2 = st.columns([0.1, 0.9])
    # print(prediction) # for testing

    # Predict success case:
    if prediction[0] == 1:
        with col1:
            st.image("success_icon.png", width=50)  # Use an icon for success
        with col2:
            st.write("### The marketing campaign will succeed!")
    # Predict failure case:
    elif prediction[0] == 0:
        with col1:
            st.image("failure_icon.png", width=50)  # Use an icon for failure
        with col2:
            st.write("### The marketing campaign will fail.")

# Draft of the appliation
st.title("Term Deposit Subscription Prediction Application")
st.subheader(" ")

# Create tabs for different model
tabs = st.tabs(["Decision Tree", "Random Forest", "Extreme Gradient Boosting Trees (XGBoost)"])

# In the decision tree model tab
with tabs[0]:
    # design & ask input
    st.header("Decision Tree Model")
    st.subheader("Please Enter the Following Information:")
    user_input = user_input_form_decision_tree()

    # When the prediction button clicked 
    if st.button("Predict with Decision Tree Model"):
        try:
            # print(user_input) # print input for testing
            prediction = decision_tree_model.predict([user_input])
            # print(prediction)
            display_prediction(prediction)
        except ValueError:
            st.error("Please enter valid numeric values for all fields!")

# In the random forest model tab
with tabs[1]:
    # design & ask input
    st.header("Random Forest Model")
    st.subheader("Please Enter the Following Information:")
    user_input = user_input_form_random_forest()

    # When the prediction button clicked 
    if st.button("Predict with Random Forest Model"):
        try:
            # print(user_input) # for testing
            prediction = random_forest_model.predict([user_input])
            # print(prediction) # for testing
            display_prediction(prediction)
        except ValueError:
            st.error("Please enter valid numeric values for all fields!")

# In the xgboost model tab
with tabs[2]:
    # design & ask input
    st.header("Extreme Gradient Boosting Trees Model")
    st.subheader("Please Enter the Following Information:")
    user_input = user_input_form_xgboost()

    # When the prediction button clicked 
    if st.button("Predict with XGBoost Model"):
        try:
            # print(user_input) # for testing
            # make prediction with model
            prediction = xgboost_model.predict([user_input])
            # print(prediction) # for testing
            display_prediction(prediction)
        except ValueError:
            st.error("Please enter valid numeric values for all fields!")
