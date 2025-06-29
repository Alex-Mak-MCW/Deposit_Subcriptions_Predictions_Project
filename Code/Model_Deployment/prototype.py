import streamlit as st
import pickle
import datetime
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- CACHED RESOURCE LOADING ---
@st.experimental_memo
def load_models():
    # Load Decision Tree
    with open('../../Model/DT_Model_Deploy.pkl', 'rb') as f:
        dt_pipeline = pickle.load(f)
        dt_model = dt_pipeline.named_steps['classifier']
    # Load Random Forest
    with open('../../Model/RF_Model_Deploy.pkl', 'rb') as f:
        rf_pipeline = pickle.load(f)
        rf_model = rf_pipeline.named_steps['classifier']
    # Load XGBoost
    with open('../../Model/XGB_Model_Deploy.pkl', 'rb') as f:
        xgb_pipeline = pickle.load(f)
        xgb_model = xgb_pipeline.named_steps['classifier']
    return {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

@st.experimental_singleton
def load_data():
    # Load processed data for dashboard
    url = (
        "https://raw.githubusercontent.com/"
        "Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/refs/heads/main/Data/processed_Input.csv"
    )
    return pd.read_csv(url)

# --- REUSABLE UTILS ---

def make_prediction(model, user_input):
    return model.predict([user_input])

# Define separate input forms for each model
def user_input_form_decision_tree():
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('"Tree-based model that makes decisions by splitting data recursively on feature values"')

    # 1) Build the pros/cons table
    dt_pros_cons_df = pd.DataFrame({
        "Strengths:": [
            "🔍  Highly Interpretable",
            "🧩  Can Handle Mixed Data",
            "🛡️  Robust to outliers"
        ],
        "Weaknesses:": [
            "⚠️  Prone to Overfitting",
            "📉  High variance",
            "🌪️  Instability"
        ]
    })
    dt_pros_cons_df.index = [1, 2, 3]
    # dt_pros_cons_df.index = [''] * len(dt_pros_cons_df)
    # 2) Display it at the top
    st.table(dt_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Fill in The Customer's Values:")


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
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('"Decision Tree Model, but Made Decisions with 100 Trees"')

    # 1) Build the pros/cons table
    rf_pros_cons_df = pd.DataFrame({
        "Pros": [
            "🌲  Less Likely to Overfit",
            "📊  Handles High-dimensional Data Well",
            "🛡️  Robust to Noise & Outliers"
        ],
        "Cons": [
            "🔍  Less Interpretable than Decision Tree",
            "💾  Higher Memory Usage",
            "🐢  Slower Predictions"
        ]
    })
    # rf_pros_cons_df.index = [''] * len(rf_pros_cons_df)
    rf_pros_cons_df.index = [1, 2, 3]
    # 2) Display it at the top
    st.table(rf_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Fill in The Customer's Values:")


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

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader('"A tree that Learns from its 100 Ancestors to Make Rules."')

    # 1) Build the pros/cons table
    xgb_pros_cons_df = pd.DataFrame({
        "Pros": [
            "⚡  Most Powerful",
            "🔧  Self Correcting & Tuning",
            "☁️  Handles Missing Values Natively"
        ],
        "Cons": [
            "👓  Least Interpretable",
            "⏳  Longer Training Ttimes",
            "🛠️  Harder to Optimize"
        ]
    })
    xgb_pros_cons_df.index = [''] * len(xgb_pros_cons_df)
    xgb_pros_cons_df.index = [1, 2, 3]
    # 2) Display it at the top
    st.table(xgb_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Fill in The Customer's Values:")


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
            st.image("Visualizations/success_icon.png", width=50)  # Use an icon for success
        with col2:
            st.write("### The Marketing Campaign will Succeed!")
    # Predict failure case:
    elif prediction[0] == 0:
        with col1:
            st.image("Visualizations/failure_icon.png", width=50)  # Use an icon for failure
        with col2:
            st.write("### The Marketing Campaign will Fail.")


# --- PAGE FUNCTIONS ---

def prediction_page(models):
    st.header("Predicting Term Deposit Subscription")
    st.markdown("---")
    # st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Choose an AI model to make predictions!")
    tabs = st.tabs(list(models.keys()))
    for tab, (name, model) in zip(tabs, models.items()):
        with tab:
            # st.subheader(f"{name} Model")
            # Dispatch to the right input form
            if name == 'Decision Tree':
                inputs = user_input_form_decision_tree()
            elif name == 'Random Forest':
                inputs = user_input_form_random_forest()
            else:
                inputs = user_input_form_xgboost()

            if st.button(f"Predict with {name}", key=name):
                try:
                    pred = make_prediction(model, inputs)
                    display_prediction(pred)
                except ValueError:
                    st.error("Please enter valid numeric values for all fields!")


def dashboard_page(data):
    st.header("Interactive Dashboard")
    st.subheader("Explore key metrics and visualizations")
    # Example KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Duration (s)", int(data['duration'].mean().round()))
    col2.metric("Avg Previous Days", int(data['pdays'].mean().round()))
    col3.metric("Success Rate", f"{round(data['y'].mean() * 100,2)}%")
    col4.metric("First Contact %", f"{round((data['previous']==0).mean()*100,2)}%")
    # Example bar chart
    fig = go.Figure()
    fig.add_bar(x=['Cellular','Telephone'],
                y=[data['contact_cellular'].sum(), data['contact_telephone'].sum()])
    fig.update_layout(title="Contacts by Channel", xaxis_title="Channel", yaxis_title="Count")
    st.plotly_chart(fig)

# --- MAIN APP ---

def main():
    # st.title("Bank Term Deposit App")
    with st.sidebar:
        # title
        st.title("Deposit Subscription Prediction Data Science Project")
        st.caption("v1.1.0 • Data updated: 2025-06-28") 

        # 1) Logo
        # st.image("Visualizations/title_icon_temp.gif", width=300)

        # 2) inline link
        # st.markdown(
        #     "[Source Code](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project) • "
        #     "[Contact Us (for any questions)! ](https://www.linkedin.com/in/alex-mak-824187247/)",
        #     unsafe_allow_html=True
        # )
        st.markdown("---")

        # project_list=['Term Deposit Subscription Prediction','Interactive Dashboard', 'Dataset Overview']


        # 3) Section headers + pickers
        # st.markdown("### 📁 Functionalities")
        # # project = st.selectbox("", project_list, key="project")
        # # choice=project

        choice = option_menu(
            menu_title=None,
            options=["Home", "Deposit Subscription Prediction", "Interactive Dashboard", "Data Overview", "Data Export", "Acknowledgements"],
            icons=["house", "bank", "bar-chart-line", "table", "download", "award"],
            menu_icon="app-indicator",
            default_index=0,
            orientation="vertical"
        )

        # print(choice)

        # --- Help & feedback ---
        with st.expander("❓ Help & Docs"):
            st.write("- [User Guide](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project)")
            st.write("- [Source Code](#)")
            st.write("- [Contact Us](https://www.linkedin.com/in/alex-mak-824187247/)")
        
        st.caption("© 2025 Alex Mak, All Rights Reserved")

    # choice = st.sidebar.radio("Go to", ["Prediction", "Dashboard"] )

    models = load_models()
    data   = load_data()

    if choice == "Deposit Subscription Prediction":
        prediction_page(models)
    else:
        dashboard_page(data)

if __name__ == "__main__":
    main()

# def main():
#     # — Sidebar as full nav panel —
#     with st.sidebar:
#         # 1) Logo
#         st.image("Visualizations/logo.png", width=150)
        
#         # 2) Inline links
#         st.markdown(
#             "[Source code](https://github.com/your/repo) • "
#             "[Evidently docs](https://docs.evidentlyai.com)",
#             unsafe_allow_html=True
#         )
#         st.markdown("---")
        
#         # 3) Section headers + pickers
#         st.markdown("### 📁 Select project")
#         project = st.selectbox("", project_list, key="project")
        
#         st.markdown("### 📅 Select period")
#         period = st.selectbox("", period_list, key="period")
        
#         st.markdown("### 📊 Select report")
#         report = st.selectbox("", report_list, key="report")
#         st.markdown("---")
        
#         # 4) Top-level navigation
#         nav = st.radio(
#             "Go to",
#             ["Prediction", "Dashboard"],
#             index=0,
#             key="nav",
#             label_visibility="collapsed"
#         )
    
#     # Load once
#     models = load_models()
#     data   = load_data()
    
#     # Dispatch based on sidebar nav
#     if st.session_state.nav == "Prediction":
#         prediction_page(models)
#     else:
#         dashboard_page(data)
