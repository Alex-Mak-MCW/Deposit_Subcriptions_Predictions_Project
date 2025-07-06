import streamlit as st
import pickle
import datetime
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import io
import plotly.express as px
import altair as alt
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Bank Term Deposit App", 
    layout="wide"
)

st.markdown(
    """
    <style>
        /* make any st.metric container use the sidebar bg */
        [data-testid="stMetric"] {
            background-color: var(--sidebar-background) !important;
            padding: 1rem 1.5rem !important;
            border-radius: 0.75rem !important;
        }

        [data-testid="stMetric"] {
            background-color: #393939;
            text-align: center;
            padding: 15px 0;
        }

        [data-testid="stMetricLabel"] {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        [data-testid="stMetricDeltaIcon-Up"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }
        [data-testid="stMetricDeltaIcon-Down"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }

        div[data-testid="stPlotlyChart"] {
            background-color: var(--sidebar-background) !important;
            padding: 1rem !important;
            border-radius: 0.75rem !important;
        }
        .venn-container {
        background-color: var(--sidebar-background) !important;
        padding: 1rem !important;
        border-radius: 0.75rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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

def home_page():
    st.header("Welcome to the [Term Deposit Subscription Prediction App]!")
    st.markdown("---")
    home_html = """
        This app adopts data science and machine learning methodologies in the finance sector. With the following functionalities:
        <br><br>
        1. **Deposit Subscription Prediction**  

            Use our fine-tuned ML model to predict whether a client will subscribe!
        2. **Interactive Dashboard**  

            Uncover data-driven and hidden trends as well as insights!
        
        3. **Data Overview**  
        
            No time to dig through the data? Just read our summary here!
        
        4. **Data Export**  
        
            Download our preprocessed data as you wish! We won't ask for your tip. We promise :)
        <br><br>
        Hope you have fun playing around with it! If you get stuck or spot any bugs, let me know!

        — Alex :)
    """
    # st.markdown(home_html, unsafe_allow_html=True)
    # create two columns: left is text, right is image
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # render your HTML or Markdown
        st.markdown(home_html, unsafe_allow_html=True)

    with col2:
        # swap in your own image path
        st.image("Visualizations/title_icon_temp.gif",
                 caption="Me vibing when I am building this project",
                 use_column_width=True)

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
    st.subheader("Choose your persona & explore key metrics and visualizations: ")

    def kpi_indicator(label, value, suffix="", width=1):
        fig = go.Figure(go.Indicator(
            mode="number",
            value=value,                              # must be numeric
            title={"text": label, "font": {"size": 20}},
            number={
                "font": {"size": 60},
                "suffix": suffix                       # show the % sign here
            },
            domain={"x": [0, 1], "y": [0, 1]}
        ))
        fig.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            height=150
        )
        return fig

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    # checkbox selection to choose personas
    with k1:
        persona = st.selectbox("User Persona:", ["Salesperson", "Marketing Manager"])
        # Time Range Slider
        # selected_date = st.slider(
        #     "Time Range",
        #     min_value=datetime.date(2021, 1, 1),
        #     max_value=datetime.date(2021, 12, 31),
        #     value=(datetime.date(2021, 1, 1),datetime.date(2021, 12, 31))
        # )
        st.metric("Most Important Factor: ", 'Call Duration')
    # Display KPI 1 (conversion rate)
    with k2.container():
        st.markdown("<br>", unsafe_allow_html=True)
        k2_fig = kpi_indicator("Conversion Rate", round(data['y'].mean() * 100,2), suffix="%")
        st.plotly_chart(k2_fig, use_container_width=True)
        # k2_fig = kpi_indicator("Avg Duration (s)", int(data['duration'].mean().round()))
        # st.plotly_chart(k2_fig, use_container_width=True)
        
    # Display KPI 2 (Avg. Balance of Converters)
    with k3.container():
        st.markdown("<br>", unsafe_allow_html=True)
        if persona=='Marketing Manager':
            k3_fig = kpi_indicator("First-Time Conversion Rate", round(data[data['previous']==0]['y'].mean()* 100,2), suffix="%")
            st.plotly_chart(k3_fig, use_container_width=True)
        else:
            k5_fig = kpi_indicator("Avg. Duration of Success (mins)", data[data['y']==1]['duration'].mean()/60)
            st.plotly_chart(k5_fig, use_container_width=True)

    # Display KPI 3 (First call Conversion Rate)
    with k4.container():
        st.markdown("<br>", unsafe_allow_html=True)
        k4_fig = kpi_indicator("First Contact %", round((data['previous']==0).mean()*100,2), suffix="%")
        st.plotly_chart(k4_fig, use_container_width=True)

    # Display KPI 4 ()
    with k5.container():
        st.markdown("<br>", unsafe_allow_html=True)
        if persona=='Marketing Manager':
            k5_fig = kpi_indicator("Avg. Acct. Balance for Success", data[data['y']==1]['balance'].mean())
            st.plotly_chart(k5_fig, use_container_width=True)
        else:
            k5_fig = kpi_indicator("Avg Past Success Rate:", round(data[data['y']==1]['poutcome'].mean()*100,2))
            st.plotly_chart(k5_fig, use_container_width=True)


    st.markdown("---")

    # So default is salesperson
    if persona =="Salesperson":
        st.write("You chose: ", persona)

        # Example KPI cards
        # col1, col2, col3, col4 = st.columns(4)
        # col1, col2, col3, col4 = st.columns((1.5, 4, 2, 2))

        # plots for a salesperson
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. conversion heatmap 
        # 4. sales-based recommendations

        # 1. plot for sales overtime?
        # 1) Aggregate by month

        def daily_line_altair(df):

            df = df[df["days_in_year"] < 365]
            # 1) Aggregate your raw counts
            day_counts = (
                df
                .groupby("days_in_year", as_index=False)["y"]
                .sum()
                .rename(columns={"y": "count_of_y"})
            )
            # 2) Create a full index of days 1–364
            full = pd.DataFrame({"days_in_year": range(1, 365)})
            # 3) Left-merge and fill missing with 0
            merged = full.merge(day_counts, on="days_in_year", how="left")
            merged["count_of_y"] = merged["count_of_y"].fillna(0)

            # 4) Chart that merged DataFrame
            chart = (
                alt.Chart(merged)
                .mark_line(interpolate="linear", point=True)
                .encode(
                    x=alt.X("days_in_year:Q", title="Day of Year", scale=alt.Scale(domain=[1, 364]),   
                            axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("count_of_y:Q", title="Count of Y"),
                    tooltip=["days_in_year", "count_of_y"]
                )
                .properties(width="container", height=300,
                            title="Daily Count of Y (Days 1–364)")
            )
            return chart

        def monthly_success_altair(df):
            months = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

            # 1) Shared transforms
            base = (
                alt.Chart(df)
                .transform_calculate(
                    month_name=f"['" + "','".join(months) + f"'][datum.month - 1]"
                )
                .transform_aggregate(
                    rate="mean(y)",
                    groupby=["month_name"]
                )
                .transform_calculate(
                    rate_pct="datum.rate * 100"
                )
            )

            # 2) Line + points
            line = base.mark_line(interpolate="linear", point=True).encode(
                x=alt.X(
                    "month_name:O",
                    sort=months,                    # enforce Jan→Dec
                    title="Month",
                    axis=alt.Axis(labelAngle=360)
                ),
                y=alt.Y("rate_pct:Q", title="Success Rate (%)"),
                tooltip=[
                    alt.Tooltip("month_name:O", title="Month"),
                    alt.Tooltip("rate_pct:Q", title="Success Rate", format=".1f")
                ]
            )

            # 3) Point labels
            labels = base.mark_text(
                dy=-10,               # nudge text above each point
                fontSize=12,
                color="white"
            ).encode(
                x=alt.X("month_name:O", sort=months),
                y="rate_pct:Q",
                text=alt.Text("rate_pct:Q", format=".1f")
            )

            # 4) Combine layers
            chart = (line + labels).properties(
                width="container",
                height=300,
                title="Monthly Success Rate"
            ).configure_title(fontSize=18, anchor="start")

            return chart




        choice = st.selectbox(
            "Which chart would you like to see?",
            ["Success Count", "Monthly Success Rate"]
        )

        st.markdown("---")

        # 2) conditional display
        if choice == "Success Count":
            chart = daily_line_altair(data)
            st.altair_chart(chart, use_container_width=True)
        elif choice =="Monthly Success Rate":  # Monthly Success Rate
            chart = monthly_success_altair(data)
            st.altair_chart(chart, use_container_width=True)

        #---------------------------------------------
        # 2. plots for outcome per variable (w/select box)

        wins=data[data['y']==1]


        # FOR CONTACTS
        contact_counts_np = wins[["contact_cellular","contact_telephone"]].sum()
        # convert to Python ints
        contact_counts = [int(contact_counts_np["contact_cellular"]), int(contact_counts_np["contact_telephone"])]
        # or equivalently:
        # counts = [counts_np["contact_cellular"].item(), counts_np["contact_telephone"].item()]

        labels = ["Cellular","Telephone"]

        contact_fig = go.Figure(go.Pie(
            labels=labels,
            values=contact_counts,
            textinfo="label+value+percent",     # show label, raw value, and %
            insidetextorientation="radial"
        ))
        contact_fig.update_traces(marker=dict(line=dict(width=1, color="white")))
        contact_fig.update_layout(title_text="Contact Channel Distribution", showlegend=False)
        st.plotly_chart(contact_fig, use_container_width=True)

        

        # FOR loans
        # ── 1) Compute each subset ──
        wins["both loans"] = ((wins["housing"] == 1) & (wins["loan"] == 1)).astype(int)
        wins["no loans"]   = ((wins["housing"] == 0) & (wins["loan"] == 0)).astype(int)

        both       = int(wins["both loans"].sum())
        housing    = int(wins["housing"].sum()) - both      # housing only
        personal   = int(wins["loan"].sum())    - both      # personal only
        no_loans   = int(wins["no loans"].sum())            # neither

        # ── 2) Plot Venn ──
        fig, ax = plt.subplots()
        v = venn2(
            subsets=(housing, personal, both),
            set_labels=("Housing Loan", "Personal Loan"),
            ax=ax
        )
        # color the regions
        v.get_patch_by_id('10').set_color('#636EFA')
        v.get_patch_by_id('01').set_color('#EF553B')
        v.get_patch_by_id('11').set_color('#00CC96')
        # annotate “no loans” below
        ax.text(0.5, -0.1, f"No Loans: {no_loans}",
                ha="center", va="center", fontsize=12)
        ax.set_title("Loan Ownership Overlap")

        # wrap in our .venn-container
        st.markdown('<div class="venn-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)


        # # FOR CONTACTS
        # contact_counts_np = wins[["contact_cellular","contact_telephone"]].sum()
        # # convert to Python ints
        # contact_counts = [int(contact_counts_np["contact_cellular"]), int(contact_counts_np["contact_telephone"])]
        # # or equivalently:
        # # counts = [counts_np["contact_cellular"].item(), counts_np["contact_telephone"].item()]

        # labels = ["Cellular","Telephone"]

        # contact_fig = go.Figure(go.Pie(
        #     labels=labels,
        #     values=contact_counts,
        #     textinfo="label+value+percent",     # show label, raw value, and %
        #     insidetextorientation="radial"
        # ))
        # contact_fig.update_traces(marker=dict(line=dict(width=1, color="white")))
        # contact_fig.update_layout(title_text="Contact Channel Distribution", showlegend=False)
        # st.plotly_chart(contact_fig, use_container_width=True)


        # age plot
        def kde_age_distribution(df, field="age", filter_col="y", filter_val=1, bandwidth=5):
            """
            Render a 1D KDE plot (density estimate) of `field` for rows where df[filter_col] == filter_val.
            """
            # Filter wins
            wins = df[df[filter_col] == filter_val]
            
            # Build KDE via Vega-Lite transform_density
            chart = (
                alt.Chart(wins)
                .transform_density(
                    field,
                    as_=[field, "density"],
                    extent=[18, 100],    # compute KDE from 18 → 100
                    bandwidth=bandwidth
                )
                .mark_area(opacity=0.5, interpolate="monotone")
                .encode(
                    x=alt.X(
                        f"{field}:Q",
                        title="Age",
                        scale=alt.Scale(domain=[18, 100]),   # ⟵ clamp the axis here
                        axis=alt.Axis(tickMinStep=5)
                    ),
                    y=alt.Y("density:Q", title="Density"),
                    tooltip=[
                        alt.Tooltip(field, title="Age"),
                        alt.Tooltip("density:Q", title="Density", format=".3f")
                    ]
                )
                .properties(
                    width="container",
                    height=300,
                    title="Age Distribution of Wins (KDE)"
                )
                .configure_title(fontSize=18, anchor="start")
                .configure_axis(labelFontSize=12, titleFontSize=14)
            )
            return chart

        # Example usage in Streamlit
        st.header("KDE of Age for Winning Clients")
        # Assuming `data` is your DataFrame with 'age' and 'y' columns
        chart = kde_age_distribution(data, bandwidth=5)
        st.altair_chart(chart, use_container_width=True)


    # conversion heatmap --> which pairs





    elif persona =="Marketing Manager":
        st.write("You chose: ", persona)

        # plots for a marketing manager
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. clusering plots for customers.
        # 4. table (df)
        # 5. marketing-based recommendations


def overview_page(data, preprocessed):
    
    st.header("Data Overview & Export Page")
    st.markdown("Choose which dataset you’d like to download: MORE DATA TYPE WILL BE PROVIDED")
    
    # two equal-width columns
    col1, col2 = st.columns(2, gap="small")
    
    # Raw data box
    with col1.container():
        st.subheader("Raw Data")
        # st.write("This is the original, unprocessed dataset as ingested.")
        # CSV bytes
        data_csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Raw Data (.csv)",
            data=data_csv,
            file_name="raw_data.csv",
            mime="text/csv"
        )
        # XLSX bytes
        buffer = io.BytesIO()
        data.to_excel(buffer, index=False, sheet_name="raw", engine="openpyxl")
        buffer.seek(0)
        st.download_button(
            label="📥 Download Raw Data in Excel (.xlsx)",
            data=buffer,
            file_name="raw_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")
        st.write("This is the original, unprocessed dataset.")
    
    # Preprocessed data box
    with col2.container():
        st.subheader("Preprocessed Data")
        # CSV bytes
        pre_csv = preprocessed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Processed Data (.csv)",
            data=pre_csv,
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )
        # XLSX bytes
        buffer2 = io.BytesIO()
        preprocessed.to_excel(buffer2, index=False, sheet_name="preprocessed", engine="openpyxl")
        buffer2.seek(0)
        st.download_button(
            label="📥 Download Processed Data in Excel (.xlsx)",
            data=buffer2,
            file_name="preprocessed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")
        st.write("This version has been cleaned and feature-engineered. Including:")
        # Numbered list of steps
        st.markdown(
            """
            1. Handled missing and duplicated data.
            2. Data transfomration through different imputation, encoding, and other data techniques.
            3. Anomaly detection & removal.
            4. Feature engineering by modifying existing features and adding new features.
            """
        )
    

def acknowledgement_page(data):
    # st.header("ACKNOWLEDGEMENT PAGE")
    # st.subheader("TBA")
    st.header("Acknowledgements")
    st.markdown("---")
    ack_html = """
    First of all, this entire application comes from a graudate data science course project where my teammates **Zheng En Than** and **Emily Au**. We cleaned the collected data, 
    performed exploratory data analysis, and developed the machine learning models. I sincerely thank them for their effort and hard work. 
    I would also like to thank my course instructor **Dr. Jay Newby** for his guidance and mentorship.
    <br><br>
    Additionally, I want to acknowledge **Sérgio Moro**, **P. Cortez**, and **P. Rita** for sharing the UCI ML Bank Telemarketing Dataset which is the fundament backbomne of this project.
    <br><br>
    Last but not least, shout out to the user test group [TEMP NAMES]. Their opinions and feedback on this project should be recognized.
    """
    st.markdown(ack_html, unsafe_allow_html=True)

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
            options=["Home", "Deposit Subscription Prediction", "Interactive Dashboard", "Data Overview & Export", "Acknowledgements"],
            icons=["house", "bank", "bar-chart-line", "table", "award"],
            menu_icon="app-indicator",
            default_index=0,
            orientation="vertical"
        )

        # print(choice)

        # --- Help & feedback ---
        with st.expander("❓ Help & Docs"):
            st.write("- [User Guide](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project)")
            st.write("- [Source Code](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/tree/main/Code)")
            st.write("- [Contact Us](https://www.linkedin.com/in/alex-mak-824187247/)")
        
        st.caption("© 2025 Alex Mak, All Rights Reserved")

    # choice = st.sidebar.radio("Go to", ["Prediction", "Dashboard"] )

    models = load_models()
    data   = load_data()

    raw_data=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/refs/heads/main/Data/input.csv")

    # url = (
    #     "https://raw.githubusercontent.com/"
    #     "Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/refs/heads/main/Data/processed_Input.csv"
    # )
    # return pd.read_csv(url)


    if choice == "Home":
        home_page()
    elif choice == "Deposit Subscription Prediction":
        prediction_page(models)
    elif choice == "Interactive Dashboard":
        dashboard_page(data)
    elif choice == "Data Overview & Export":
        overview_page(raw_data, data)
    elif choice == "Acknowledgements":
        acknowledgement_page(data)

if __name__ == "__main__":
    main()
