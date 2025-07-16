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
import numpy as np

# clustering import
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import json
import plotly.io as pio

# Helper functions
def daily_line_altair(df):

    df = df[df["days_in_year"] < 365]
    # 1) Aggregate your raw counts
    day_counts = (
        df
        .groupby("days_in_year", as_index=False)["y"]
        .sum()
        .rename(columns={"y": "Contacts Made"})
    )
    # 2) Create a full index of days 1–364
    full = pd.DataFrame({"days_in_year": range(1, 365)})
    # 3) Left-merge and fill missing with 0
    merged = full.merge(day_counts, on="days_in_year", how="left")
    merged["Contacts Made"] = merged["Contacts Made"].fillna(0)

    # 4) Chart that merged DataFrame
    chart = (
        alt.Chart(merged)
        .mark_line(interpolate="linear", point=True)
        .encode(
            x=alt.X("days_in_year:Q", title="Day of Year", scale=alt.Scale(domain=[1, 364]),   
                    axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Contacts Made:Q", title="Contacts Made"),
            tooltip=["days_in_year", "Contacts Made"]
        )
        .properties(width="container", height=300,
                    title="Daily Contacts Made Over a Year")
    )
    return chart

def monthly_line_altair(df):
    # 1) Aggregate by month
    month_counts = (
        df
        .groupby("month", as_index=False)["y"]
        .sum()
        .rename(columns={"y": "Contacts Made"})
    )
    # 2) Ensure all 12 months appear
    full_months = pd.DataFrame({"month": list(range(1,13))})
    merged = (
        full_months
        .merge(month_counts, on="month", how="left")
        .fillna(0)
    )
    # 3) Map to month names & ordered categorical
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    merged["month_name"] = pd.Categorical(
        [names[m-1] for m in merged["month"]],
        categories=names,
        ordered=True
    )

    # 4a) base chart transforms
    base = alt.Chart(merged)

    # 4b) line + points layer
    line = base.mark_line(interpolate="linear", point=True).encode(
        x=alt.X("month_name:O", title="Month", sort=names, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Contacts Made:Q", title="Contacts Made"),
        tooltip=[
            alt.Tooltip("month_name:O", title="Month"),
            alt.Tooltip("Contacts Made:Q", title="Contacts Made")
        ]
    )

    # 4c) text‐label layer, nudged above each point
    labels = base.mark_text(
        dy=-10,              # move labels 10px above points
        fontSize=12,
        color="white"
    ).encode(
        x=alt.X("month_name:O", sort=names),
        y="Contacts Made:Q",
        text=alt.Text("Contacts Made:Q")
    )

    # 4d) combine and style
    chart = (line + labels).properties(
        width="container", height=300, title="Monthly Contacts Made"
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
        y=alt.Y("rate_pct:Q", title="Success Rate (%)",  scale=alt.Scale(domain=[0, 60])),
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


def contact_channel_pie(df, filter_col="y", filter_val=1):
    """
    Given a DataFrame `df`, filters for df[filter_col] == filter_val,
    then builds & returns a Plotly Pie chart of contact_cellular vs contact_telephone.
    """
    # 1) Filter to “wins”
    wins = df[df[filter_col] == filter_val]

    # 2) Sum up the two channels
    sums = wins[["contact_cellular", "contact_telephone"]].sum().astype(int)
    counts = [sums["contact_cellular"], sums["contact_telephone"]]

    # 3) Build the Pie
    labels = ["Cellular", "Telephone"]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        textinfo="label+value+percent",
        insidetextorientation="radial"
    ))
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        title_text="Wins per Contact Channel (Count & Proportion in %)",
        showlegend=False
    )
    return fig

def plot_loan_venn(df, filter_col="y", filter_val=1, container_class="venn-container"):
    wins=df[df['y']==1]
    
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
    ax.text(0.5, 0.5, f"No Loans: {no_loans}",
            ha="center", va="center", fontsize=12)
    ax.set_title("Wins per Loan Ownership")

    return fig


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

def plot_age_duration_heatmap(df, 
                             age_start=18, age_end=66, age_step=5,
                             dur_start=0, dur_end=1200, dur_step=60):
    """
    Returns an Altair heatmap of conversion rate (y) by 5-year age bins vs. duration bins.
    """
    heatmap_data = df.copy()
    # 1) Age bins
    age_bins   = list(range(age_start, age_end + age_step, age_step))
    age_labels = [f"{b}-{b+age_step-1}" for b in age_bins[:-1]]
    heatmap_data['age_bin'] = pd.cut(
        heatmap_data['age'],
        bins=age_bins,
        labels=age_labels,
        right=False
    )

    # 2) Duration bins
    dur_bins   = list(range(dur_start, dur_end + dur_step, dur_step))
    # dur_labels = [f"{b}-{b+dur_step-1}" for b in dur_bins[:-1]]
    dur_labels = [
        f"{int(b/60)}–{int((b + dur_step)/60)}"
        for b in dur_bins[:-1]
    ]
    heatmap_data['duration_bin'] = pd.cut(
        heatmap_data['duration'],
        bins=dur_bins,
        labels=dur_labels,
        right=False
    )

    # 3) Pivot + fill
    heatmap_df = (
        heatmap_data
          .pivot_table(
             index='age_bin',
             columns='duration_bin',
             values='y',
             aggfunc='mean'
          )
          .fillna(0)
    )

    # 4) Melt back to long form
    hm_long = heatmap_df.reset_index().melt(
        id_vars='age_bin',
        var_name='duration_bin',
        value_name='conversion_rate'
    )

    # 5) Altair heatmap
    chart = (
        alt.Chart(hm_long)
          .mark_rect()
          .encode(
            x=alt.X("duration_bin:N", title="Minutes", sort=dur_labels, axis=alt.Axis(labelAngle=0, labelAlign="center", labelFontSize=9)),
            y=alt.Y("age_bin:O",      title="Age Bin",      sort=age_labels[::-1]),
            color=alt.Color("conversion_rate:Q", 
                            title="Conversion Rate", 
                            scale=alt.Scale(scheme="blues")),
            tooltip=[
              alt.Tooltip("age_bin:O",        title="Age Bin"),
              alt.Tooltip("duration_bin:N",   title="Minutes"),
              alt.Tooltip("conversion_rate:Q",title="Conversion Rate", format=".1%")
            ]
          )
          .properties(
            width="container", height=400,
            title="Conversion Rate Heatmap (Age × Duration in Minutes)"
          )
    )
    return chart


def plot_loans_duration_heatmap(df, dur_start=0, dur_end=1200, dur_step=60):
    """
    Returns an Altair heatmap of conversion rate (y) by loan category vs. duration bins.
    """
    heatmap_data = df.copy()

    # 1) Duration bins
    dur_bins   = list(range(dur_start, dur_end + dur_step, dur_step))
    # dur_labels = [f"{b}-{b+dur_step-1}" for b in dur_bins[:-1]]
    dur_labels = [
        f"{int(b/60)}–{int((b + dur_step)/60)}"
        for b in dur_bins[:-1]
    ]
    heatmap_data['duration_bin'] = pd.cut(
        heatmap_data['duration'],
        bins=dur_bins,
        labels=dur_labels,
        right=False
    )

    # 2) Loan category
    conditions = [
      (heatmap_data["housing"] == 1) & (heatmap_data["loan"] == 1),  # both
      (heatmap_data["housing"] == 0) & (heatmap_data["loan"] == 0),  # none
      (heatmap_data["housing"] == 1) & (heatmap_data["loan"] == 0),  # housing only
      (heatmap_data["housing"] == 0) & (heatmap_data["loan"] == 1)   # personal only
    ]
    choices = ["both_loans","no_loans","housing_loans","personal_loans"]
    heatmap_data["loans?"] = np.select(conditions, choices, default="unknown")

    # 3) Pivot + fill
    heatmap_df = (
        heatmap_data
          .pivot_table(
             index='loans?',
             columns='duration_bin',
             values='y',
             aggfunc='mean'
          )
          .fillna(0)
    )

    # 4) Melt
    hm_long = heatmap_df.reset_index().melt(
        id_vars='loans?',
        var_name='duration_bin',
        value_name='conversion_rate'
    )

    # 5) Altair heatmap
    chart = (
        alt.Chart(hm_long)
          .mark_rect()
          .encode(
            x=alt.X("duration_bin:N", title="Minutes", sort=dur_labels, axis=alt.Axis(labelAngle=0, labelAlign="center", labelFontSize=9)),
            y=alt.Y("loans?:O",      title="Loan Type",     sort=choices),
            color=alt.Color("conversion_rate:Q", 
                            title="Conversion Rate", 
                            scale=alt.Scale(scheme="blues")),
            tooltip=[
              alt.Tooltip("loans?:O",        title="Loan Type"),
              alt.Tooltip("duration_bin:N",  title="Minutes"),
              alt.Tooltip("conversion_rate:Q",title="Conversion Rate", format=".1%")
            ]
          )
          .properties(
            width="container", height=400,
            title="Conversion Rate Heatmap (Loans × Duration in Minutes)"
          )
    )
    return chart


# def plot_hierarchical_clustering(df, method='ward', metric='euclidean'):
#     """
#     Builds a hierarchical clustering dendrogram using the specified features.
    
#     Parameters:
#     - df: DataFrame containing the data
#     - features: list of column names to use for clustering
#     - method: linkage method for hierarchical clustering (e.g., 'ward', 'single', 'complete')
#     - metric: distance metric for pdist (e.g., 'euclidean', 'cityblock')
    
#     Returns:
#     - fig: Plotly Figure containing the dendrogram
#     """

#     # 0) declare clusering features
#     features=['age', 'balance']
#     # features=['age', 'education', 'default', 'balance', 'y']

#     # 1) Extract feature subset and drop missing
#     data = df[features].dropna()

#     # 1A) Apply Scaling 
#     scaler = StandardScaler()
#     data.loc[:, 'balance'] = scaler.fit_transform(data[['balance']])
#     # data['balance'] = scaler.fit_transform(data[['balance']])
    
#     # 2) Compute pairwise distances
#     dist_matrix = pdist(data.values, metric=metric)
    
#     # 3) Perform hierarchical/agglomerative clustering
#     linkage_matrix = hierarchy.linkage(dist_matrix, method=method)
    
#     # 4) Create Plotly dendrogram (using the linkage)
#     fig = ff.create_dendrogram(
#         data.values,
#         orientation='right',
#         labels=data.index.astype(str),
#         linkagefun=lambda _: linkage_matrix
#     )
#     fig.update_layout(
#         width=800,
#         height=600,
#         title=f'Hierarchical Clustering Dendrogram ({method.capitalize()} linkage)'
#     )
    
#     return fig


def previous_donut(df, filter_col="poutcome", filter_val=1):
    """
    Given a DataFrame `df`, filters for df[filter_col] == filter_val,
    then builds & returns a Plotly Pie chart of contact_cellular vs contact_telephone.
    """
    # 1) Filter to “wins”
    wins = df[df[filter_col] == filter_val]

    # 2) Count 0 vs 1 in target_col
    counts = wins["y"].value_counts().sort_index()
    zero_ct = int(counts.get(0, 0))
    one_ct  = int(counts.get(1, 0))

    # 3) Build labels & values
    labels = ["Losses", "Wons"]
    values = [zero_ct, one_ct]

    # 4) Create the donut
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+value+percent",
        insidetextorientation="horizontal",
        textposition="inside",
        marker=dict(
            colors=["#EF553B", "#636EFA"],   # first slice blue, second red
            line=dict(width=1, color="white")
        ),
        textfont=dict(
            color=["white", "white"],    # both slices in white
            size=14
        )
    ))

    if filter_val==0:
        fig_title="Proportion of Campaign Wins when its Previous Campaign Fails"
    elif filter_val==0.5:
        fig_title="Proportion of Campaign Wins when its Previous Campaign Outcome is Inconclusive"
    elif filter_val==1:
        fig_title="Proportion of Campaign Wins when its Previous Campaign Succeeds"


    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        title_text=fig_title,
        showlegend=False
    )
    return fig





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
    if persona =="Marketing Manager":
        # st.write("You chose: ", persona)

        # --- Create our 2×2 grid ---
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        # plots for a salesperson
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. conversion heatmap 
        # 4. sales-based recommendations

        # 1. plot for sales overtime?
        # 1) Aggregate by month

        # Sales-based recommendation
        with row1_col1:
            st.title("Marketing-based Recommendations:")

            st.markdown(
            """
            1. Customers are more likely to subscribe on specific months (Mar, Aug, Nov, Dec).

            2. Most customers are around in their early 30s to late 30s.

            3. Conversion rate increases a lot when duration goes up, age doesn't play a huge factor.

            4. Customers with no loans are more likely to subscribe in when the call was not long.

            5. Most customers have cellular samples thus they may not have a lot of time --> need to develop strategies that can make them stay longer.

            6. Over 50% of previous successful case still subscribes when we call, focus on these customers. 
            """
        )

        # Works both Sales and Marketing
        with row1_col2:
            st.subheader("Campaign Trend Over Time")
            ts_tab, ms_tab = st.tabs(["Monthly Count","Monthly Success"])
            with ts_tab:
                # daily number of success over time plot
                st.altair_chart(monthly_line_altair(data), use_container_width=True)
                # st.altair_chart(daily_line_altair(data), use_container_width=True)
            with ms_tab:
                # monthly succeess rate
                st.altair_chart(monthly_success_altair(data), use_container_width=True)
    
        # Works both Sales and Marketing
        with row2_col1:
            st.subheader("Outcome by Channel & Loans")
            contact_tab, loan_tab = st.tabs(["Contact Channel","Loan Overlap"])
            with contact_tab:
                # Plot 3: contact type pie (Plotly)
                contact_fig= contact_channel_pie(data)    
                st.plotly_chart(contact_fig, use_container_width=True)
            with loan_tab:
                # Plot 4: loan Venn (matplotlib)
                venn_fig = plot_loan_venn(data)
                st.markdown('<div class="venn-container">', unsafe_allow_html=True)
                st.pyplot(venn_fig)  
                st.markdown('</div>', unsafe_allow_html=True)

        # Change this:

        # Clustering: 

        # Box (2,2): distributions & heatmaps (Plots 5, 6 & 7)
        with row2_col2:
            st.subheader("Distributions & Heatmaps")
            dist_tab, heat_tab, loan_heat_tab = st.tabs([
                "Age KDE","Age×Duration Heatmap","Loan×Duration Heatmap"
            ])

            with dist_tab:
                # Plot 5: age KDE (Altair)
                st.altair_chart(kde_age_distribution(data), use_container_width=True)

            with heat_tab:
                # Plot 6: conversion heatmap by age & duration (Altair)
                st.altair_chart(plot_age_duration_heatmap(data), use_container_width=True)

            with loan_heat_tab:
                # Plot 7: conversion heatmap by loan type & duration (Altair)
                st.altair_chart(plot_loans_duration_heatmap(data), use_container_width=True)


    elif persona =="Salesperson":
        # st.write("You chose: ", persona)

        # --- Create our 2×2 grid ---
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        # plots for a salesperson
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. conversion heatmap 
        # 4. sales-based recommendations

        # 1. plot for sales overtime?
        # 1) Aggregate by month

        # Sales-based recommendation
        with row1_col1:
            st.title("Sales-based Recommendations:")

            st.markdown(
            """
            1. If the client has past subscribed our product before, they are way more likely to subscribe again!

            2. Choose to contact the clients on either summer or around Christmas.

            3. Don't worry about clients who owed us! 40% of clients that has loans still subscibes to us!

            4. Over 50% of previous successful case still subscribes when we call, focus on these customers. 

            5. Call as long as possible (ideally over 9 minutes). Call duration is the most important factor determining the campaign outcome!

            6. Most customers have cellular samples thus they may not have a lot of time, you need to attract their interest quickly!
            """
        )

        # Works both Sales and Marketing
        with row1_col2:
            st.subheader("Campaign Trend Over Time")
            ts_tab, ms_tab = st.tabs(["Daily Count","Monthly Success"])
            with ts_tab:
                # daily number of success over time plot
                st.altair_chart(daily_line_altair(data), use_container_width=True)
            with ms_tab:
                # monthly succeess rate
                st.altair_chart(monthly_success_altair(data), use_container_width=True)
    
        # Works both Sales and Marketing
        with row2_col1:
            st.subheader("Outcome by Channel & Loans")
            contact_tab, loan_tab = st.tabs(["Contact Channel","Loan Overlap"])
            with contact_tab:
                # Plot 3: contact type pie (Plotly)
                contact_fig= contact_channel_pie(data)    
                st.plotly_chart(contact_fig, use_container_width=True)
            with loan_tab:
                # Plot 4: loan Venn (matplotlib)
                venn_fig = plot_loan_venn(data)
                st.markdown('<div class="venn-container">', unsafe_allow_html=True)
                st.pyplot(venn_fig)  
                st.markdown('</div>', unsafe_allow_html=True)

        # Works both Sales and Marketing
        # Box (2,2): distributions & heatmaps (Plots 5, 6 & 7)
        with row2_col2:
            st.subheader("Outcome Based on Past Campaign's Outcomes")
            no_past_tab, past_tab, inconclusive_tab= st.tabs([
                "No Past Campaign","Successful Past Campaign", "Inconclusive Past Campaign"
            ])


            # Previous vs conversion rate 


            with no_past_tab:
                # Plot 5: age KDE (Altair)
                # st.altair_chart(kde_age_distribution(data), use_container_width=True)
                st.plotly_chart(previous_donut(df=data, filter_val=0), use_container_width=True)

            with past_tab:
                # Plot 6: conversion heatmap by age & duration (Altair)
                # st.altair_chart(plot_age_duration_heatmap(data), use_container_width=True)
                st.plotly_chart(previous_donut(df=data, filter_val=1), use_container_width=True)

            with inconclusive_tab:
                st.plotly_chart(previous_donut(df=data, filter_val=0.5), use_container_width=True)

def clustering_page(data):
    st.header("TBA")



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
            options=["Home", "Deposit Subscription Prediction", "Interactive Dashboard", "Customer Segmentation", "Data Overview & Export", "Acknowledgements"],
            icons=["house", "bank", "bar-chart-line", "pie-chart-fill", "cloud-download", "award"],
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
    elif choice == "Customer Segmentation":
        clustering_page(data)
    elif choice == "Data Overview & Export":
        overview_page(raw_data, data)
    elif choice == "Acknowledgements":
        acknowledgement_page(data)

if __name__ == "__main__":
    main()
