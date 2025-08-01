#  Import packages

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
import seaborn as sns

# clustering import
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import json
import plotly.io as pio
# from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import hdbscan
from shap.plots._waterfall import waterfall_legacy
import streamlit.components.v1 as components

import base64

# -----------------------------------------------


# Helper functions

## Plot functions---------------------------------
## -----------------------------------------------
## -----------------------------------------------

# function that plots daily line graph
def daily_line_altair(df):
    # limit to be within 365 (omit day 366)
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

# plot monthly trend_line graph
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

# plot monthly success rate chart
def monthly_success_altair(df):
    months = ["Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"]

    # 1) Perform transformations
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

    # 3) Data labels
    labels = base.mark_text(
        dy=-10,               # nudge text above each point
        fontSize=12,
        color="white"
    ).encode(
        x=alt.X("month_name:O", sort=months),
        y="rate_pct:Q",
        text=alt.Text("rate_pct:Q", format=".1f")
    )

    # 4) Combine plot & data labels
    chart = (line + labels).properties(
        width="container",
        height=300,
        title="Monthly Success Rate"
    ).configure_title(fontSize=18, anchor="start")

    return chart

# Plot pie chart for contact channel
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

# plot venn diagram for success cases over loan types
def plot_loan_venn(df, filter_col="y", filter_val=1, container_class="venn-container"):
    
    # isolate success cases
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


# Function that displays KDE plot
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

# Function that plot the plot x age duration heatmap
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

# plot the loan type X duration heatmap
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

# plot the donut chart for proportion of each previous outcome 
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

## Clustering-related Functions
## -----------------------------------------------
## -----------------------------------------------

# Function to perform HDBScan clustering algorithm
def auto_hdbscan(X, min_size=10):
    """HDBSCAN clusters automatically—no k needed (noise = -1)."""
    return hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(X)

# Function to show feature example and descriptions table
def show_example_table(data, selected_cols):

    st.header("Feature Descriptions & Examples:")

    # 1) define your master mappings once:
    DESC = {
        "age":              "age (whole number)",
        "education":        "education level (1,2,3…)",
        "default":          "previous default? (1=yes,0=no)",
        "balance":          "bank balance (decimal)",
        "contact_telephone":"contact channel (1=telephone,0=cellular)",
        "housing":          "housing loan? (1=yes,0=no)",
        "loan":             "personal loan? (1=yes,0=no)",
        "day":              "day of month of last campaign (1–31)",
        "month":            "month of year of last campaign (1–12)",
        "duration":         "call duration in seconds",
        "campaign":         "times contacted this campaign (incl. this)",
        "pdays":            "days since last campaign (-1=first time)",
        "previous":         "times contacted before this campaign",
        "poutcome":         "outcome of previous campaign (failure/nonexistent/success)",
        "days_in_year":     "current day of year (1–366)",
    }
    EX = {
        "age":              "42",
        "education":        "2",
        "default":          "0",
        "balance":          "10000.00",
        "contact_telephone":"1",
        "housing":          "0",
        "loan":             "1",
        "day":              "15",
        "month":            "7",
        "duration":         "900",
        "campaign":         "3",
        "pdays":            "5",
        "previous":         "2",
        "poutcome":         "success",
        "days_in_year":     "212",
    }

    # 2) Build a DataFrame with exactly |selected_cols| columns
    feature_info = pd.DataFrame(
        index=["Description", "Example"],
        columns=selected_cols,
        data=""   # placeholder
    )

    # 3) Fill each row by looking up in the dicts
    feature_info.loc["Description"] = [DESC.get(f, "") for f in selected_cols]
    feature_info.loc["Example"]     = [EX.get(f,  "") for f in selected_cols]

    # 4) Transpose & reformat
    fi = feature_info.T.reset_index().rename(columns={"index": "Feature"})
    fi.index = range(1, len(fi) + 1)

    # 5) Display
    st.table(fi)



#  Function that computes and plots raw Cluster Feature Means
def show_cluster_feature_means_raw(data, selected_cols):

    # ─── Cluster Means & Δ-Means Tables ───────────────────────────────
    st.header("Cluster Feature Means Table")
    cluster_means = data.groupby("Cluster")[selected_cols].mean().round(2)
    overall_mean  = data[selected_cols].mean()
    delta_means   = (cluster_means.subtract(overall_mean, axis=1)).round(2)

    # rename index: -1 → "Noise", others → "Index X"
    def make_label(idx):
        return "Outliers" if idx == -1 else f"Customer Group {1+idx}"
    cluster_means.index = cluster_means.index.map(make_label)
    # now move "Outliers" to the end
    if "Outliers" in cluster_means.index:
        # build a new index order: everything except Outliers, then Outliers
        new_order = [lab for lab in cluster_means.index if lab != "Outliers"] + ["Outliers"]
        cluster_means = cluster_means.loc[new_order]

    delta_means.index   = delta_means.index.map(make_label)
    if "Outliers" in delta_means.index:
        # build a new index order: everything except Outliers, then Outliers
        new_order = [lab for lab in delta_means.index if lab != "Outliers"] + ["Outliers"]
        delta_means = delta_means.loc[new_order]

    styled_means = (
        cluster_means
        .style
        .background_gradient(cmap="vlag")
        .format("{:.2f}")
    )
    st.dataframe(styled_means)

    st.write("Δ-means (Customer Group Mean - Overall Mean):")
    styled_delta = (
        delta_means
        .style
        .background_gradient(cmap="vlag")
        .format("{:.2f}")
    )
    st.dataframe(styled_delta)



# Function that build Violin Plots on Raw Data
def plot_violin_top_features_raw(data, selected_cols, top_n=3):
    # st.subheader(f"2. Violin Plots (Original Scale) for Top {top_n} Features (TBA)")

    # 1) Figure out the top-n features by variance of raw cluster means
    cluster_means = data.groupby("Cluster")[selected_cols].mean()
    top_feats = cluster_means.var().sort_values(ascending=False).index[:top_n].tolist()

    # Codebase for printing the violin plots

    # # 2) Build a label mapping: -1 → Noise, else → Cluster {i}
    # unique_idxs = sorted(data["Cluster"].unique())
    # label_map = {idx: ("Noise" if idx == -1 else f"Cluster {1+idx}") for idx in unique_idxs}

    # # 3) Copy data and add a human-readable cluster column
    # df = data.copy()
    # df["Cluster_label"] = df["Cluster"].map(label_map)

    # # 4) Create columns for side-by-side plots
    # cols = st.columns(len(top_feats))
    # for i, feat in enumerate(top_feats):
    #     with cols[i]:
    #         fig, ax = plt.subplots()
    #         sns.violinplot(
    #             x="Cluster_label",
    #             y=feat,
    #             data=df,
    #             inner="quartile",
    #             order=[label_map[idx] for idx in unique_idxs],  # preserve ordering
    #             ax=ax
    #         )
    #         ax.set_title(f"{feat} distribution by cluster")
    #         ax.set_xlabel("")  # optional: remove repeated x-axis labels
    #         ax.tick_params(axis='x', rotation=45)
    #         st.pyplot(fig)
    
    return top_feats


# Function that plots Tree-Based Importance to show importance of each factor
def plot_tree_feature_importance(data, X_scaled, selected_cols, top_n=5):
    st.header("Important Factors That Helps Building the Customer Groups")

    # build tabs for users to traverse
    cluster_labels = sorted(data["Cluster"].unique())
    tab_labels = [
        ("Outliers" if cl == -1 else f"Customer Group {cl+1}")
        for cl in cluster_labels
    ]
    tab_labels.append(tab_labels.pop(0))


    tabs = st.tabs(tab_labels)
    
    # display the plot under each tab
    rf_models = {}
    for tab, cl in zip(tabs, cluster_labels):
        with tab:
            # Create binary target: 1 if in this cluster, 0 otherwise
            y = (data["Cluster"] == cl).astype(int)
            
            # Train surrogate
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            rf_models[cl] = rf
            
            # Get top_n importances
            imps = pd.Series(rf.feature_importances_, index=selected_cols)
            imps = imps.nlargest(top_n)
            
            # Plot
            fig, ax = plt.subplots()
            imps.plot.bar(ax=ax)
            
            # ─── annotate each bar ───
            for bar in ax.patches:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # x: center of bar
                    height,                             # y: top of bar
                    f"{height:.2f}",                    # label text
                    ha="center", va="bottom",           # align
                    fontsize=10                         # adjust if you like
                )
            
            # plot setup
            ax.set_title(f"Top {top_n} Features for {'Outliers' if cl==-1 else f'Customer Group {cl+1}'}")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance Score")
            ax.set_xticklabels(imps.index, rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)

    
    return rf_models

# Plot SHAP Explanation for Custom Point (via Top-Feature Sliders)
def show_shap_explanation_custom(
    rf_model,
    scaler,
    data,
    selected_cols,
    top_n: int = 5
):
    """
    Renders sliders for the top_n most important features, predicts the cluster
    for the custom point via rf_model, and shows a SHAP waterfall plot explaining it.
    """
    st.subheader("SHAP Explanation for Added Custom Customer")

    # 1) Identify top-n features by RF importance
    importances = pd.Series(rf_model.feature_importances_, index=selected_cols)
    top_feats = importances.nlargest(top_n).index.tolist()

    # 2) Get global means for defaults
    global_means = data[selected_cols].mean()

    # 3) Build sliders
    st.write(f"Adjust values for top {top_n} features:")
    raw_vals = {}
    for feat in top_feats:
        lo = data[feat].min()
        hi = data[feat].max()
        mean = global_means[feat]

        if feat == "balance":
            # continuous slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=float(lo),
                max_value=float(hi),
                value=float(mean),
                step=0.01,
                format="%.2f",
                key=f"slider_{feat}"
            )
        else:
            # **integer** slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=int(lo),
                max_value=int(hi),
                value=int(round(mean)),
                step=1,
                format="%d",              # <-- force integer formatting
                key=f"slider_{feat}"
            )

    # 4) Build raw point
    raw_point  = np.array([ raw_vals.get(f, global_means[f]) 
                            for f in selected_cols ]).reshape(1, -1)
    scaled_pt  = scaler.transform(raw_point)

    # # 5) Predict cluster
    pred = rf_model.predict(scaled_pt)[0]
    label = "Outliers" if pred == -1 else f"Customer Group{pred}"
    proba = rf_model.predict_proba(scaled_pt)[0][pred]
    st.write(f"**Predicted Customer Group: ** {label} (p={proba:.2f})")

    # Build a single‐output explainer for the “pred” class probability:
    explainer = shap.Explainer(
        lambda d: rf_model.predict_proba(d)[:, pred], 
        data[selected_cols]    # or X you trained on
    )

    # Compute the Explanation object (shape: 1×n_features)
    full_exp = explainer(scaled_pt)

    # Extract the first (and only) explanation
    single_exp = full_exp[0]

    # Now plot exactly one waterfall
    shap.initjs()
    # draw into an Axes
    ax = shap.plots.waterfall(single_exp, show=False)

    # extract the parent Figure
    fig = ax.figure

    # render that
    st.pyplot(fig)


# Function that plots LIME Explanation for Custom Point (via Top-Feature Sliders)
def show_lime_explanation_custom(
    rf_model,
    scaler,
    data: pd.DataFrame,
    selected_cols: list,
    top_n: int = 5
):
    st.subheader("LIME Explanation for Added Custom Customer")

    # 1) Identify top-n features by RF importance
    importances = pd.Series(rf_model.feature_importances_, index=selected_cols)
    top_feats = importances.nlargest(top_n).index.tolist()

    # 2) Get global means for defaults
    global_means = data[selected_cols].mean()

    # 3) Sliders
    st.write(f"Adjust values for top {top_n} features:")
    raw_vals = {}
    for feat in top_feats:
        lo = data[feat].min()
        hi = data[feat].max()
        mean = global_means[feat]

        if feat == "balance":
            # continuous slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=float(lo),
                max_value=float(hi),
                value=float(mean),
                step=0.01,
                format="%.2f",
                key=f"slider_{feat}"
            )
        else:
            # **integer** slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=int(lo),
                max_value=int(hi),
                value=int(round(mean)),
                step=1,
                format="%d",              # <-- force integer formatting
                key=f"slider_{feat}"
            )

    # 4) Build and scale the point
    raw_point = np.array([raw_vals.get(f, global_means[f]) 
                          for f in selected_cols]).reshape(1, -1)
    scaled_pt = scaler.transform(raw_point)

    # 5) Predict label & proba
    pred_label = rf_model.predict(scaled_pt)[0]
    # get proba array in the model's class order:
    probas     = rf_model.predict_proba(scaled_pt)[0]

    # 6) Align class_names with rf_model.classes_
    sk_classes = list(rf_model.classes_)  # e.g. [-1, 0, 2]
    class_names = [
        "Outliers" if cid == -1 else f"Group {cid+1}"
        for cid in sk_classes
    ]
    # find the index of our predicted label in that list
    pred_index = sk_classes.index(pred_label)

    # 7) Show textual prediction
    st.write(f"**Predicted Customer Group:** {class_names[pred_index]} (probability = {probas[pred_index]*100:.0f}%)")
    st.markdown("---")

    # 8) Build LIME explainer & explanation
    explainer = LimeTabularExplainer(
        training_data      = data[selected_cols].values,
        feature_names      = selected_cols,
        class_names        = class_names,       # aligned now
        discretize_continuous=True
    )
    exp = explainer.explain_instance(
        raw_point[0],
        lambda x: rf_model.predict_proba(scaler.transform(x)),
        labels=(pred_index,),
        num_features=top_n
    )

    # 9) Render LIME’s HTML
    html = exp.as_html()
    wrapper = f"""
    <div style="
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        display: flex;
        flex-wrap: wrap;
        gap: 50px;
        align-items: flex-start;
    ">
      {html}
    </div>
    """
    components.html(wrapper, height=500, scrolling=True)

# Function that plots 3D Scatter on Raw
def plot_3d_clusters_raw(data, selected_cols, top_features):
    st.header("3D Cluster Visualization")

    # 1) show number of clusters
    n_clusters = data["Cluster"].nunique()
    st.markdown(f"**Number of Groups:** {n_clusters}")

    # 2) show the counts table
    counts = data["Cluster"].value_counts().sort_index()
    counts_df = (
        counts
        .rename_axis("cluster_label")
        .reset_index(name="Count")
        .set_index("cluster_label")
    )
    # build human labels & ordered list
    uniques = list(counts_df.index)
    label_map = {}
    ordered_labels = []

    # 4) first handle all non–Outlier clusters
    for i, cl in enumerate([c for c in uniques if c != -1]):
        lbl = f"Customer Group {i+1}"
        label_map[cl] = lbl
        ordered_labels.append(lbl)

    # 5) then, if you have outliers, put them last
    if -1 in uniques:
        label_map[-1] = "Outliers"
        ordered_labels.append("Outliers")

    # 6) rename your table’s index
    counts_df.index = [label_map[x] for x in counts_df.index]

    # 7) reorder rows in the display to match ordered_labels
    counts_df = counts_df.reindex(ordered_labels)

    # 8) styling output
    styled = (
        counts_df.style
        .set_caption("**Decision-Tree: Pros & Cons**")
        .set_table_styles([
            # Header styling
            {
                "selector": "thead th",
                "props": [
                ("background-color", "#4B8BBE"),
                ("color", "white"),
                ("font-size", "1.3em"),       # larger header text
                ("text-align", "center"),     # center header
                ("padding", "0.6em")
                ]
            },
            # Body cells: font size and center
            {
                "selector": "tbody td",
                "props": [
                ("font-size", "1.1em"),       # bump up body text
                ("text-align", "center"),     # center cell text
                ("padding", "0.5em")
                ]
            },
            # Zebra-striping
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f9f9f9")]
            }
        ])
    )

    st.dataframe(styled)

    # # get the top 3 features from all the top features got back in the violin plot

    # 9) add a categorical column for plotting
    df = data.copy()
    df["Cluster_label"] = df["Cluster"].map(label_map)

    # 10) pick top 3 features
    top3 = top_features[:3]

    # 11) pick a discrete palette & map each label to a color
    palette = px.colors.qualitative.Plotly  # has ~10 distinct colors
    color_map = {
        lbl: palette[i % len(palette)]
        for i, lbl in enumerate(ordered_labels)
    }

    # 12) scatter_3d with discrete legend
    fig3d = px.scatter_3d(
        df,
        x=top3[0], y=top3[1], z=top3[2],
        color="Cluster_label",
        category_orders={"Cluster_label": ordered_labels},
        color_discrete_map=color_map,
        title=f"3D view of Customer Groups & Outliers ",
        width=700, height=500
    )
    st.plotly_chart(fig3d)

# XAI-related Functions
#-----------------------------------------------
#-----------------------------------------------
# @st.cache(allow_output_mutation=True)
@st.cache(
    allow_output_mutation=True,
    hash_funcs={
        pd.DataFrame: lambda _: None,  # DataFrame unhashable → ignore it
        dict:        lambda _: None,   # same for dict/list
        bytearray:   lambda _: None,
    }
)

# Function that load XAI (LIME & SHAP) explainers
def load_explainers(model, df: pd.DataFrame, feature_names: tuple):
    """
    Builds two SHAP explainers (for P(Yes) and P(No)) + one LIME explainer
    all on exactly the same feature set.
    """
    X = df.loc[:, list(feature_names)]
    # SHAP: single‐output explainer for P(Yes)
    shap_explainer = shap.Explainer(
        lambda data: model.predict_proba(data)[:, 1],
        X
    )
    # LIME: full classifier explainer (we’ll ask for label=1 later)
    lime_explainer = LimeTabularExplainer(
        X.values,
        feature_names   = list(feature_names),
        class_names     = ['Lose','Win'],
        discretize_continuous=True
    )
    return shap_explainer, lime_explainer

# Function that displays the explanations
def show_explanations(model, inputs, shap_explainer, lime_explainer, max_lime_features: int = 10):
    # ─── normalize inputs to 1×n_features DataFrame ───
    if isinstance(inputs, dict):
        X = pd.DataFrame([inputs])
    elif isinstance(inputs, (list, np.ndarray)):
        arr = np.array(inputs).reshape(1, -1)
        cols = lime_explainer.feature_names
        X = pd.DataFrame(arr, columns=cols)
    else:
        X = inputs.copy()
    assert X.shape[0] == 1, "Need exactly one row of inputs"

    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Through Explainable AI (XAI):")


    # ─── Display LIME output for label=1 (“Yes”) ───
    st.markdown("**1. LIME: Made a local model for your input to highligjt which feature matters the most!**")
    lime_exp = lime_explainer.explain_instance(
        X.values.flatten(),
        model.predict_proba,
        labels=(1,),
        # labels=(1,0),
        num_features=min(max_lime_features, X.shape[1])
    )

    lime_html = lime_exp.as_html()

    # wrap it in a white box with some padding & rounded corners
    wrapper = """
    <div style="
        background-color: white;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    ">
        {inner}
    </div>
    """.format(inner=lime_html)

    components.html(wrapper, height=500)
    # components.html(lime_exp.as_html(), height=350)

    # ─── SHAP force plot for P(Yes) ───
    st.markdown("**2. SHAP: Shows how much each of your input impact the final prediction!**")
    expl = shap_explainer(X)     # Explanation with shape (1, n_features)
    single_exp = expl[0]          # pick the one row
    shap.initjs()
    fig = shap.plots.force(single_exp, matplotlib=True, show=False)
    st.pyplot(fig, bbox_inches="tight")



# MAIN CODE
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------
st.set_page_config(
    page_title="Bank Term Deposit App", 
    layout="wide"
)

# customer css styling
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
# @st.experimental_memo
@st.cache_data
# function that load the ML predictive models
def load_models():
    # Load Decision Tree (or Resampled Model)
    # with open('../../Model/DT_Model_Deploy.pkl', 'rb') as f:
    with open('../../Model/DT_Resampled_Model_Deploy.pkl', 'rb') as f:
        dt_pipeline = pickle.load(f)
        dt_model = dt_pipeline.named_steps['classifier']
    # Load Random Forest (or Resampled Model)
    # with open('../../Model/RF_Model_Deploy.pkl', 'rb') as f:
    with open('../../Model/RF_Resampled_Model_Deploy.pkl', 'rb') as f:
        rf_pipeline = pickle.load(f)
        rf_model = rf_pipeline.named_steps['classifier']
    # Load XGBoost (or Resampled Model)
    # with open('../../Model/XGB_Model_Deploy.pkl', 'rb') as f:
    with open('../../Model/XGB_Resampled_Model_Deploy.pkl', 'rb') as f:
        xgb_pipeline = pickle.load(f)
        xgb_model = xgb_pipeline.named_steps['classifier']
    return {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

# @st.experimental_singleton
@st.cache_resource
# functuion that load the data for this app
def load_data():
    # Load processed data for dashboard
    url = (
        "https://raw.githubusercontent.com/"
        "Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/refs/heads/main/Data/processed_Input.csv"
    )
    return pd.read_csv(url)

# --- REUSABLE UTILS ---
# Function that makes model prediction based on model input
def make_prediction(model, user_input):
    return model.predict(user_input)

# Takes user input for decisiion tree model
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

    # 3) Show input
    # IF USING REGULAR DT MODEL
    # # Assign inputs
    # age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=0)
    # balance = st.number_input("What is you bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=1)
    # duration = st.slider("How long was the Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=2)
    # campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=3)
    # pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=4)
    # poutcome = st.selectbox("What is the outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=5)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=6)

    # IF USING Resampled DT MODEL
    # Assign inputs
    age = st.slider("What is your client's age (18-65)?", min_value=18, max_value=65, value=42, key=0)
    balance = st.number_input("What is your client's bank account balance?", min_value=0, max_value=100000000, value=10000, key=1)
    housing = st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=2)  # Default value is 0
    duration = st.slider("How long was the duration of your client's last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=3)
    # campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=3)
    # pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=4)
    poutcome = st.selectbox("What is the outcome of your client' last campaign?", ["Unknown", "Failure", "Success"], index=0, key=4)  # Default value is 0
    days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=5)

    
    # 4) Input Handling
    age          = float(age)
    balance      = float(balance)
    if housing == "Yes":
        housing = 1
    else:
        housing = 0
    duration     = float(duration) * 60   # convert minutes → seconds
    # campaign     = float(campaign)
    # pdays        = float(pdays)
    days_in_year = float(days_in_year)
    if poutcome == "Failure":
        poutcome = 0
    elif poutcome == "Unknown":
        poutcome = 0.5
    else:
        poutcome = 1

    # 5) Return output as dict
    # **NEW**: return a dict instead of a list
    # return {
    #     "age":          age,
    #     "balance":      balance,
    #     "duration":     duration,
    #     "campaign":     campaign,
    #     "pdays":        pdays,
    #     "poutcome":     poutcome,
    #     "days_in_year": days_in_year,
    # }


    return {
        "age":          age,
        "balance":      balance,
        "housing":     housing,
        "duration":     duration,
        "poutcome":     poutcome,
        "days_in_year": days_in_year,
    }

# Takes user input for random forest (resampled) model
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

    # 3) take user input
    # Regular RF Model
    # Assign inputs
    # age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=10)
    # balance = st.number_input("What is your bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=11)
    # duration = st.slider("How long was the duration of your client's last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=12)
    # campaign = st.slider("How many times did we contact your client?", min_value=0, max_value=15, value=0, key=13)
    # pdays = st.slider("How many days ago when we last contacted your client?", min_value=0, max_value=1000, value=5, key=14)
    # poutcome = st.selectbox("What is the outcome of your client's last campaign?", ["Unknown", "Failure", "Success"], index=0, key=15)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=16)

    age = st.slider("What is your client's age (18-65)?", min_value=18, max_value=65, value=42, key=10)
    balance = st.number_input("What is your client' bank account balance?", min_value=0, max_value=100000000, value=10000, key=11)
    housing = st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=12)  # Default value is 0
    duration = st.slider("How long was the duration of your client's last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=13)
    pdays = st.slider("How many days ago when we last contacted your client?", min_value=0, max_value=1000, value=5, key=14)
    poutcome = st.selectbox("What is the outcome of your client's last campaign?", ["Unknown", "Failure", "Success"], index=0, key=15)  # Default value is 0
    marital_married = st.selectbox("Is your client married?", ["No", "Yes"], index=0, key=16)  # Default value is 0
    job_blue_collar = st.selectbox("Does your client have a blue collor job?", ["No", "Yes"], index=0, key=17)  # Default value is 0
    days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=18)
    
    # 4) input handling
    age = float(age)
    balance = float(balance)
    if housing == "Yes":
        housing = 1
    else:
        housing = 0
    duration = float(duration) *60
    pdays        = float(pdays)
    if marital_married == "Yes":
        marital_married = 1
    else:
        marital_married = 0
    if job_blue_collar == "Yes":
        job_blue_collar = 1
    else:
        job_blue_collar = 0

    days_in_year = float(days_in_year)

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    # 5) return input values in dict
    #  IF original model
    # return {
    #     "age":          age,
    #     "balance":      balance,
    #     "duration":     duration,
    #     "campaign":     campaign,
    #     "pdays":        pdays,
    #     "poutcome":     poutcome,
    #     "days_in_year": days_in_year,
    # }

    return {
        "age":          age,
        "balance":      balance,
        "housing":      housing,
        "duration":     duration,
        "pdays":        pdays,
        "poutcome":     poutcome,
        "marital_married": marital_married,
        "job_blue_collar": job_blue_collar,
        "days_in_year": days_in_year,
    }

# take user input for XGBoost
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

    # 3) Take user input
    # IF Regular XGB Model
    # input handling
    # housing = yes_no_mapping[st.selectbox("Do you have any housing loans?", ["No", "Yes"], index=0, key=20)]
    # loan = yes_no_mapping[st.selectbox("Do you have any personal loans?", ["No", "Yes"], index=0, key=21)]
    # duration = st.slider("Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=22)
    # pdays = st.slider("How many days ago was your last contacted by us?", min_value=0, max_value=1000, value=5, key=23)
    # poutcome = st.selectbox("Outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=24)  # Default value is 0
    # marital_married = yes_no_mapping[st.selectbox("Are you married?", ["No", "Yes"], index=0, key=25)]
    # job_blue_collar = yes_no_mapping[st.selectbox("Do you work as a blue collar job?", ["No", "Yes"], index=0, key=26)]
    # job_housemaid = yes_no_mapping[st.selectbox("Do you work as a housemaid?", ["No", "Yes"], index=0, key=27)]
    # days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=28)

    housing = yes_no_mapping[st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=20)]
    loan = yes_no_mapping[st.selectbox("Do your client have any personal loans?", ["No", "Yes"], index=0, key=21)]
    duration = st.slider("What is the duration of your client's last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=22)
    poutcome = st.selectbox("What is the outcome of your client's last campaign?", ["Unknown", "Failure", "Success"], index=0, key=23)  # Default value is 0
    contact_cellular = yes_no_mapping[st.selectbox("Do you contact your client based on his/her cellphone?", ["No", "Yes"], index=0, key=24)]
    # marital_single = yes_no_mapping[st.selectbox("Is your client single?", ["No", "Yes"], index=0, key=25)]
    # marital_married = yes_no_mapping[st.selectbox("Is your client married?", ["No", "Yes"], index=0, key=25)]
    # marital_divorced = yes_no_mapping[st.selectbox("Is your client divorced?", ["No", "Yes"], index=0, key=25)]

    marital_overall = st.selectbox("What is your client's marital status?", ["Single", "Married", "Divorced"], index=0, key=25)  # Default value is 0
    job_overall = st.selectbox("What does your client work as?", ["Unknown", "Blue Collar", "Management", "Services", "Technician"], index=0, key=26)  # Default value is 0

    # job_blue_collar = yes_no_mapping[st.selectbox("Does your client have a blue collar job?", ["No", "Yes"], index=0, key=26)]
    # job_management = yes_no_mapping[st.selectbox("Does your client have a management job?", ["No", "Yes"], index=0, key=26)]
    # job_services = yes_no_mapping[st.selectbox("Does your client work in ?", ["No", "Yes"], index=0, key=26)]
    # job_technician = yes_no_mapping[st.selectbox("Does your client have a blue collar job?", ["No", "Yes"], index=0, key=26)]

    
    # 4) input handling
    housing = float(housing)
    loan = float(loan)
    duration = float(duration) *60
    # duration*=60
    # days_in_year = float(days_in_year)
    if contact_cellular=="No":
        contact_cellular=0
    else:
        contact_cellular=1

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    single=0
    married=0
    divorced=0

    # marital
    if marital_overall== "Single":
        single=1
    elif marital_overall =="Married":
        married=1
    elif marital_overall =="Divorced":
        divorced=1

    # "Unknown", "Blue Collar", "Management", "Services", "Technician"
    blue_collar=0
    management=0
    services=0
    technician=0

    if job_overall== "Blue Collar":
        blue_collar=1
    elif job_overall =="Management":
        management=1
    elif job_overall =="Services":
        services=1
    elif job_overall =="Technician":
        technician=1

    # 5) return output in dict
    # IF Original XGB Model
    # return input values in dict
    # return {
    #     "housing":          housing,
    #     "loan":             loan,
    #     "duration":         duration,
    #     "pdays":            float(pdays),
    #     "poutcome":         poutcome,
    #     "marital_married":  marital_married,
    #     "job_blue_collar":  job_blue_collar,
    #     "job_housemaid":    job_housemaid,
    #     "days_in_year":     days_in_year,
    # }


    return {
        "housing":          housing,
        "loan":             loan,
        "duration":         duration,
        "poutcome":         poutcome,
        "contact_cellular": contact_cellular,
        "marital_single":   single,
        "marital_married":  married,
        "marital_divorced": divorced,
        "job_blue_collar":  blue_collar,
        "job_management":   management,
        "job_services":     services,
        "job_technician":   technician
    }

# Function that displays success/failed prediction
def display_prediction(prediction):
    col1, col2 = st.columns([0.1, 0.9])
    # print(prediction) # for testing

    # Predict success case:
    if prediction[0] == 1:
        with col1:
            st.image("Visualizations/Result_Icons/success_icon.png", width=50)  # Use an icon for success
        with col2:
            st.write("### The Marketing Campaign will Succeed!")
    # Predict failure case:
    elif prediction[0] == 0:
        with col1:
            st.image("Visualizations/Result_Icons/failure_icon.png", width=50)  # Use an icon for failure
        with col2:
            st.write("### The Marketing Campaign will Fail.")


# --- PAGE FUNCTIONS ---
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------

# Function convert image to base64
def _img_to_base64(path):
    """Read a local image file and return a base64 data-URI string."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"

# function that builds the homepage layout
def home_page(models, data, raw_data):
    # introduction
    st.header("Welcome to the [Term Deposit Subscription Prediction App]!")
    st.markdown("---")
    st.write(
        "This app uses data science and machine learning methodologies to improve the performance of a term deposit financial product, especially in product analytics, demand forecasting, and customer churning. "
    )
    st.write("Come try and pick one of the boxes below to get started!")
    st.markdown("<br>", unsafe_allow_html=True)

    # now each card is (Title, Description, page_key, ImagePath)
    cards = [
        (
            "Subscription Prediction",
            "Use our ML model to predict will a client subscribe the term deposit subscription!",
            "Deposit Subscription Prediction",
            "Visualizations/Homepage_Icons/predictive-icon.jpg"
        ),
        (
            "Interactive Dashboard",
            "Find out underlying trends and insights via exploratory data analysis (EDA)!",
            "Interactive Dashboard",
            "Visualizations/Homepage_Icons/dashboard-icon.jpg"
        ),
        (
            "Customer Segmentation",
            "Try to intelligently assign customer into groups with our clustering algorithm!",
            "Customer Segmentation",
            "Visualizations/Homepage_Icons/cluster-analysis-icon.jpg"
        ),
        (
            "Data Overview & Export",
            "Download & use our original data/ cleaned data after conudcting data preprcessing!",
            "Data Overview & Export",
            "Visualizations/Homepage_Icons/export-data-icon.jpg"
        ),
    ]

    # displays 4 boxes for the app's functionalities
    cols = st.columns(4, gap="large")
    for col, (title, desc, page_key, img_path) in zip(cols, cards):
        with col:
            # convert the image to base64
            uri = _img_to_base64(img_path)

            # build card HTML, with the <img> inside
            card_html = f"""
            <div style="
                border:2px solid #ccc;
                border-radius:8px;
                padding:16px 16px 24px 16px;    /* extra 8px bottom padding */
                height:300px;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
                background-color: transparent;  /* so your app’s dark bg shows through */
            ">
            <div>
                <h4 style="margin:0 0 8px 0; color:#fff;">{title}</h4>
                <p  style="font-size:0.9em;
                        color:#fff;             /* white description */
                        margin:0 0 12px 0;">
                {desc}
                </p>
            </div>
            <div style="
                text-align:center;
                margin-bottom:24px;            /* more space below the image */
            ">
                <img
                src="{uri}"
                style="max-width:100%;
                        max-height:120px;
                        object-fit:contain;"
                />
            </div>
            </div>
            """

            st.markdown(card_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # add a page navigation button
            if col.button("Try it out!", key=f"btn_{page_key}"):
                st.session_state.page = page_key
                st.experimental_rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    # st.write("— Alex 🙂")

# Prediction Page
def prediction_page(models, data):
    # introduction
    st.header("Predicting Term Deposit Subscription")
    st.markdown("---")
    st.subheader("Choose an AI model to make predictions!")

    # 0) build full_feature_list once
    full_feature_list = [c for c in data.columns if c != "y"]

    # Build the tabs for users to choose which model
    tabs = st.tabs(list(models.keys()))
    for tab, (name, model) in zip(tabs, models.items()):
        with tab:
            # st.subheader(f"{name} Model")
            # Dispatch to the right input form

            if name == 'Decision Tree':
                inputs_dict = user_input_form_decision_tree()
            elif name == 'Random Forest':
                inputs_dict = user_input_form_random_forest()
            else:
                inputs_dict = user_input_form_xgboost()

            # store input
            inputs = pd.DataFrame([inputs_dict])
            # print(inputs)
            feature_names = tuple(inputs_dict.keys())

            # button that allos prediction
            if st.button(f"Predict with {name}", key=name):
                pred = make_prediction(model, inputs)
                display_prediction(pred)

                # Load XAI explaianers to prepare for explaining prediction output
                shap_exp, lime_exp = load_explainers(model, data, feature_names)

                print("XAI:", inputs)
                st.markdown("---")

                # st.markdown("""<br></br>""")

                # Wrap around XAI explanations in an expander
                with st.expander("🔍 Check how the model makes its decision!", expanded=True):
                    show_explanations(
                        model,
                        inputs,
                        shap_exp,
                        lime_exp
                    )

# Function displaying the interactive dashboard page
def dashboard_page(data):
    # ─── Inject CSS for the card shape ─────────────────────────────────
    st.markdown(
        """
        <style>
        .kpi-card {
          background-color: #ffffff;
          border-radius: 12px;
          padding: 1rem;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          text-align: center;
          margin-bottom: 1rem;
        }
        /* If you use Plotly inside the card, make its paper transparent */
        .kpi-card .js-plotly-plot .plotly {
          background-color: transparent !important;
        }

        [data-testid="stMarkdownContainer"] h4 {
        background-color: #393939;
        color: white;
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        }

        /* 2) Style each of your 2×2 boxes */
        .box-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        }
        .rec-card {
        background-color: #393939;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        }
        .rec-card h2, .rec-card li {
        color: white; 
        margin: 0.5rem 0;
        }
        .rec-card ul {
        padding-left: 1.2rem;
        }

        /* NEW: style every Altair chart like a box */
        div[data-testid="stVegaLiteChart"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stPyplot"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        margin-bottom: 1rem !important;
        }

        /* make sure the chart canvas is transparent so the white shows through */
        div[data-testid="stPlotlyChart"] .plotly,
        div[data-testid="stVegaLiteChart"] svg {
        background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # display introductuin
    st.header("Interactive Dashboard")
    st.subheader("Choose Your Persona & Explore Key Metrics and Visualizations:")
    st.markdown("---")

    # sub function: visually shows KPIs
    def kpi_indicator(label, value, suffix="", color="#000000"):
        fig = go.Figure(go.Indicator(
            mode="number",
            value=value,
            title={
                "text": label,
                "font": {"size": 20, "color": "#FFFFFF"}      # title in your colour
            },
            number={
                "font": {"size": 48, "color": color},     # number in your colour
                "suffix": suffix
            },
            domain={"x": [0,1], "y": [0,1]}
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",  
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=140
        )
        return fig

    # ─── Layout: 5 equal columns ────────────────────────────────────────

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    CARD_START = '<div class="kpi-card" style="background:#fff;border-radius:12px;padding:1rem;box-shadow:0 4px 12px rgba(0,0,0,0.15);margin:1rem 0;">'
    CARD_END   = '</div>'

    # Col 1: persona selector + metric
    with k1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        persona = st.selectbox("User Persona:", ["Salesperson", "Marketing Manager"])
        st.metric("Most Important Factor:", "Call Duration")
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 2: Conversion Rate
    with k2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        fig = kpi_indicator("Conversion Rate", round(data['y'].mean()*100,2), "%", color="#e28743")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 3: Persona-specific KPI
    with k3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if persona == 'Marketing Manager':
            fig = kpi_indicator(
                "First-Time Conversion Rate",
                round(data[data['previous']==0]['y'].mean()*100,2),
                "%", color="#eab676"
            )
        else:
            fig = kpi_indicator(
                "Avg. Duration of Success (mins)",
                round(data[data['y']==1]['duration'].mean()/60,2),
                "", color="#76b5c5"
            )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 4: First Contact %
    with k4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        fig = kpi_indicator("First Contact %", round((data['previous']==0).mean()*100,2), "%", color="#FFB6C1")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 5: Persona-specific KPI
    with k5:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if persona == 'Marketing Manager':
            fig = kpi_indicator(
                "Avg. Acct. Balance for Success",
                round(data[data['y']==1]['balance'].mean(),2), color="#abdbe3"
            )
        else:
            fig = kpi_indicator(
                "Avg Past Success Rate",
                round(data[data['y']==1]['poutcome'].mean()*100,2),
                "%", color="#1e81b0"
            )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown("---")

    # different display outpuet for each persona
    if persona =="Marketing Manager":

        # --- Create our 2×2 grid ---
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        # plots for a salesperson
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. conversion heatmap 
        # 4. sales-based recommendations


        # Top-left box: Marketing-based recommendation
        with row1_col1:

            recommendations_html = """
            <div class="rec-card">
            <h2>Marketing-based Recommendations</h2>
            <ul>
                <li>Customers are more likely to subscribe on specific months (Mar, Aug, Nov, Dec).</li>
                <li>Most customers are around in their early 30s to late 30s.</li>
                <li>Conversion rate increases a lot when duration goes up, age doesn't play a huge factor.</li>
                <li>Customers with no loans are more likely to subscribe when the call was not long.</li>
                <li>Most customers have cellular samples thus they may not have a lot of time → need to develop strategies that can make them stay longer.</li>
                <li>Over 50% of previous successful cases still subscribe when we call; focus on these customers.</li>
            </ul>
            </div>
            """
            st.markdown(recommendations_html, unsafe_allow_html=True)

        # top-right box: display plots for both Sales and Marketing
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
    
        # bottom-left box: works both Sales and Marketing
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

        # Bottom-right box: distributions & heatmaps (Plots 5, 6 & 7)
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

    # if persona is salesperson
    elif persona =="Salesperson":

        # --- Create our 2×2 grid ---
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        # plots for a salesperson
        # 1. plot for sales overtime?
        # 2. plots for outcome per variable (w/select box)
        # 3. conversion heatmap 
        # 4. sales-based recommendations

        # Top-left box: Sales-based recommendation
        with row1_col1:

            recommendations_html = """
            <div class="rec-card">
            <h2>Sales-based Recommendations</h2>
            <ul>
                <li>If the client has past subscribed our product before, they are way more likely to subscribe again!</li>
                <li>Choose to contact the clients on either summer or around Christmas.</li>
                <li>Don't worry about clients who owed us! 40% of clients that has loans still subscibes to us!</li>
                <li>Over 50% of previous successful case still subscribes when we call, focus on these customers.</li>
                <li>Call as long as possible (ideally over 9 minutes). Call duration is the most important factor determining the campaign outcome!</li>
                <li>Most customers have cellular samples thus they may not have a lot of time, you need to attract their interest quickly!</li>
            </ul>
            </div>
            """
            st.markdown(recommendations_html, unsafe_allow_html=True)

        # Top-right box: display plots for both Sales and Marketing
        with row1_col2:
            st.subheader("Campaign Trend Over Time")
            ts_tab, ms_tab = st.tabs(["Daily Count","Monthly Success"])
            with ts_tab:
                # daily number of success over time plot
                st.altair_chart(daily_line_altair(data), use_container_width=True)
            with ms_tab:
                # monthly succeess rate
                st.altair_chart(monthly_success_altair(data), use_container_width=True)
    
        # Bottom-left box: display plots for both Sales and Marketing
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

        # Bottom-right box: displays plot for both Sales and Marketing
        # Box (2,2): distributions & heatmaps (Plots 5, 6 & 7)
        with row2_col2:
            st.subheader("Outcome Based on Past Campaign's Outcomes")
            no_past_tab, past_tab, inconclusive_tab= st.tabs([
                "No Past Campaign","Successful Past Campaign", "Inconclusive Past Campaign"
            ])

            with no_past_tab:
                # Plot 5: age donut for past failed scenarios
                st.plotly_chart(previous_donut(df=data, filter_val=0), use_container_width=True)

            with past_tab:
                # Plot 6: age donut for past success scenarios
                st.plotly_chart(previous_donut(df=data, filter_val=1), use_container_width=True)

            with inconclusive_tab:
                # Plot 7: age donut for past inconclusive scenarios
                st.plotly_chart(previous_donut(df=data, filter_val=0.5), use_container_width=True)

# Displays clustering page
def clustering_page(data): 
    st.header("Customer Segmentation")
    
    # let users to choose the features they want to use for clustering
    st.subheader('Feature Selection')

    # 1) Define your groups here:
    FEATURE_GROUPS = {
        "Personal Information": [
            "age", "education", "balance", "contact_telephone"
        ],
        "Loans": [
            "housing", "loan", "default"
        ],
        "Campaign Metrics": [
            "day", "month", "duration", "campaign", "pdays",
            "previous", "poutcome", "days_in_year"
        ],
        # "Contact Information": [
        #     "contact_telephone"
        # ],
        # "Marital Status": [
        #     "marital_divorced", "marital_married", "marital_single"
        # ],
        # "Employment Information": [
        #     "job_admin.", "job_blue_collar", "job_entrepreneur",
        #     "job_housemaid", "job_management", "job_retired",
        #     "job_self_employed", "job_services", "job_student",
        #     "job_technician", "job_unemployed", "job_unknown"
        # ],
    }

    FEATURE_DESCRIPTIONS = {
        "Personal Information":    "Core numeric features (age, education level, default flag, balance, contact means)",
        "Loans":                   "Whether the client has housing and/or personal loans",
        "Campaign Metrics":        "Details of past campaign contacts (timing, counts, outcomes)",
        # "Contact Information":     "Which channel was used to contact the client (cellular vs telephone)"
        # "Marital Status":          "One-hot flags for marital status categories",
        # "Employment Information":  "One-hot flags for each job category"
    }

    FEATURE_EXAMPLES = {
        "Personal Information":    "e.g. age, education level, personal balance etc.",
        "Loans":                   "e.g. any personal or housing loans",
        "Campaign Metrics":        "e.g. duration, previous contact, previous success etc.",
        # "Contact Information":     "e.g. contact through cellphone or homephone"
        # "Marital Status":          "e.g. married, single, or divorced",
        # "Employment Information":  "e.g. different areas of jobs"
    }

    # 2) Let the user pick feature groups
    group_names = list(FEATURE_GROUPS.keys())
    chosen_groups = st.multiselect("Select at least one feature group below for customer segmentation:", group_names)

    if not chosen_groups:
        st.warning("Please select at least one feature group.")
    else:
        # 3) Build a display table of Group → Description → Examples
        rows = []
        for g in chosen_groups:
            rows.append({
                "Feature Group": g,
                "Description":   FEATURE_DESCRIPTIONS.get(g, ""),
                "Examples":      FEATURE_EXAMPLES.get(g, "")
            })
        display_df = pd.DataFrame(rows)
        display_df.index = np.arange(1, len(display_df)+1)

        # 4) display selected feature group(s)
        st.subheader("The Features Groups Selected:")
        st.table(display_df)

        selected_cols = []
        for g in chosen_groups:
            selected_cols.extend(FEATURE_GROUPS[g])
        selected_cols = list(dict.fromkeys(selected_cols))
        st.write(f"{len(selected_cols)} columns are selected.")

        # ─────────────────────────────────────────────────
        # Invalidate old clustering if the feature set has changed
        # ─────────────────────────────────────────────────
        if st.session_state.get("last_selected_cols") != selected_cols:
            for key in ("clustering_done", "cluster_labels", "rf_model", "scaler"):
                st.session_state.pop(key, None)
            st.session_state["last_selected_cols"] = selected_cols

        # 5) Require at least two features
        if len(selected_cols) < 2:
            st.error("Pick at least two features to cluster.")
            # return
        else:
            # 6) Scale & cluster
            # only once per session, initialize our flags/holders
            if 'clustering_done' not in st.session_state:
                st.session_state['clustering_done']   = False
                st.session_state['cluster_labels']    = None
                st.session_state['rf_model']          = None
                st.session_state['scaler']            = None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data[selected_cols])

            # 7) When the user clicks Run Clustering, do the work *once*
            if st.button("Run HDBSCAN Clustering Algorithm for Customer Segmentation"):
                # a) cluster
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(round(X_scaled.shape[0]/100)),
                    min_samples=None,
                    cluster_selection_method='eom'
                )
                labels = clusterer.fit_predict(X_scaled)
                # b) train surrogate
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_scaled, labels)
                # c) stash everything in session_state
                st.session_state["clustering_done"]   = True
                st.session_state["cluster_labels"]    = labels
                st.session_state["rf_model"]          = rf
                st.session_state["scaler"]            = scaler

            # 8) If users cicked ever clicked Run Clustering bitton, show the explainers *always*
            if st.session_state['clustering_done']:
                # restore
                data["Cluster"] = st.session_state["cluster_labels"]
                rf_model        = st.session_state["rf_model"]
                scaler          = st.session_state["scaler"]
                X_scaled        = scaler.transform(data[selected_cols])

                st.markdown("---")
                show_example_table(data,selected_cols)
                st.markdown("---")
                # get the top features
                top_features=plot_violin_top_features_raw(data, selected_cols, top_n=3)
                # show 3d
                plot_3d_clusters_raw(data, selected_cols, top_features)
                st.markdown("---")
                # 1. Show table of feature means
                show_cluster_feature_means_raw(data, selected_cols)
                st.markdown("---")
                 # 3. Use RF's feature importance to find important factor of clustering
                plot_tree_feature_importance( data, X_scaled, selected_cols )
                # 4. SHAP & LIME Explanations
                st.markdown("---")

                with st.expander("Try enter a new customer and see which customer group he/she belongs"):
                    # st.header("Try enter a new customer and see where he/she goes using explainable AI (XAI)")
                    # show_shap_explanation_custom(rf_model, scaler, data, selected_cols, top_n=5 )
                    show_lime_explanation_custom(rf_model, scaler, data, selected_cols, top_n=5 )

# Showing the data overview & export page
def overview_page(data, preprocessed):
    
    st.header("Data Overview & Export")
    st.markdown("---")
    st.write("This page lets you download the dataset used for this app, either the original “raw” dataset or the cleaned & feature-engineered version.")
        
    st.write("This dataset captures information from direct marketing campaigns from a Portuguese banking institution. Its goal is to predict whether its clients will subscribe a term deposit or not.")
    
    st.markdown("---")

    # 2 box with a divider separating them
    col1, col_div, col2 = st.columns([1, 0.02, 1])

    st.markdown("---")

    
    # Raw data box
    # ─── Raw Data ──────────────────────────────────────────────
    with col1:
        st.subheader("Raw Data")

        # 1) Brief description
        st.markdown(
            f"""
            - **Rows:** {data.shape[0]:,}  
            - **Columns:** {data.shape[1]:,}  
            """
        )
        st.markdown(
            """
            **Contents:**  
            1. Client's personal and financial information 
            2. Client's Contact & campaign history 
            3. The final subscription outcome 
            """
        )

        # 2) Download buttons
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Raw Data (.csv)",
            data=csv,
            file_name="raw_data.csv",
            mime="text/csv",
        )
        buf = io.BytesIO()
        data.to_excel(buf, index=False, sheet_name="raw", engine="openpyxl")
        buf.seek(0)
        st.download_button(
            "📥 Download Raw Data (.xlsx)",
            data=buf,
            file_name="raw_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ─── Divider ───────────────────────────────────────────────
    with col_div:
        st.markdown(
        """
        <div style="
            border-left:2px solid #ccc;
            height:100%;
            min-height:500px;   /* adjust to match your content */
            margin:0 auto;
        "></div> 
        """,
        unsafe_allow_html=True
        )
    
    # Preprocessed data box
    # ─── Preprocessed Data ─────────────────────────────────────
    with col2:
        st.subheader("Preprocessed Data")

        # 1) Brief description
        st.markdown(
            f"""
            - **Rows:** {preprocessed.shape[0]:,} (-11)  
            - **Columns:** {preprocessed.shape[1]:,} (+15) 
            """
        )
        st.markdown(
            """
            **This version has been cleaned & feature‐engineered for further data & AI usages, including:**  
            1. **Data Cleaning:** to remove missing & duplicate entries, include anomaly detection & removal.
            2. **Data Transformation:** Encoding categorical features & scaling numerics.  
            3. **Create/ modify features** from existing data.  
            """
        )

        # 2) Download buttons
        csv2 = preprocessed.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Processed Data (.csv)",
            data=csv2,
            file_name="preprocessed_data.csv",
            mime="text/csv",
        )
        buf2 = io.BytesIO()
        preprocessed.to_excel(buf2, index=False, sheet_name="processed", engine="openpyxl")
        buf2.seek(0)
        st.download_button(
            "📥 Download Processed Data (.xlsx)",
            data=buf2,
            file_name="preprocessed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    
# Displays the final acknowledgement page
def acknowledgement_page(data):

    # display text
    st.header("Acknowledgements")
    st.markdown("---")
    ack_html = """
    First of all, this entire application comes from a graudate data science course project where my teammates **Zheng En Than** and **Emily Au**. We cleaned the collected data, 
    performed exploratory data analysis, and developed the machine learning models. I sincerely thank them for their effort and hard work. 
    I would also like to thank my course instructor **Dr. Jay Newby** for his guidance and mentorship.
    <br><br>
    This app is made with a purpose to applied our research in a high level, if you are interested to learn more about the scientific details of our work, please visit the <a href="https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project" target="_blank"><strong>User Guide</strong></a>!
    <br><br>
    Additionally, I want to acknowledge **Sérgio Moro**, **P. Cortez**, and **P. Rita** for sharing the UCI ML Bank Telemarketing Dataset which is the fundament backbomne of this project.
    <br><br>
    Last but not least, shout out to the user test group [TEMP NAMES]. Their opinions and feedback on this project should be recognized.

    """
    st.markdown(ack_html, unsafe_allow_html=True)

    st.image("Visualizations/title_icon_temp.gif", width=300, caption="Me vibin' when I am creating this project :)")

# --- MAIN APP ---
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------


# global dictionary to link page and index
PAGE_TO_INDEX = {
    "Home":                              0,
    "Deposit Subscription Prediction":  1,
    "Interactive Dashboard":            2,
    "Customer Segmentation":            3,
    "Data Overview & Export":           4,
    "Acknowledgements":                 5,
}

# The main function that puts everything together
def main():
    # st.title("Bank Term Deposit App")

    # only update page, sidebar will be sync iin a different way
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    # if "sidebar_choice" not in st.session_state:
    #     st.session_state.sidebar_choice = "Home"

    # define a sidebar callback
    def sidebar_navigate():
        st.session_state.page = st.session_state.sidebar_choice

    # Sync sidebar here
    def _sync_page_with_sidebar(new_choice):
        # ignore new_choice, just copy over the widget state
        st.session_state.page = st.session_state.sidebar_choice

    # display sidebar
    with st.sidebar:
        # title
        st.title("Deposit Subscription Prediction Data Science Project")
        st.caption("v1.1.0 • Data updated: 2025-06-28") 

        st.markdown("---")

        current = st.session_state.page
        idx     = PAGE_TO_INDEX.get(current, 0)

        # option menu for users to select pages to navigate
        choice = option_menu(
            menu_title=None,
            # options=["Home", "Deposit Subscription Prediction", "Interactive Dashboard", "Customer Segmentation", "Data Overview & Export", "Acknowledgements"],
            options=list(PAGE_TO_INDEX.keys()),
            icons=["house", "bank", "bar-chart-line", "pie-chart-fill", "cloud-download", "award"],
            menu_icon="app-indicator",
            # default_index=0,
            default_index=idx,
            orientation="vertical",
            # key="sidebar_choice",
            # on_change=lambda: st.session_state.update(page=st.session_state.sidebar_choice)
            # on_change=lambda *args: st.session_state.update(page=st.session_state.sidebar_choice)
            # on_change=_sync_page_with_sidebar
        )
        if choice != current:
            st.session_state.page = choice



        # --- Help & feedback Expander---
        with st.expander("❓ Help & Docs"):
            st.write("- [User Guide](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project)")
            st.write("- [Source Code](https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/tree/main/Code)")
            st.write("- [Contact Us](https://www.linkedin.com/in/alex-mak-824187247/)")
        
        st.caption("© 2025 Alex Mak, All Rights Reserved")

    # choice = st.sidebar.radio("Go to", ["Prediction", "Dashboard"] )

    # init functions: load models and data
    models = load_models()
    data   = load_data()

    raw_data=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project/refs/heads/main/Data/input.csv", sep=";")

    # get current page
    page = st.session_state.page


    # back page button at the top of every page except home page
    if st.session_state.page != "Home":
        if st.button("← Back to Home"):
            st.session_state.page = "Home"
            st.experimental_rerun()

    # page navigation based on user's selection
    if page == "Home":
        home_page(models, data, raw_data)
    elif page == "Deposit Subscription Prediction":
        prediction_page(models, data)
    elif page == "Interactive Dashboard":
        dashboard_page(data)
    elif page == "Customer Segmentation":
        clustering_page(data)
    elif page == "Data Overview & Export":
        overview_page(raw_data, data)
    elif page == "Acknowledgements":
        acknowledgement_page(data)

if __name__ == "__main__":
    main()
