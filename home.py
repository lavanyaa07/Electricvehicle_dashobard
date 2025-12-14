import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="EV Analytics & ML Dashboard",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("EV_Cleaned.csv")

df = load_data()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚡ EV Dashboard")
st.sidebar.markdown("Analyze Electric Vehicle data using **EDA & ML**")

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "Overview",
        "Dataset",
        "Exploratory Data Analysis",
        "EV Clustering (ML)",
        "Key Insights & Conclusion"
    ]
)

# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "Overview":
    st.title("⚡ Electric Vehicle Analytics Dashboard")

    st.markdown("""
    ### 📘 Project Objective
    This dashboard analyzes **Electric Vehicle (EV) population data** using:
    - 📊 Exploratory Data Analysis (EDA)
    - 🤖 Machine Learning (K-Means Clustering)

    The goal is to **understand EV adoption trends** and **segment EVs into meaningful market groups**.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Unique EV Makes", df["Make"].nunique())

    st.success("✅ Dashboard loaded successfully")

# =========================================================
# DATASET PAGE
# =========================================================
elif page == "Dataset":
    st.title("📁 Dataset Overview")

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# =========================================================
# EDA PAGE
# =========================================================
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    chart = st.selectbox(
        "Select Analysis",
        [
            "Top EV Manufacturers",
            "EV Adoption by Model Year",
            "Electric Range Distribution",
            "Top Cities by EV Count",
            "Top States by EV Count",
            "Base MSRP Distribution",
            "EV Type Distribution",
        ]
    )

    if chart == "Top EV Manufacturers":
        top_makes = df["Make"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_makes.values, y=top_makes.index, ax=ax)
        ax.set_title("Top 15 EV Manufacturers")
        st.pyplot(fig)

    elif chart == "EV Adoption by Model Year":
        year_counts = df.groupby("Model_Year").size()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax)
        ax.set_title("EV Adoption Trend Over Years")
        st.pyplot(fig)

    elif chart == "Electric Range Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df["Electric_Range"], bins=40, kde=True, ax=ax)
        ax.set_title("Electric Range Distribution")
        st.pyplot(fig)

    elif chart == "Top Cities by EV Count":
        top_cities = df["City"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_cities.values, y=top_cities.index, ax=ax)
        ax.set_title("Top 15 Cities with EV Adoption")
        st.pyplot(fig)

    elif chart == "Top States by EV Count":
        top_states = df["State"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_states.values, y=top_states.index, ax=ax)
        ax.set_title("Top 15 States by EV Count")
        st.pyplot(fig)

    elif chart == "Base MSRP Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df["Base_MSRP"], ax=ax)
        ax.set_title("Base MSRP Distribution")
        st.pyplot(fig)

    elif chart == "EV Type Distribution":
        type_counts = df["Electric_Vehicle_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=type_counts.index, y=type_counts.values, ax=ax)
        ax.set_title("EV Type Distribution (BEV vs PHEV)")
        st.pyplot(fig)

# =========================================================
# CLUSTERING PAGE
# =========================================================
elif page == "EV Clustering (ML)":
    st.title("🤖 EV Market Segmentation using K-Means")

    st.markdown("""
    ### 🧠 Machine Learning Approach
    - Algorithm: **K-Means Clustering**
    - Features Used:
      - Electric Range
      - Base MSRP
    - Only EVs with **valid price information** are considered.
    """)

    clean_df = df[df["Base_MSRP"] > 0].copy()
    features = clean_df[["Electric_Range", "Base_MSRP"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clean_df["Cluster"] = kmeans.fit_predict(scaled_features)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=clean_df,
        x="Electric_Range",
        y="Base_MSRP",
        hue="Cluster",
        palette="Set1",
        ax=ax
    )
    ax.set_title("EV Clusters based on Range and Price")
    st.pyplot(fig)

    st.subheader("📊 Cluster-wise Average Values")
    st.dataframe(
        clean_df.groupby("Cluster")[["Electric_Range", "Base_MSRP"]].mean()
    )

# =========================================================
# INSIGHTS PAGE
# =========================================================
elif page == "Key Insights & Conclusion":
    st.title("💡 Key Insights & Conclusion")

    st.markdown("""
    ### 🔍 Key Findings
    - Tesla is the dominant EV manufacturer.
    - EV adoption has increased significantly after 2020.
    - Most short-range EVs are PHEVs.
    - Base MSRP shows wide variation, indicating diverse EV market segments.

    ### 🤖 ML Conclusion
    - K-Means clustering successfully segmented EVs into:
      - **Budget EVs**
      - **Mid-range EVs**
      - **Premium EVs**
    - Removing missing price values significantly improved clustering quality.

    ### ✅ Final Outcome
    This project demonstrates an **end-to-end data science pipeline**:
    Data Cleaning → EDA → Machine Learning → Visualization.
    """)

    st.success("🎉 Project completed successfully!")
