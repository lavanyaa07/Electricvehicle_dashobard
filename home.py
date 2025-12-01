import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="EV Dashboard", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("EV_Cleaned.csv")

df = load_data()

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("📊 EV Dashboard Menu")
page = st.sidebar.radio("Navigate", ["Home", "Dataset Preview", "EDA Charts", "Insights"])

# ----------------------------------------
# HOME PAGE
# ----------------------------------------
if page == "Home":
    st.title("⚡ Electric Vehicle Dashboard")
    st.write("Welcome to your EV Dashboard! Explore dataset, EDA insights, and analysis.")
    st.success("Streamlit done successfully!!")

# ----------------------------------------
# DATASET PREVIEW
# ----------------------------------------
elif page == "Dataset Preview":
    st.title("📁 Dataset Preview")
    st.write("Preview the first few rows of your dataset:")
    st.dataframe(df.head())

    st.write("### Dataset Shape")
    st.write(df.shape)

    st.write("### Missing Values")
    st.write(df.isnull().sum())

# ----------------------------------------
# EDA CHARTS
# ----------------------------------------
elif page == "EDA Charts":
    st.title("📈 EDA Charts")

    chart_type = st.selectbox(
        "Choose Chart Type",
        [
            "Top 15 Makes",
            "Model Year Trend",
            "Electric Range Distribution",
            "Top Cities",
            "Top States",
            "Base MSRP Distribution",
            "EV Type Count",
            "Make vs Electric Range",
            "State vs Electric Range",
        ],
    )

    # PLOT: TOP MAKES
    if chart_type == "Top 15 Makes":
        st.subheader("Top 15 Vehicle Makes")
        top_makes = df["Make"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top_makes.values, y=top_makes.index, palette="Blues_r", ax=ax)
        st.pyplot(fig)

    # PLOT: MODEL YEAR TREND
    elif chart_type == "Model Year Trend":
        st.subheader("EV Count by Model Year")
        year_counts = df.groupby("Model_Year").size()
        fig, ax = plt.subplots(figsize=(10,4))
        sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax)
        st.pyplot(fig)

    # ELECTRIC RANGE DISTRIBUTION
    elif chart_type == "Electric Range Distribution":
        st.subheader("Electric Range Distribution")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.histplot(df["Electric_Range"], kde=True, bins=40, color="skyblue", ax=ax)
        st.pyplot(fig)

    # TOP CITIES
    elif chart_type == "Top Cities":
        st.subheader("Top 15 Cities")
        top_cities = df["City"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top_cities.values, y=top_cities.index, palette="viridis", ax=ax)
        st.pyplot(fig)

    # TOP STATES
    elif chart_type == "Top States":
        st.subheader("Top 15 States")
        top_states = df["State"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top_states.values, y=top_states.index, palette="magma", ax=ax)
        st.pyplot(fig)

    # BASE MSRP
    elif chart_type == "Base MSRP Distribution":
        st.subheader("Base MSRP Distribution")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.boxplot(x=df["Base_MSRP"], color="orange", ax=ax)
        st.pyplot(fig)

    # EV TYPE COUNT
    elif chart_type == "EV Type Count":
        st.subheader("EV Type Count (BEV vs PHEV)")
        type_counts = df["Electric_Vehicle_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x=type_counts.index, y=type_counts.values, palette="Set2", ax=ax)
        st.pyplot(fig)

    # MAKE VS ELECTRIC RANGE
    elif chart_type == "Make vs Electric Range":
        st.subheader("Make vs Electric Range")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(
            data=df[df["Electric_Range"] > 0].groupby("Make")["Electric_Range"].mean().sort_values(ascending=False).head(15).reset_index(),
            x="Electric_Range",
            y="Make",
            palette="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

    # STATE VS ELECTRIC RANGE
    elif chart_type == "State vs Electric Range":
        st.subheader("State vs Electric Range")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(
            data=df[df["Electric_Range"] > 0],
            x="State",
            y="Electric_Range",
            ax=ax
        )
        plt.xticks(rotation=90)
        st.pyplot(fig)

# ----------------------------------------
# INSIGHTS PAGE
# ----------------------------------------
elif page == "Insights":
    st.title("💡 Insights & Observations")

    st.write("### 🔍 Key Findings from the EDA")
    st.write("""
    **1️⃣ Electric Range Missing for ~60% Rows**  
    - Most 0 values in `Electric_Range` represent **unknown/battery range not recorded**, not actual zero range.

    **2️⃣ Tesla is the Most Popular EV Manufacturer**  
    - Tesla dominates the dataset with the highest EV count.

    **3️⃣ Washington State Cities Lead EV Adoption**  
    - Cities like Seattle, Bellevue, and Redmond appear frequently.

    **4️⃣ EV Type Distribution**  
    - Battery Electric Vehicles (BEV) dominate ~80%.
    - Plug-in Hybrid Electric Vehicles (PHEV) form the rest.

    **5️⃣ Electric Range Distribution**  
    - Many EVs have ranges between **20–60 miles** (mostly PHEVs).
    - Highest range values reach **337 miles**.

    **6️⃣ Base MSRP**  
    - Many EV entries have MSRP = 0 (missing).
    - Real values show a large spread — from 20k to 180k.

    **7️⃣ Model Year Trend**  
    - EV adoption grows year after year.
    - Peak entries appear from **2020–2024**.

    """)

    st.success("🎉 THANK YOUUU!!!")


