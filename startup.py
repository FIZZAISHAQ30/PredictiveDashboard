import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance MVP",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔧"
)

# ---------------------------
# Full Black Theme CSS
# ---------------------------
st.markdown(
    """
    <style>
    /* Main background & text */
    .reportview-container, .main, .block-container {
        background-color: #000000;
        color: #ffffff;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    /* DataFrame */
    .stDataFrame table {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    /* Buttons */
    .stButton>button {
        background-color: #1abc9c;
        color: #000000;
    }
    /* Alerts */
    .stError {color: #ff4b4b;}
    .stSuccess {color: #1abc9c;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Title & Instructions
# ---------------------------
st.title("🔧 Virtual Predictive Maintenance Dashboard")
st.markdown("""
Simulated real-time machine monitoring — No hardware required.  

**Instructions:**  
- Use the sidebar sliders to simulate live machine readings.  
- Red alert = potential machine failure.  
- Green = normal operating range.
""")

# ---------------------------
# Function: Simulate Data
# ---------------------------
def simulate_data(n_points=50):
    machines = ['M1', 'M2', 'M3']
    data = []
    for _ in range(n_points):
        for m in machines:
            temp = np.random.randint(60, 100)
            vib = round(np.random.uniform(1, 6), 2)
            rpm = np.random.randint(900, 2000)
            failure = 1 if temp > 85 or vib > 4.5 else 0
            data.append([m, temp, vib, rpm, failure])
    df = pd.DataFrame(data, columns=['Machine', 'Temp', 'Vibration', 'RPM', 'Failure'])
    return df

# ---------------------------
# Train ML Model
# ---------------------------
def train_model(df):
    X = df[['Temp', 'Vibration', 'RPM']]
    y = df['Failure']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# ---------------------------
# Initial Data & Model
# ---------------------------
df = simulate_data()
model = train_model(df)

# ---------------------------
# Sidebar for Live Input
# ---------------------------
st.sidebar.header("🚨 Live Machine Status Input")
temp_in = st.sidebar.slider("Temperature (°C)", 60, 120, 75)
vib_in = st.sidebar.slider("Vibration Level", 1.0, 6.0, 3.0)
rpm_in = st.sidebar.slider("RPM", 900, 2000, 1500)

input_data = pd.DataFrame([[temp_in, vib_in, rpm_in]], columns=['Temp', 'Vibration', 'RPM'])
pred = model.predict(input_data)[0]

if pred == 1:
    st.error("❗ ALERT: Machine Failure Predicted")
else:
    st.success("✔ Machine Status: NORMAL")

# ---------------------------
# Placeholders for live charts
# ---------------------------
st.subheader("📌 Live Machine Data")
data_placeholder = st.empty()

st.subheader("📊 Temperature & Vibration Scatter Plot")
scatter_placeholder = st.empty()

st.subheader("📈 Machine Failure Count by Machine")
bar_placeholder = st.empty()

# ---------------------------
# Live Simulation Loop
# ---------------------------
for _ in range(20):
    live_df = simulate_data(n_points=3)
    data_placeholder.dataframe(live_df)
    
    scatter_placeholder.scatter_chart(live_df[['Temp', 'Vibration']])
    
    failure_count = live_df.groupby('Machine')['Failure'].sum().reset_index()
    bar_placeholder.bar_chart(failure_count.set_index('Machine'))
    
    time.sleep(2)

# ---------------------------
# Footer (Centered Developer Name)
# ---------------------------
st.write("")
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; font-weight: bold; color: white; font-size: 18px;'>
        Developed by Fizza R Ishaq
    </div>
    """,
    unsafe_allow_html=True
)
