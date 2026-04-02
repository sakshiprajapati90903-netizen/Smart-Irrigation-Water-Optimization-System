
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgroSense - Smart Irrigation AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load("smart_irrigation_model.pkl")
    le_crop = joblib.load("crop_encoder.pkl")
    le_soil = joblib.load("soil_encoder.pkl")
    le_region = joblib.load("region_encoder.pkl")
    le_weather = joblib.load("weather_encoder.pkl")
    return model, le_crop, le_soil, le_region, le_weather

@st.cache_data
def load_data():
    df = pd.read_csv("DATASET.csv")
    df.columns = [
        'Crop_Type',
        'Soil_Type',
        'Region',
        'Temperature',
        'Weather_Condition',
        'Water_Requirement'
    ]
    return df

model, le_crop, le_soil, le_region, le_weather = load_model()
df = load_data()

# ---------------- FUNCTIONS ----------------
def convert_temp_mean(temp_range):
    low, high = map(int, temp_range.split('-'))
    return (low + high) / 2

# ---------------- CSS ----------------
st.markdown("""
<style>
/* -------- Global -------- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #050b16;
    color: #e5e7eb;
}

/* -------- Sidebar -------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020817 0%, #06111f 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
    min-width: 290px !important;
    max-width: 290px !important;
}

section[data-testid="stSidebar"] * {
    color: #d1d5db !important;
}

.logo-title {
    font-size: 2rem;
    font-weight: 800;
    color: #4ade80;
    margin-bottom: 0.2rem;
}

.logo-sub {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

.sidebar-section-title {
    color: #6b7280;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.sidebar-info {
    font-size: 1rem;
    margin-bottom: 0.6rem;
}

.sidebar-green {
    color: #4ade80 !important;
    font-weight: 600;
}

/* -------- Main Heading -------- */
.page-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.3rem;
}

.page-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

/* -------- Cards -------- */
.card {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
    margin-bottom: 20px;
}

.card-title {
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
}

.metric-number {
    font-size: 2.3rem;
    font-weight: 800;
    color: #4ade80;
}

.metric-label {
    color: #6b7280;
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* -------- Input Labels -------- */
.stSelectbox label, .stTextInput label {
    color: #d1d5db !important;
    font-weight: 500;
}

/* -------- Buttons -------- */
.stButton > button {
    background: #2ea043 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.3rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100%;
    transition: 0.3s ease;
}

.stButton > button:hover {
    background: #38c172 !important;
    transform: translateY(-2px);
}

/* -------- Result Box -------- */
.result-box {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px;
    padding: 35px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(0,0,0,0.2);
}

.result-water {
    font-size: 2.5rem;
    font-weight: 800;
    color: #4ade80;
}

.result-sub {
    color: #9ca3af;
    font-size: 1rem;
    margin-top: 8px;
}

/* -------- Feature Tags -------- */
.tech-pill {
    display: inline-block;
    background: #0b1220;
    border: 1px solid rgba(255,255,255,0.06);
    color: #d1d5db;
    padding: 10px 16px;
    border-radius: 12px;
    margin: 6px 8px 6px 0;
    font-size: 0.95rem;
}

.green-dot {
    color: #4ade80;
    font-size: 1rem;
    margin-right: 8px;
}

/* -------- Footer -------- */
.footer {
    text-align: center;
    color: #4b5563;
    font-size: 0.95rem;
    margin-top: 50px;
    padding-top: 20px;
}

/* -------- Dataframe -------- */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}

/* -------- Chart background -------- */
[data-testid="stPlotlyChart"], [data-testid="stImage"], canvas {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<div class='logo-title'>🌿 AgroSense</div>", unsafe_allow_html=True)
    st.markdown("<div class='logo-sub'>Smart Irrigation AI · v1.0</div>", unsafe_allow_html=True)

    page = st.selectbox(
        "",
        ["About", "Prediction", "Data & Insights"]
    )

    st.markdown("---")

    st.markdown("<div class='sidebar-section-title'>Model Info</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>Algorithm&nbsp;&nbsp;<span class='sidebar-green'>Random Forest</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>Features&nbsp;&nbsp;<span class='sidebar-green'>5 inputs</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>Target&nbsp;&nbsp;<span class='sidebar-green'>Water (L/day)</span></div>", unsafe_allow_html=True)

# ---------------- ABOUT PAGE ----------------
# ---------------- ABOUT PAGE ----------------
if page == "About":
    st.markdown("<div class='page-title'>Smart Irrigation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>AI-powered irrigation support for gardens, landscapes, and building complexes</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number'>{df.shape[0]}</div>
            <div class='metric-label'>Data Records</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number'>{df['Crop_Type'].nunique()}</div>
            <div class='metric-label'>Crop Types</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-number'>{df['Region'].nunique()}</div>
            <div class='metric-label'>Regions</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-number'>5</div>
            <div class='metric-label'>Input Features</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # TOP SECTION
    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Project Objective</div>", unsafe_allow_html=True)
        st.write("""
This project is designed to support **automatic irrigation management**
for **building complexes, society gardens, landscaped areas, and smart green spaces**.

The main goal is to help decide when a **water pump should be turned ON or OFF**
based on agricultural and environmental conditions such as:

- 🌱 Crop / Plant Type  
- 🪨 Soil Type  
- 📍 Region  
- 🌡️ Temperature  
- ☁️ Weather Condition  

Instead of manually watering plants every day, the system helps estimate
the **required water level**, which can be further used in a real-world setup
to automate irrigation scheduling.
""")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Tech Stack</div>", unsafe_allow_html=True)

        techs = [
            "Python", "Streamlit", "Scikit-learn",
            "Random Forest", "Pandas", "Matplotlib",
            "NumPy", "Joblib", "Seaborn"
        ]

        cols = st.columns(2)
        for i, tech in enumerate(techs):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:#0b1220;padding:14px 16px;border-radius:12px;
                border:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">
                    <span style="color:#4ade80;">●</span> {tech}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # SECOND SECTION
    left2, right2 = st.columns([1.7, 1])

    with left2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>How It Helps</div>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            <div style="background:#0b1220;padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">
            <h4>💧 Reduce Wastage</h4>
            <p style="color:#9ca3af;">Avoid unnecessary watering with smarter irrigation estimates</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#0b1220;padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,0.05);">
            <h4>⚡ Pump ON/OFF Logic</h4>
            <p style="color:#9ca3af;">Can be connected to an automatic pump control system</p>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.markdown("""
            <div style="background:#0b1220;padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">
            <h4>🏢 Useful for Complex Gardens</h4>
            <p style="color:#9ca3af;">Ideal for residential gardens, campus lawns, and society landscapes</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#0b1220;padding:18px;border-radius:14px;border:1px solid rgba(255,255,255,0.05);">
            <h4>📊 Data-Based Decisions</h4>
            <p style="color:#9ca3af;">Supports irrigation planning using machine learning</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Input Features</div>", unsafe_allow_html=True)

        features = [
            ("Crop Type", "categorical"),
            ("Soil Type", "categorical"),
            ("Region", "categorical"),
            ("Temperature", "numeric"),
            ("Weather Condition", "categorical"),
        ]

        for f, t in features:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.06);">
                <span>{f}</span>
                <span style="background:#1e3a5f;color:#bfdbfe;padding:6px 10px;border-radius:10px;font-size:0.85rem;">{t}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION PAGE ----------------
elif page == "Prediction":
    st.markdown("<div class='page-title'>Water Requirement Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Estimate irrigation needs and simulate automatic pump decision support for smart gardens.</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    # ---------- LEFT INPUT PANEL ----------
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Crop & Soil</div>", unsafe_allow_html=True)

        crop = st.selectbox("Crop Type", list(le_crop.classes_))
        soil = st.selectbox("Soil Type", list(le_soil.classes_))

        st.write("")
        st.markdown("<div class='card-title'>Location & Climate</div>", unsafe_allow_html=True)

        region = st.selectbox("Region", list(le_region.classes_))
        temperature = st.selectbox("Temperature Range (°C)", ['10-20', '20-30', '30-40', '40-50'])
        weather = st.selectbox("Weather Condition", list(le_weather.classes_))

        st.write("")
        predict_btn = st.button("Run Prediction →")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RIGHT RESULT PANEL ----------
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:4rem;margin-bottom:10px;'>💧</div>", unsafe_allow_html=True)

        if predict_btn:
            temp_numeric = convert_temp_mean(temperature)

            sample = pd.DataFrame({
                'Crop_Type': [le_crop.transform([crop])[0]],
                'Soil_Type': [le_soil.transform([soil])[0]],
                'Region': [le_region.transform([region])[0]],
                'Temperature': [temp_numeric],
                'Weather_Condition': [le_weather.transform([weather])[0]]
            })

            prediction = model.predict(sample)[0]

            # -------- SMART PUMP DECISION LOGIC --------
            if prediction < 8:
                pump_status = "OFF"
                recommendation = "No irrigation required right now"
                status_color = "#f59e0b"
                automation_note = "Soil and weather conditions indicate low irrigation demand."
                status_msg = "Low irrigation requirement — pump can remain OFF."
                status_type = "success"

            elif prediction > 8 and prediction < 12:
                pump_status = "ON (Controlled)"
                recommendation = "Balanced irrigation recommended"
                status_color = "#3b82f6"
                automation_note = "System suggests moderate watering for healthy plant growth."
                status_msg = "Moderate irrigation requirement — controlled watering suggested."
                status_type = "info"

            else:
                pump_status = "ON"
                recommendation = "Immediate irrigation required"
                status_color = "#22c55e"
                automation_note = "High water demand detected. Pump should be activated."
                status_msg = "High irrigation requirement — pump should be turned ON."
                status_type = "warning"

            # -------- MAIN RESULT BOX --------
            st.markdown(f"""
            <div class='result-box'>
                <div class='result-water'>{prediction:.2f} L/day</div>
                <div class='result-sub'>Estimated daily water requirement</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # -------- STATUS CARDS --------
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 18px;
                    padding: 22px;
                    text-align:center;
                    margin-bottom:15px;">
                    <div style="color:#9ca3af;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase;">Pump Status</div>
                    <div style="font-size:2rem;font-weight:800;color:{status_color};margin-top:12px;">{pump_status}</div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
                    border: 1px solid rgba(255,255,255,0.06);
                    border-radius: 18px;
                    padding: 22px;
                    text-align:center;
                    margin-bottom:15px;">
                    <div style="color:#9ca3af;font-size:0.9rem;letter-spacing:2px;text-transform:uppercase;">System Recommendation</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-top:12px;">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)

            # -------- AUTOMATION INSIGHT --------
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Automation Insight</div>
                <p style="font-size:1.02rem;color:#d1d5db;line-height:1.8;">
                    This smart irrigation dashboard can support <b>automatic pump ON/OFF decisions</b>
                    for <b>building gardens, society landscapes, campus lawns, and green spaces</b>
                    using crop and environmental conditions.
                </p>
                <p style="color:#9ca3af; line-height:1.8;">
                    <b>System Note:</b> {automation_note}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # -------- ALERT --------
            if status_type == "success":
                st.success(status_msg)
            elif status_type == "info":
                st.info(status_msg)
            else:
                st.warning(status_msg)

        else:
            st.markdown("""
            <div style='text-align:center;color:#6b7280;font-size:1.2rem;margin-top:10px;line-height:1.8;'>
                Fill in the fields on the left and click<br>
                <span style='color:#4ade80;font-weight:700;'>Run Prediction</span> to simulate smart irrigation.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DATA PAGE ----------------
elif page == "Data & Insights":
    st.markdown("<div class='page-title'>Dataset & Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Explore the training data and understand feature importance</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Dataset Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Water Requirement Distribution</div>", unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(df["Water_Requirement"], bins=20, edgecolor='limegreen')
        ax1.set_facecolor("#111827")
        fig1.patch.set_facecolor("#111827")
        ax1.tick_params(colors='white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        ax1.set_xlabel("Water Requirement")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Feature Importance</div>", unsafe_allow_html=True)

        feature_names = ['Crop_Type', 'Soil_Type', 'Region', 'Temperature', 'Weather_Condition']
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.barh(importance['Feature'], importance['Importance'], color=['#22364d','#22364d','#22364d','#22364d','#2ea043'])
        ax2.set_facecolor("#111827")
        fig2.patch.set_facecolor("#111827")
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        ax2.set_xlabel("Importance Score")
        for i, v in enumerate(importance['Importance']):
            ax2.text(v + 0.005, i, f"{v:.3f}", color="white", va='center')
        st.pyplot(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Avg Water by Crop Type</div>", unsafe_allow_html=True)

        crop_avg = df.groupby("Crop_Type")["Water_Requirement"].mean().sort_values(ascending=True)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        bars = ax3.barh(crop_avg.index, crop_avg.values)
        for bar in bars:
            bar.set_color("#9ddc7c")
        bars[-1].set_color("#0d7a3a")

        ax3.set_facecolor("#111827")
        fig3.patch.set_facecolor("#111827")
        ax3.tick_params(colors='white')
        ax3.xaxis.label.set_color('white')
        ax3.yaxis.label.set_color('white')
        ax3.title.set_color('white')
        ax3.set_xlabel("Avg Water Requirement")
        st.pyplot(fig3)

        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    c5, c6 = st.columns(2)

    with c5:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Water by Weather Condition</div>", unsafe_allow_html=True)

        weather_avg = df.groupby("Weather_Condition")["Water_Requirement"].mean().sort_values(ascending=False)

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.bar(weather_avg.index, weather_avg.values, color="#22364d", edgecolor="#2ea043")
        ax4.set_facecolor("#111827")
        fig4.patch.set_facecolor("#111827")
        ax4.tick_params(colors='white', rotation=25)
        ax4.xaxis.label.set_color('white')
        ax4.yaxis.label.set_color('white')
        ax4.title.set_color('white')
        ax4.set_ylabel("Avg Water (L/day)")
        st.pyplot(fig4)

        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Water by Soil Type</div>", unsafe_allow_html=True)

        soil_avg = df.groupby("Soil_Type")["Water_Requirement"].mean().sort_values(ascending=False)

        fig5, ax5 = plt.subplots(figsize=(7, 4))
        ax5.bar(soil_avg.index, soil_avg.values, color="#22364d", edgecolor="#2ea043")
        ax5.set_facecolor("#111827")
        fig5.patch.set_facecolor("#111827")
        ax5.tick_params(colors='white', rotation=25)
        ax5.xaxis.label.set_color('white')
        ax5.yaxis.label.set_color('white')
        ax5.title.set_color('white')
        ax5.set_ylabel("Avg Water (L/day)")
        st.pyplot(fig5)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
<hr style="border:1px solid rgba(255,255,255,0.06);">
Built with Python & Streamlit · Smart Irrigation Internship Project
</div>
""", unsafe_allow_html=True)