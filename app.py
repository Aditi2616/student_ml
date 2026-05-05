import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="StudentMate : Performance Predictor", layout="wide")

# Custom CSS for Dark Mode and enhanced UI
st.markdown("""
<style>
    /* --- HIDE STREAMLIT BRANDING & CLOUD BADGES --- */
    footer {visibility: hidden !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stToolbar"] a {display: none !important;} 
    [data-testid="manage-app-button"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    
    /* Remove huge top padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    .stApp {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 { 
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Metrics box */
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        border: 1px solid var(--primary-color);
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: var(--text-color);
    }
    
    .grade-A { color: #00FF88; font-size: 3.5rem; font-weight: 800; line-height: 1; margin-top: 10px; display: block; text-shadow: 0 0 2px rgba(0,0,0,0.1); }
    .grade-B { color: #00D5FF; font-size: 3.5rem; font-weight: 800; line-height: 1; margin-top: 10px; display: block; text-shadow: 0 0 2px rgba(0,0,0,0.1); }
    .grade-C { color: #FFB300; font-size: 3.5rem; font-weight: 800; line-height: 1; margin-top: 10px; display: block; text-shadow: 0 0 2px rgba(0,0,0,0.1); }
    .grade-D { color: #FF3366; font-size: 3.5rem; font-weight: 800; line-height: 1; margin-top: 10px; display: block; text-shadow: 0 0 2px rgba(0,0,0,0.1); }
    
    /* Insight boxes */
    .warning-box {
        background: rgba(255, 51, 102, 0.15);
        border-left: 4px solid #FF3366;
        padding: 16px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .success-box {
        background: rgba(0, 255, 136, 0.15);
        border-left: 4px solid #00FF88;
        padding: 16px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .info-box {
        background: rgba(0, 213, 255, 0.15);
        border-left: 4px solid #00D5FF;
        padding: 16px;
        border-radius: 0 4px 4px 0;
        margin-bottom: 16px;
        font-size: 0.95rem;
    }

    /* Buttons Overrides */
    div.stDownloadButton > button {
        border: 1px solid var(--primary-color);
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    /* File Uploader override to make it pop */
    section[data-testid="stFileUploadDropzone"] {
        background-color: var(--secondary-background-color) !important;
        border: 2px dashed var(--primary-color) !important;
        border-radius: 8px !important;
        opacity: 1 !important;
        padding: 2rem !important;
    }
    section[data-testid="stFileUploadDropzone"] * {
        color: var(--text-color) !important;
        opacity: 1 !important;
    }
    section[data-testid="stFileUploadDropzone"] button {
        border: 1px solid var(--primary-color) !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Spacing between elements */
    .stSlider { margin-bottom: 1.5rem; }
    .stSelectbox { margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
FI_PATH = os.path.join(BASE_DIR, 'feature_importance.csv')

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error("Model file (model.pkl) not found. Please train the model first.")
        return None

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv(FI_PATH)
    except Exception:
        return None

model = load_model()
fi_df = load_feature_importance()

# Helper Functions
def get_grade(score):
    if score >= 85: return "A+", "grade-A"
    elif score >= 70: return "B", "grade-B"
    elif score >= 50: return "C", "grade-C"
    else: return "D", "grade-D"

def get_insights(study, attendance, stress, motivation):
    insights = []
    if study < 3.0:
        insights.append('<div class="warning-box"><strong>Warning:</strong> Low study hours detected. Dedicating more time to studying can significantly boost scores.</div>')
    elif study > 8.0:
        insights.append('<div class="success-box"><strong>Great:</strong> High study hours demonstrate excellent dedication.</div>')
        
    if attendance < 75.0:
        insights.append('<div class="warning-box"><strong>Warning:</strong> Attendance is concerning. Missing classes negatively impacts subject understanding.</div>')
        
    if stress == "High":
        insights.append('<div class="warning-box"><strong>Warning:</strong> High stress level observed. Consider mindfulness, breaks, or speaking to a counselor.</div>')
        
    if motivation == "Low":
        insights.append('<div class="warning-box"><strong>Tip:</strong> Low motivation can hinder progress. Try setting smaller, achievable goals to build momentum.</div>')
        
    if not insights:
        insights.append('<div class="success-box"><strong>On Track:</strong> Core metrics appear healthy. Keep up the good work.</div>')
        
    return "".join(insights)

# --- DATA ROBUSTNESS LOGIC ---
def process_batch_dataframe(df):
    """
    Cleans, standardizes, and imputes the uploaded dataframe
    to guarantee the ML model will not crash.
    """
    # 1. Standardize column names mapping
    col_mapping = {
        'studyhours': 'StudyHours',
        'study hours': 'StudyHours',
        'study_hours': 'StudyHours',
        'attendance': 'Attendance',
        'attendance %': 'Attendance',
        'attendance rate': 'Attendance',
        'assignments': 'Assignments',
        'assignment': 'Assignments',
        'hw score': 'Assignments',
        'homework': 'Assignments',
        'motivation': 'Motivation',
        'internetaccess': 'InternetAccess',
        'internet access': 'InternetAccess',
        'internet_access': 'InternetAccess',
        'internet': 'InternetAccess',
        'projects_hackathons': 'Projects_Hackathons',
        'projects': 'Projects_Hackathons',
        'hackathons': 'Projects_Hackathons',
        'projects and hackathons': 'Projects_Hackathons',
        'projects & hackathons': 'Projects_Hackathons',
        'stresslevel': 'StressLevel',
        'stress level': 'StressLevel',
        'stress_level': 'StressLevel',
        'stress': 'StressLevel',
        'age': 'Age'
    }
    
    # Rename columns flexibly based on map
    new_cols = {}
    for c in df.columns:
        c_clean = str(c).strip().lower()
        if c_clean in col_mapping:
            new_cols[c] = col_mapping[c_clean]
        else:
            new_cols[c] = c
    df = df.rename(columns=new_cols)
    
    # Define required model columns and their default safe fallback values
    required_cols = {
        'Age': 20,
        'StudyHours': 4.0,
        'Attendance': 70.0,
        'Assignments': 70.0,
        'Motivation': 'Medium',
        'InternetAccess': 'No',
        'Projects_Hackathons': 'No',
        'StressLevel': 'Medium'
    }
    
    # Validate if the user uploaded a completely unrelated CSV
    found_keys = [k for k in required_cols.keys() if k in df.columns]
    if len(found_keys) == 0:
        raise ValueError("We couldn't find any relevant data columns (like Study Hours, Attendance, etc.) in your CSV. Please make sure your file matches the Sample Template before uploading.")
        
    missing_handled = False
    
    # 2. Handle completely missing columns
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
            missing_handled = True
            
    # 3. Handle NaNs in existing columns
    for col, default_val in required_cols.items():
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                fill_val = float(mean_val) if not np.isnan(mean_val) else default_val
                df[col] = df[col].fillna(fill_val)
            else:
                df[col] = df[col].fillna(default_val)
            missing_handled = True
            
    return df, missing_handled


# --- MAIN APP UI ---
st.title("StudentMate AI: Performance Predictor")
st.markdown("Predict student exam outcomes using Machine Learning and derive actionable insights to maximize academic success.")

if model:
    tab1, tab2 = st.tabs(["Single Prediction Dashboard", "Batch Processing Upload"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col_gap, col2 = st.columns([1, 0.1, 1.5])
        
        with col1:
            st.markdown("### Student Profile")
            st.markdown("Adjust the features below to instantly see the predicted score.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            age = st.number_input("Age", min_value=10, max_value=40, value=20, step=1)
            study_hours = st.slider("Study Hours (Daily)", 0.0, 10.0, 4.5, 0.5)
            attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0, 1.0)
            assignments = st.slider("Assignment Completion (%)", 0.0, 100.0, 85.0, 1.0)
            motivation = st.select_slider("Motivation Level", options=["Low", "Medium", "High"], value="Medium")
            
            st.markdown("### Contextual Factors")
            internet = st.selectbox("Internet Access", ["Yes", "No"])
            projects = st.selectbox("Projects & Hackathons", ["Yes", "No"])
            stress = st.selectbox("Stress Level", ["Low", "Medium", "High"], index=1)
            
        with col2:
            st.markdown("### AI Prediction Results")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create a DataFrame for the single prediction
            input_data = pd.DataFrame({
                'Age': [age],
                'StudyHours': [study_hours],
                'Attendance': [attendance],
                'Assignments': [assignments],
                'Motivation': [motivation],
                'InternetAccess': [internet],
                'Projects_Hackathons': [projects],
                'StressLevel': [stress]
            })
            
            # Predict
            pred_score = model.predict(input_data)[0]
            pred_score = max(0, min(100, pred_score)) # clamp 0-100
            
            grade, grade_class = get_grade(pred_score)
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Predicted Score</h4>
                    <span style="font-size:3.5rem; font-weight:800; line-height:1; color:var(--text-color); margin-top:10px; display:block;">{pred_score:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
            with metrics_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Expected Letter Grade</h4>
                    <span class="{grade_class}">{grade}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### AI Driven Insights")
            insights_html = get_insights(study_hours, attendance, stress, motivation)
            st.markdown(insights_html, unsafe_allow_html=True)
            
            st.markdown("### Model Feature Impact")
            if fi_df is not None:
                fig = px.bar(fi_df.head(6), x='Importance', y='Feature', orientation='h',
                             title="Top Drivers of Exam Scores",
                             color='Importance',
                             color_continuous_scale="PuBuGn")
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: BATCH PREDICTION ---
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Bulk Processing with CSV")
        st.markdown("Upload a CSV file representing multiple students to generate predictions simultaneously. Missing values and generic column names are automatically handled.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Give them a sample CSV they can download
        sample_df = pd.DataFrame({
            'Age': [20, 19, 21],
            'StudyHours': [4.5, 8.0, 2.0],
            'Attendance': [85.0, 95.0, 60.0],
            'Assignments': [90.0, 100.0, 45.0],
            'Motivation': ['Medium', 'High', 'Low'],
            'InternetAccess': ['Yes', 'Yes', 'No'],
            'Projects_Hackathons': ['Yes', 'No', 'Yes'],
            'StressLevel': ['Medium', 'Low', 'High']
        })
        
        sample_csv = sample_df.to_csv(index=False).encode('utf-8')
        col_btn, col_rest = st.columns([1, 3])
        with col_btn:
            # Replaced emoji with explicit active phrasing
            st.download_button(
                label="Download Sample CSV Template",
                data=sample_csv,
                file_name="sample_student_template.csv",
                mime="text/csv",
                type="primary"
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Student Data JSON or CSV", type=["csv"])
        
        if uploaded_file is not None:
            raw_batch_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(raw_batch_data, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Run Batch Prediction", type="primary"):
                try:
                    # 1. Preprocess and safeguard the dataframe
                    processed_data, missing_handled = process_batch_dataframe(raw_batch_data.copy())
                    
                    if missing_handled:
                        st.markdown('<div class="info-box">Missing columns or values were automatically handled to ensure successful prediction.</div>', unsafe_allow_html=True)
                    
                    # 2. Extract strictly the features the model needs
                    model_features = ['Age', 'StudyHours', 'Attendance', 'Assignments', 'InternetAccess', 'Projects_Hackathons', 'StressLevel', 'Motivation']
                    pred_input = processed_data[model_features]
                    
                    # 3. Predict
                    preds = model.predict(pred_input)
                    preds = [max(0, min(100, p)) for p in preds] # Clamp array
                    
                    # 4. Attach generated results to original dataset structure to preserve user columns if any
                    results_data = processed_data.copy()
                    results_data.insert(0, 'Predicted_Grade', [get_grade(p)[0] for p in preds])
                    results_data.insert(0, 'Predicted_Score', [round(p, 1) for p in preds])
                    
                    st.success("Batch processing complete.")
                    st.dataframe(results_data, use_container_width=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Output download
                    output_csv = results_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results",
                        data=output_csv,
                        file_name="batch_predictions_results.csv",
                        mime="text/csv",
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
