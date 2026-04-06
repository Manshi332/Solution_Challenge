import streamlit as st
import pandas as pd
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix        
import ai_auditor as m4_ai # Member 3/4 logic

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FairFrame | AI Bias Auditor",
    page_icon="⚖️",
    layout="wide"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.button(
    "🌙 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode", 
    on_click=toggle_theme
)

if st.session_state.dark_mode:
    st.markdown("""
        <style>
            /* 1. Fix the Header and Main App Background */
            header[data-testid="stHeader"] { background-color: #0e1117 !important; }
            .stApp { background-color: #0e1117; color: #ffffff; }

            /* 2. FIX BUTTON VISIBILITY (Toggle and others) */
            .stButton>button {
                background-color: #21262d !important;
                color: #ffffff !important;
                border: 1px solid #30363d !important;
                width: 100%;
            }
            /* Explicitly define Hover and Active states */
            .stButton>button:hover {
                border-color: #8b949e !important;
                color: #ffffff !important;
                background-color: #30363d !important;
            }
            .stButton>button:active {
                background-color: #21262d !important;
                color: #ffffff !important;
            }

            /* 3. FIX FILE UPLOADER (The 'Browse Files' text) */
            [data-testid="stFileUploader"] {
                background-color: #161b22;
                border: 2px dashed #30363d;
                padding: 1rem;
                border-radius: 10px;
            }
            /* Target the 'Browse files' button text and 'Drag and drop' label */
            [data-testid="stFileUploader"] section button {
                background-color: #21262d !important;
                color: #ffffff !important;
            }
            [data-testid="stFileUploader"] label, 
            [data-testid="stFileUploader"] p, 
            [data-testid="stFileUploader"] small {
                color: #ffffff !important;
            }

            /* 4. Sidebar Consistency */
            [data-testid="stSidebar"] {
                background-color: #0d1117 !important;
                border-right: 1px solid #30363d;
            }
            [data-testid="stSidebar"] .stMarkdown, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] p {
                color: #ffffff !important;
            }

            /* 5. Overall Text and Metrics */
            h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    
    st.markdown("""
        <style>
            header[data-testid="stHeader"] { background-color: #ffffff !important; }
            [data-testid="stFileUploader"] { background-color: #f0f2f6; border: 2px dashed #ced4da; }
        </style>
    """, unsafe_allow_html=True)

# 2. SIDEBAR NAVIGATION
st.sidebar.title("FairFrame Control")
st.sidebar.info("Upload your dataset and model to begin the AI Ethics Audit.")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Technical Docs", "About Team"])

# 3. MAIN APP LOGIC
if menu == "Dashboard":
    st.title("⚖️ FairFrame: Responsible AI Auditor")
    st.markdown("---")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Audit Configuration")
    audit_type = st.sidebar.radio(
            "Choose Strategy:", 
            ["Individual Deep Dive", "Audit All Groups"]
    )
    
    # Initialize loaded_model as None (Add model uploader in data_handler if needed)
    loaded_model = None 

    # M1: Data Ingestion
    df, target_col, protected_col = m1.show_data_ui()

    if df is not None:
        st.success(f"✅ Successfully loaded data. Auditing '{target_col}' based on '{protected_col}'.")

        # Layout: Audit on Left, AI Insights on Right
        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            # M2: Bias Detection
            if audit_type == "Individual Deep Dive":
                results = m2_audit.run_audit(df, target_col, protected_col, loaded_model)
            else:
                all_cols = [c for c in df.columns if c != target_col]
                results = m2_audit.run_audit_all(df, target_col, all_cols)
            
            st.divider()
            # Call Member 2 (Fixer) logic
            is_fixed, final_gap = m2_fix.apply_mitigation(df, target_col, protected_col, results)

        with col_right:
            st.subheader("🤖 AI Auditor Insight")
            
            if results:
                # M4: AI Interpretability (Gemini Report)
                # Show report for the initial audit
                m4_ai.generate_ai_report(results, is_mitigated=False)
                
                # Check for hidden proxies (e.g., Zip Code acting as Race)
                m4_ai.show_proxy_warning(df, protected_col)
                
                # If the user applied mitigation, show the 'Post-Fix' report
                if is_fixed:
                    st.divider()
                    st.markdown("#### ✨ Post-Mitigation Analysis")
                    results['gap'] = final_gap
                    m4_ai.generate_ai_report(results, is_mitigated=True)
            else:
                st.info("Waiting for audit results to generate AI insights...")

elif menu == "Technical Docs":
    st.header("📘 How FairFrame Works")
    st.write("""
    FairFrame uses a multi-member architecture to ensure AI accountability:
    - **Data Handler:** Performs automated cleaning and imputation.
    - **Bias Detector:** Calculates Disparity Ratios and Fairness Gaps.
    - **Mitigation Engine:** Uses Statistical Reweighing to balance datasets.
    - **AI Auditor:** Leverages Gemini 1.5 Flash to provide ethical context and proxy detection.
    """)

else:
    st.header("👥 The Team")
    st.write("Built by a team of 4 dedicated to making AI transparent and fair.")
    st.markdown("- **Member 1:** Data Ingestion Specialist")
    st.markdown("- **Member 2:** Statistical Bias Researcher")
    st.markdown("- **Member 3:** AI Integration & Mitigation Engineer")
    st.markdown("- **Member 4:** Ethical Auditor & UX Designer")

# 4. FOOTER
st.sidebar.markdown("---")
st.sidebar.write("✨ Powered by Gemini 1.5 Flash")
