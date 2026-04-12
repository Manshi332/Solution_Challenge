import streamlit as st
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix
import ai_auditor as ai 
import about_team as team  

st.set_page_config(page_title="FairFrame Pro | AI Auditor", page_icon="⚖️", layout="wide")

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
menu = st.sidebar.radio("Navigate", ["Audit Dashboard", "Technical Methodology", "About Team"])

if menu == "Audit Dashboard":
    st.title("Responsible AI Audit Dashboard")
    st.markdown("---")
    
    df, target_col, protected_cols, model = m1.show_data_ui()

    if df is not None and target_col and protected_cols:
       # --- SECTION 1: BIAS DETECTION ---
        st.markdown("---")
        st.markdown("### Step 3: 📊 Bias Detection Analysis") 
        st.info("The system is now scanning all features for potential bias patterns and calculating disparity scores.")
        
        all_pot = [c for c in df.columns if c != target_col]
        m2_audit.run_audit_all(df, target_col, all_pot)
        
        st.divider()
        st.session_state.results = m2_audit.run_audit(df, target_col, protected_cols)

        # --- SECTION 2: MITIGATION ENGINE ---
        if "results" in st.session_state:
            st.markdown("---")
            st.markdown("### Step 4: 🛠️ Bias Mitigation Engine")
            st.info("Applying mathematical corrections to rebalance model outcomes for protected groups.")
            
            with st.container(border=True):
                m2_fix.apply_mitigation(df, target_col, protected_cols, st.session_state.results)

        # --- SECTION 3: PROFESSIONAL REPORT ---
        if "results" in st.session_state:
            st.markdown("---")
            st.markdown("### Step 5: 📜 Professional Audit Report")
            
            m2_audit.show_proxy_warning(df, protected_cols[0])

            with st.spinner("Analyzing findings and generating professional report..."):
                insight = ai.generate_ai_report(st.session_state.results)

            with st.container(border=True):
                st.markdown("#### 🤖 AI Auditor Findings")
                st.markdown(insight["finding"])
                
                col_r1, col_r2 = st.columns([1, 3])
                with col_r1:
                    st.metric("Risk Level", insight["risk"])
                with col_r2:
                    st.success("Analysis complete. Document ready for export.")

            # Full-width download button for a professional finish
            pdf_data = ai.create_pdf(st.session_state.results, insight["finding"])
            st.download_button(
                label="📥 Download Professional Report (PDF)",
                data=pdf_data,
                file_name="FairFrame_Audit_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.success("✅ This system ensures fairness-aware decision making using AI-driven auditing and mitigation.")
            
elif menu == "Technical Methodology":
    tech_method.show_technical_methodology()
    
elif menu == "About Team": 
    team.show_about_team()

# 4. FOOTER
st.sidebar.markdown("---")
st.sidebar.write("🚀 Powered by Gemini 1.5 Flash")
