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

# 2. SIDEBAR NAVIGATION
st.sidebar.title("🚀 FairFrame Control")
st.sidebar.info("Upload your dataset to begin the AI Ethics Audit.")
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
            
            # M3: Mitigation Engine
            # We pass 'results' to ensure the fixer knows the gap and attribute
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