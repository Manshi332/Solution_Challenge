import streamlit as st
import pandas as pd
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix       

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="FairFrame | AI Bias Auditor",
    layout="wide"
)

# 2. SIDEBAR NAVIGATION
st.sidebar.title(" FairFrame Control")
st.sidebar.info("Upload your dataset and model to begin the AI Ethics Audit.")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Technical Docs", "About Team"])

# 3. MAIN APP LOGIC
if menu == "Dashboard":
    st.title("FairFrame: Responsible AI Auditor")
    st.markdown("---")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Audit Configuration")
    audit_type = st.sidebar.radio(
            "Choose Strategy:", 
            ["Individual Deep Dive", "Audit All Groups"]
    )
    df, target_col, protected_col, loaded_model = m1.show_data_ui()

    if df is not None:
       
        st.success(f"Successfully loaded data. Auditing '{target_col}' based on '{protected_col}'.")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            if audit_type == "Individual Deep Dive":
                results = m2_audit.run_audit(df, target_col, protected_col, loaded_model)
            else:
                all_cols = [c for c in df.columns if c != target_col]
                results = m2_audit.run_audit_all(df, target_col, all_cols)
            st.divider()
            is_fixed, final_gap = m2_fix.apply_mitigation(df, target_col, protected_col, results)

        with col_right:
            st.subheader(" AI Auditor Insight")
          
           
elif menu == "Technical Docs":
    st.header("How FairFrame Works")
    st.write("This tool detects bias in datasets and models, then applies reweighing or post-processing fixes.")

else:
    st.header("The Team")
    st.write("Built by a team of 4 for the Hackathon.")

# 4. FOOTER
st.sidebar.markdown("---")
st.sidebar.write(" Powered by Gemini 1.5 Flash")
