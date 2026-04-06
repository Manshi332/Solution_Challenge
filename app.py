import streamlit as st
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix
import ai_auditor as ai 
import about_team as team

st.set_page_config(page_title="FairFrame Pro | AI Auditor", page_icon="⚖️", layout="wide")

# Sidebar
st.sidebar.title("⚖️ FairFrame Pro")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigate", ["Audit Dashboard", "Technical Methodology", "About Team"])

if menu == "Audit Dashboard":
    st.title("Responsible AI Audit Dashboard")
    st.markdown("---")
    
    df, target_col, protected_cols, model = m1.show_data_ui()

    if df is not None and target_col and protected_cols:
        tab1, tab2, tab3 = st.tabs(["📊 Bias Detection", "🛠️ Mitigation Engine", "📜 Professional Report"])
        
        with tab1:
            all_pot = [c for c in df.columns if c != target_col]
            m2_audit.run_audit_all(df, target_col, all_pot)
            st.divider()
            st.session_state.results = m2_audit.run_audit(df, target_col, protected_cols)

        with tab2:
            if "results" in st.session_state:
                m2_fix.apply_mitigation(df, target_col, protected_cols, st.session_state.results)

        with tab3:
            if "results" in st.session_state and st.session_state.results:
                # Proxy Warning logic from your working snippet
                m2_audit.show_proxy_warning(df, protected_cols[0]) 
                
                with st.spinner("Gemini is analyzing ethics report..."):
                    # Calling the separated AI auditor module
                    insight = ai.generate_ai_report(st.session_state.results)
                
                st.subheader("Ethics Report Card")
                st.write(f"**Current Risk Level:** {insight['risk']}")
                st.info(f"**Expert Finding:**\n\n{insight['finding']}")
                
                # PDF Download using ai_auditor logic
                pdf_data = ai.create_pdf(st.session_state.results, insight)
                st.download_button(
                    label="📥 Download Official Audit PDF",
                    data=pdf_data,
                    file_name="FairFrame_Ethics_Report.pdf",
                    mime="application/pdf"
                )

elif menu == "About Team": 
    team.show_about_team()
