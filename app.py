import streamlit as st
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix
import ai_auditor as ai 

st.set_page_config(page_title="FairFrame Pro | AI Auditor", page_icon="⚖️", layout="wide")

st.sidebar.title("⚖️ FairFrame Pro")
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
            if "results" in st.session_state:

                m2_audit.show_proxy_warning(df, protected_cols[0])

                with st.spinner("Generating professional AI report..."):
                    insight = ai.generate_ai_report(st.session_state.results)

                st.markdown("## 📜 Professional AI Audit Report")

                with st.container(border=True):
                    st.markdown(insight["finding"])

                st.subheader("Risk Level")
                st.write(insight["risk"])

                # ✅ ONLY ONE PDF BUTTON
                pdf_data = ai.create_pdf(st.session_state.results, insight["finding"])

                st.download_button(
                    label="📥 Download Professional Report",
                    data=pdf_data,
                    file_name="FairFrame_Report.pdf",
                    mime="application/pdf"
                )

                st.success("✅ This system ensures fairness-aware decision making using AI-driven auditing and mitigation.")