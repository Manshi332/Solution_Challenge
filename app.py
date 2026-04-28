import streamlit as st
import data_handler as m1 
import bias_detector as m2_audit 
import bias_fixer as m2_fix
import ai_auditor as ai 
import about_team as team  
import technical_methodology as tech_method

st.set_page_config(page_title="FairFrame Pro | AI Auditor", page_icon="⚖️", layout="wide")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

st.sidebar.button(
    "🌙 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode", 
    on_click=toggle_theme
)

#theme
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
            [data-testid="stFileUploader"]{
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

#main dashboard
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

        #ai analysis 
        with st.container(border=True):
            st.markdown("🤖 AI Analysis Insight")
            with st.spinner("🤖 AI thinking..."):
                insight = ai.generate_micro_insight(
                    "analysis",
                    stats=st.session_state.results
                )
                st.info(insight)

        # --- SECTION 2: MITIGATION ENGINE ---
        if "results" in st.session_state:
            st.markdown("---")
            st.markdown("### Step 4: 🛠️ Bias Mitigation Engine")
            st.info("Applying mathematical corrections to rebalance model outcomes for protected groups.")
            
            with st.container(border=True):
                m2_fix.apply_mitigation(df, target_col, protected_cols, st.session_state.results)

        # 🤖 AI MITIGATION
        if "results" in st.session_state:
            with st.container(border=True):
                st.markdown("🤖 AI Mitigation Strategy")
                with st.spinner("🤖 AI thinking..."):
                    insight = ai.generate_micro_insight("mitigation")
                    st.info(insight)

        # --- app.py SECTION 3 ---


        if "results" in st.session_state:
            st.divider()
            st.subheader("📜 Step 5: Advanced AI Audit & Implementation Handbook")

            # Generate the deep-dive report
            with st.spinner("📑 Synthesizing executive report and technical snippets..."):
                report_data = ai.generate_ai_report(st.session_state.results)

            # UI: TABS FOR PROFESSIONAL HIERARCHY
            tab1, tab2 = st.tabs(["📄 Professional Report", "💻 Technical Implementation"])

            with tab1:
                with st.container(border=True):
                    st.markdown(report_data["finding"])
                    
                    # Export Section at bottom of report
                    pdf_data = ai.create_pdf(st.session_state.results, report_data["finding"])
                    st.download_button(
                        label="📥 Download Audit Certificate (PDF)",
                        data=pdf_data,
                        file_name="FairFrame_Audit_Report.pdf",  # Explicitly name the file
                        mime="application/pdf",                  # Explicitly set the MIME type
                        use_container_width=True
                    )

            with tab2:
                st.info("💡 Use the following code to apply the fairness weights from your generated CSV.")
                
                # FIX: Get the protected attribute name from session state
                target_col = st.session_state.get('target_col', 'target_column')
                # Get the first protected column selected by the user
                protected_attr = st.session_state.get('protected_cols', ['protected_column'])[0]

                st.code(f"""
            # 1. Load your newly generated "Fairness-Aware" CSV
            import pandas as pd
            from sklearn.ensemble import RandomForestClassifier

            df_new = pd.read_csv("fair_mitigated_data.csv")

            # 2. Extract features, target, and the NEW weight column
            # We drop the target, the weights, and the sensitive attribute itself
            X = df_new.drop(['{target_col}', 'weights', '{protected_attr}'], axis=1)
            y = df_new['{target_col}']
            w = df_new['weights']  # <-- Generated by FairFrame

            # 3. Train the model using the weights to neutralize bias
            model = RandomForestClassifier()
            model.fit(X, y, sample_weight=w) 

            print("✅ Model trained successfully with Fairness-Aware Re-weighting!")
                """, language="python")
                
                st.success(f"The 'weights' column ensures that the bias found in '{protected_attr}' is neutralized during training.")

elif menu == "Technical Methodology":
    tech_method.show_technical_methodology()

#about the team page
elif menu == "About Team": 
    team.show_about_team()
# --- 3.5 SIDEBAR CHATBOT (NEW LOCATION) ---
with st.sidebar:
    st.divider()
    st.markdown("### 💬 Ethics Strategy Chat")
    
    if "results" in st.session_state:
        # Style these as "Suggested Questions"
        st.caption("Suggested Questions:")
        if st.button("📊 Explain the Bias Score", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What does this bias score mean?"})
            response = ai.get_chatbot_response("score", st.session_state.results, "")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        if st.button("🛠️ How do I fix this?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How do I fix this bias?"})
            response = ai.get_chatbot_response("fix", st.session_state.results, "")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            
        if st.button("🕵️ Check for Proxies", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Are there hidden proxies?"})
            response = ai.get_chatbot_response("proxy", st.session_state.results, "")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # ... your chat history loop ...

        # Chat History Container (scrollable)
        chat_container = st.container(height=300)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input
        chat_input = st.chat_input("Ask the Auditor...")
        final_prompt = chat_input or st.session_state.get("temp_prompt")

        if final_prompt:
            st.session_state.temp_prompt = None
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            
            # Generate AI Response
            df_ctx = f"Columns: {list(df.columns)}" if 'df' in locals() else "Data not loaded"
            response = ai.get_chatbot_response(final_prompt, st.session_state.results, df_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun() # Refresh to show new messages in the sidebar
    else:
        st.warning("Please upload data to enable the Ethics Chat.")

# 4. FOOTER
st.sidebar.markdown("---")
st.sidebar.write("🚀 Powered by Gemini 1.5 Flash")
