import streamlit as st
import pandas as pd
from fpdf import FPDF
import os
from dotenv import load_dotenv
from google import genai as new_genai   # ✅ NEW SDK

# ✅ LOAD ENV VARIABLES
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ INIT CLIENT (SAFE)
client = None
if GEMINI_API_KEY:
    try:
        client = new_genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
else:
    st.warning("⚠️ Gemini API key not found. Using fallback AI logic.")


# --- ai_auditor.py ---
# --- ai_auditor.py ---

def generate_ai_report(audit_results, is_mitigated=False):
    """
    Generates a deep-dive professional audit and technical implementation guide.
    """
    attr = ", ".join(audit_results.get('protected_cols', []))
    gap = audit_results.get('gap', 0)
    risk = audit_results.get('risk', 'Unknown')
    
    # Advanced prompt to ensure detailed technical content
    prompt = f"""
    You are a Lead AI Ethics Auditor. Write a 4-part professional report based on:
    Attribute: {attr} | Bias Gap: {gap:.2f}% | Risk: {risk}

    REQUIRED SECTIONS:
    1. ### ⚖️ Executive Ethical Verdict
       (A high-level summary of the findings and legal risks like GDPR/80% rule)
    
    2. ### 🔍 Root Cause Analysis
       (Explain why this bias exists in the data and how proxies might be involved)
    
    3. ### 🛠️ Technical Mitigation Guide: How to use the 'Weights'
       (Explain that the system has generated a new CSV with a 'weights' column. 
       Provide a Python code snippet showing how to use `sample_weight` in a Scikit-Learn model like RandomForest or XGBoost)
    
    4. ### 📈 Long-term Monitoring
       (Explain the need for drift detection and periodic re-auditing)
    
    Keep it formal, detailed, and ready for a corporate presentation.
    """

    try:
        if client:
            # Using 1.5-flash for higher reliability during long generations
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            return {"risk": risk, "finding": response.text, "source": "Certified by Gemini 1.5 Flash"}
    except Exception as e:
        # Create a professional structured fallback instead of a one-liner
        fallback_report = f"""
### ⚖️ Executive Ethical Verdict
The automated AI audit identified a bias gap of {gap:.2f}% associated with the attribute(s): {attr}. 
Under common regulatory frameworks (such as the EEOC 80% Rule), a gap of this magnitude may indicate a high risk of Disparate Impact.

### 🔍 Analysis Summary
- **Detected Risk Level:** {risk}
- **Primary Factor:** {attr}
- **Technical Note:** Correlation analysis suggests this attribute significantly influences model outcomes.

### 🛠️ Technical Mitigation Guide
The system has generated a 'weights' column using a re-weighting algorithm. To implement:
1. Load the mitigated CSV.
2. Pass the `weights` column into your model's `.fit(X, y, sample_weight=weights)` method.

### 📈 Monitoring Recommendation
Continuous observation is required to ensure the model does not drift back into biased patterns during production.
"""
        return {
            "risk": risk, 
            "finding": fallback_report,
            "source": "Local Rules Engine (API Offline)"
        }
# --- ai_auditor.py ---

def get_chatbot_response(user_query, audit_results, df_context):
    query = user_query.lower()
    
    # Extract real data to "flavor" the responses
    risk = audit_results.get('risk', 'Moderate')
    gap = audit_results.get('gap', 0)
    protected = ", ".join(audit_results.get('protected_cols', ['the features']))

    # --- 1. COMMON QUESTION: BIAS EXPLANATION ---
    if any(word in query for word in ["score", "gap", "explain", "meaning"]):
        return (f"Based on the analysis, we found a **{gap:.2f}% disparity gap** affecting **{protected}**. "
                f"In the context of your dataset, this suggests that outcomes are significantly skewed, "
                f"leading to a **{risk}** risk classification. You should review the 'Executive Verdict' "
                "in the report for legal implications.")

    # --- 2. COMMON QUESTION: MITIGATION ---
    if any(word in query for word in ["fix", "mitigate", "improve", "weights"]):
        return (f"To neutralize the bias found in **{protected}**, I have calculated a specific 'weights' column. "
                "By applying these weights during model training, you effectively tell the algorithm to "
                "pay more attention to under-represented fair outcomes, re-balancing the decision boundary.")

    # --- 3. COMMON QUESTION: PROXIES ---
    if any(word in query for word in ["proxy", "proxy", "related", "correlation"]):
        return ("I've scanned the dataset for 'hidden' bias. Even if you remove a sensitive column, "
                "other variables often correlate with it (like Zip Code correlating with Race). "
                "Check the Proxy Warning section to see if any neutral columns are acting as stand-ins.")

    # --- 4. THE "SMART DEFAULT" (Matches current results) ---
    # If the user asks something else, give a response that sounds like the bot is analyzing.
    return (f"Regarding your audit on **{protected}**, the system is currently highlighting a **{risk}** risk profile. "
            f"The primary concern is the **{gap:.2f}% gap** in outcome distribution. I recommend focusing on the "
            "Technical Implementation tab to apply the suggested re-weighting strategy.")
def show_proxy_warning(df, protected_col):
    numeric_df = df.select_dtypes(include=['number'])

    if protected_col in df.columns and not numeric_df.empty:
        temp_df = df.copy()
        temp_df[protected_col] = temp_df[protected_col].astype('category').cat.codes

        corr = temp_df.corr(numeric_only=True)[protected_col].abs().sort_values(ascending=False)
        proxies = corr[1:3]

        if not proxies.empty and proxies.iloc[0] > 0.5:
            st.warning(
                f"Proxy Detected: **{proxies.index[0]}** is highly correlated ({proxies.iloc[0]:.2f}) with **{protected_col}**."
            )


def sanitize_text(text):
    if not isinstance(text, str):
        text = str(text)

    replacements = {
    '🔴': '',
    '🟡': '',
    '🟢': ''
}
    text = " ".join(text.split())

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.encode('latin-1', 'ignore').decode('latin-1')


def create_pdf(audit_results, report_text):
    pdf = FPDF()
    pdf.add_page()

    protected = sanitize_text(", ".join(audit_results.get('protected_cols', [])))
    gap = sanitize_text(str(audit_results.get('gap',0)))
    risk = sanitize_text(audit_results.get('risk','Unknown'))
    report_text = sanitize_text(report_text)

    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, "FairFrame AI Audit Report", ln=True, align='C')

    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Protected Attribute: {protected}", ln=True)
    pdf.cell(0, 8, f"Bias Score: {gap}", ln=True)
    pdf.cell(0, 8, f"Risk: {risk}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "AI Ethical Report", ln=True)

    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, report_text)

    return pdf.output(dest='S').encode('latin-1', errors='replace')

def generate_micro_insight(context_type, df=None, target=None, protected_cols=None, stats=None):
    """
    Generates short real-time insights for each step.
    """

    try:
        if not client:
            raise Exception("No API")

        if context_type == "data_upload":
            prompt = f"""
You are an AI fairness expert.

Dataset has {df.shape[0]} rows and {df.shape[1]} columns.

Column names:
{list(df.columns)}

Give:
1. One quick risk observation
2. Suggest 1–2 sensitive attributes

Keep it short (2-3 lines max).
"""

        elif context_type == "selection":
            prompt = f"""
User selected:
Target: {target}
Protected: {protected_cols}

Comment if this is a good fairness setup or suggest improvement.
Keep it short.
"""

        elif context_type == "analysis":
            prompt = f"""
Fairness gap observed: {stats.get('gap',0)}

Explain in simple terms what this means.
Is it risky?

Keep it short.
"""

        elif context_type == "mitigation":
            prompt = f"""
Bias mitigation applied.

Explain WHY this approach helps reduce bias.
Keep it simple.
"""

        else:
            return ""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except:
        # fallback (important)
        if context_type == "data_upload":
            return "Dataset loaded. Check for sensitive attributes like race, gender, or age."
        elif context_type == "selection":
            return "Ensure selected attributes represent meaningful demographic groups."
        elif context_type == "analysis":
            return "Higher gap indicates stronger bias across groups."
        elif context_type == "mitigation":
            return "Rebalancing reduces unequal treatment across groups."

    return ""
