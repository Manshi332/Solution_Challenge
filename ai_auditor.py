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
    except Exception:
        return {
            "risk": risk, 
            "finding": "### ⚖️ Executive Summary\nManual Review Required. Technical documentation unavailable.",
            "source": "Fallback Engine"
        }

# --- ai_auditor.py ---

def get_chatbot_response(user_query, audit_results, df_context):
    """
    Expert Assistant logic with specific instructions for CSV handling.
    """
    # Normalize query for quick checks
    query = user_query.lower()
    
    # 1. Direct Handle for "How to use new CSV"
    if "new csv" in query or "upload" in query or "change file" in query:
        return (
            "To use a **new CSV file**, simply go to the **Left Sidebar** and click the 'Browse files' button. "
            "Once you upload a new file, FairFrame will automatically reset and start a fresh audit for that data. "
            "Pro-tip: Ensure your new CSV has similar column names if you want to compare results!"
        )

    # 2. Dynamic Gemini Logic for everything else
    prompt = f"""
    You are the FairFrame Ethics Assistant. 
    Audit Results: {audit_results.get('risk')} risk, {audit_results.get('gap', 0):.2f}% bias gap.
    Dataset Context: {df_context}
    
    User asked: "{user_query}"
    
    Task: Provide a sharp, professional response (max 3 sentences). 
    If they ask about specific columns, explain their potential impact on fairness.
    """
    
    try:
        if client:
            response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            return response.text
    except Exception:
        return "I'm currently focused on the current audit. Feel free to ask about the bias score or column risks!"
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
    Context-aware dynamic insights (NOT hardcoded)
    """

    try:
        if not client:
            raise Exception("No API")

        if context_type == "data_upload":
            prompt = f"""
You are an AI fairness auditor.

Dataset shape: {df.shape}
Columns: {list(df.columns)}

Identify:
- 1 potential proxy risk column
- 1 good protected attribute suggestion

Be specific. No generic statements.
Max 3 lines.
"""

        elif context_type == "selection":
            prompt = f"""
User selected:
Target: {target}
Protected: {protected_cols}

Evaluate:
- Is this a valid fairness setup?
- Any better suggestion?

Be precise and practical.
"""

        elif context_type == "analysis":
            gap = stats.get('gap', 0)
            risk = stats.get('risk', 'Unknown')
            attrs = stats.get('protected_cols', [])

            prompt = f"""
Fairness audit results:

Protected attributes: {attrs}
Bias gap: {gap:.2f}
Risk level: {risk}

Explain:
- What this means in real-world terms
- Whether it violates fairness standards (like 80% rule)

Be sharp and professional (2–3 lines).
"""

        elif context_type == "mitigation":
            gap = stats.get('gap', 0)

            prompt = f"""
Bias mitigation applied using reweighing.

Original bias gap: {gap:.2f}

Explain:
- Why reweighing is appropriate here
- What improvement it achieves

Be concise and technical.
"""

        else:
            return ""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    except:
        # fallback (still contextual)
        if context_type == "analysis" and stats:
            gap = stats.get("gap", 0)
            if gap > 15:
                return "High disparity detected. Likely unfair across groups."
            elif gap > 5:
                return "Moderate disparity. Needs monitoring."
            else:
                return "Low disparity. System is relatively fair."

        return "AI insight unavailable. Please check configuration."