import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_proxy_warning(df, protected_col):
    """
    Detects if other columns are acting as proxies for the protected attribute.
    """
    numeric_df = df.select_dtypes(include=['number'])

    if protected_col in df.columns and not numeric_df.empty:
        temp_df = df.copy()

        # Convert categorical to numeric codes
        temp_df[protected_col] = temp_df[protected_col].astype('category').cat.codes

        # Correlation check
        corr = temp_df.corr(numeric_only=True)[protected_col].abs().sort_values(ascending=False)
        proxies = corr[1:3]

        if not proxies.empty and proxies.iloc[0] > 0.5:
            st.warning(
                f"⚠️ Proxy Detected: **{proxies.index[0]}** is highly correlated ({proxies.iloc[0]:.2f}) with "
                f"**{protected_col}**. Removing the sensitive attribute alone will NOT fix bias!"
            )


def get_task_info(df, target):
    """Determines if the task is classification or regression."""
    return df[target].nunique() <= 5


def run_audit(df, target, protected_cols, model=None):
    st.markdown("### 🔍 Intersectional Deep Dive")

    is_classification = get_task_info(df, target)

    df_temp = df.copy()

    # Ensure target is numeric
    if df_temp[target].dtype == 'object':
        df_temp[target] = pd.factorize(df_temp[target])[0]

    # Create intersectional groups
    df_temp['Demographic Groups'] = df_temp[protected_cols].astype(str).agg(' & '.join, axis=1)

    group_stats = df_temp.groupby('Demographic Groups')[target].mean().sort_values()

    # 📊 Plot
    st.write("#### Success Rate by Intersectional Group")

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        '#ff4b4b' if (x == group_stats.min() or x == group_stats.max()) else '#0078ff'
        for x in group_stats
    ]

    group_stats.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Mean Outcome (Success Rate)")

    st.pyplot(fig)

    # ✅ Save for PDF
    fig.savefig("bias_plot.png", bbox_inches='tight')

    # 📉 Bias Calculation
    if is_classification:
        gap = (group_stats.max() - group_stats.min()) * 100
        risk_level = "🔴 HIGH RISK" if gap > 15 else "🟡 MODERATE" if gap > 5 else "🟢 LOW RISK"
        display_gap = f"{gap:.2f}%"
    else:
        gap = group_stats.max() / (group_stats.min() + 1e-6)
        risk_level = "🔴 HIGH RISK" if gap > 1.5 else "🟡 MODERATE" if gap > 1.2 else "🟢 LOW RISK"
        display_gap = f"{gap:.2f}x"

    # 📌 Metrics
    col1, col2 = st.columns(2)
    col1.metric("Bias Score", display_gap)
    col2.metric("Risk Status", risk_level)

    return {
        "gap": gap,
        "protected_cols": protected_cols,
        "is_classification": is_classification,
        "stats": group_stats,
        "target": target,
        "risk": risk_level
    }


def run_audit_all(df, target, all_cols):
    """AUTOMATED GLOBAL SCAN: Finds bias across every column."""
    st.markdown("#### 📊 Global Attribute Risk Scan")

    temp_df = df.copy()

    # Ensure target numeric
    if temp_df[target].dtype == 'object':
        temp_df[target] = pd.factorize(temp_df[target])[0]

    scan_results = []
    is_classification = get_task_info(df, target)

    for col in all_cols:
        if temp_df[col].nunique() > 20 or temp_df[col].nunique() < 2:
            continue

        try:
            group_stats = temp_df.groupby(col)[target].mean().dropna()
            if group_stats.empty:
                continue

            if is_classification:
                score = (group_stats.max() - group_stats.min()) * 100
                risk = "🔴 High" if score > 15 else "🟡 Med" if score > 5 else "🟢 Low"
            else:
                score = group_stats.max() / (group_stats.min() + 1e-6)
                risk = "🔴 High" if score > 1.5 else "🟡 Med" if score > 1.2 else "🟢 Low"

            scan_results.append({
                "Attribute": col,
                "Fairness Score": round(abs(score), 2),
                "Risk Level": risk
            })

        except:
            continue

    if scan_results:
        results_df = pd.DataFrame(scan_results)
        results_df = results_df.sort_values(by="Fairness Score", ascending=False)
        st.table(results_df)
    else:
        st.info("No categorical columns detected for automated scanning.")


def get_gemini_insight(results):
    """Generates structured data for the Report tab and PDF."""
    if not results:
        return None

    unit = "%" if results['is_classification'] else "x"

    return {
        "summary": f"Audit performed on '{results['target']}' across {len(results['protected_cols'])} attributes.",
        "finding": f"The maximum observed disparity is {results['gap']:.2f}{unit}.",
        "risk": results['risk'],
        "recommendation": "Apply reweighing (Kamiran-Calders) to balance group outcomes and reduce bias."
    }