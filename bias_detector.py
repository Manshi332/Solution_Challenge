import streamlit as st
import pandas as pd
import numpy as np

def get_task_info(df, target):
    unique_count = df[target].nunique()
    is_classification = unique_count <= 5
    return is_classification

# --- OPTION 1: INDIVIDUAL AUDIT  ---
def run_audit(df, target, protected, model=None):
    st.markdown(f" Deep Dive Audit: {protected}")
    is_classification = get_task_info(df, target)
    
    # 1. Dataset Bias Logic
    if is_classification:
        st.info(" Task: Classification Detected")
        stats = df.groupby(protected)[target].mean() * 100
        gap = stats.max() - stats.min()
        metric_label = "Fairness Gap"
        metric_value = f"{gap:.1f}%"
        is_high_risk = gap > 15
        detail_text = f"The difference in success rates between groups is **{gap:.1f}%**."
    else:
        st.info(" Task: Numerical/Regression Detected")
        group_stats = df.groupby(protected)[target].mean()
        ratio = group_stats.max() / group_stats.min() if group_stats.min() != 0 else 0
        gap = ratio * 10 # Normalize for fixer logic
        metric_label = "Disparity Ratio"
        metric_value = f"{ratio:.2f}x"
        is_high_risk = ratio > 1.5
        detail_text = f"The highest group receives **{ratio:.1f}x** more on average."

    # 2. Model Bias
    model_biased = False
    discrepancy_count = 0
    if model is not None:
        try:
            test_data = df.drop(columns=[target]).head(10)
            # Ensure we don't pass helper columns if they exist
            test_data = test_data[[c for c in test_data.columns if c not in ['is_success', 'sample_weight']]]
            
            preds_orig = model.predict(test_data)
            test_data_flipped = test_data.copy()
            groups = df[protected].unique()
            if len(groups) >= 2:
                test_data_flipped[protected] = test_data_flipped[protected].map({groups[0]: groups[1], groups[1]: groups[0]})
                preds_flipped = model.predict(test_data_flipped)
                discrepancy_count = np.sum(preds_orig != preds_flipped)
                model_biased = discrepancy_count > 0
        except Exception as e:
            st.warning(f"Model audit skipped: {e}")

    # 3. Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric(metric_label, metric_value, 
                  delta="- High Risk" if is_high_risk else "Fair", delta_color="inverse")
        st.write(detail_text)
    
    with col2:
        if model:
            status = " Biased" if model_biased else "Stable"
            st.metric("Model Logic Status", status, delta=f"{discrepancy_count} flips" if model_biased else "Consistent")
        else:
            st.metric("Model Status", "Not Provided")

    return {"gap": gap, "protected_col": protected, "model_biased": model_biased, "is_classification": is_classification}

# --- OPTION 2: AUDIT ALL  ---
def run_audit_all(df, target, all_potential_cols):
    st.markdown("### 🌍 Global Dataset Scan")

    is_classification = df[target].nunique() <= 5
    st.write(f"Scanning attributes against **{target}** ({'Classification' if is_classification else 'Numerical'})...")
    
    scan_results = []
    
    for col in all_potential_cols:

        if col in [target, 'is_success', 'sample_weight']: 
            continue

        if df[col].nunique() > 20: 
            continue

        if df[col].nunique() == len(df):
            continue

        try:
            group_stats = df.groupby(col)[target].mean()
            
            if is_classification:
                score = (group_stats.max() - group_stats.min()) * 100
                metric_name = "Fairness Gap (%)"
                risk_threshold = 15 
            else:
                score = group_stats.max() / group_stats.min() if group_stats.min() != 0 else 0
                metric_name = "Disparity Ratio (x)"
                risk_threshold = 1.5 

            scan_results.append({
                "Attribute": col,
                metric_name: round(score, 2),
                "Risk Level": "🔴 High Risk" if score > risk_threshold else "🟢 Low Risk"
            })
        except Exception as e:
            continue 

    # 4. Display Results
    if scan_results:
        results_df = pd.DataFrame(scan_results)

        metric_col = results_df.columns[1]
        results_df = results_df.sort_values(by=metric_col, ascending=False)
        
        st.table(results_df)

        worst = results_df.iloc[0]
        st.warning(f" **Recommendation:** The attribute **{worst['Attribute']}** shows the highest disparity. Consider targeting this for mitigation.")
        
        return {
            "gap": worst[metric_col], 
            "protected_col": worst['Attribute'], 
            "is_classification": is_classification
        }
    else:
        st.error("No valid categorical attributes found for auditing.")
        return None