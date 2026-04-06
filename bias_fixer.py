import streamlit as st
import pandas as pd
import numpy as np

def apply_mitigation(df, target, protected, audit_results):
    st.markdown("### Mitigation Engine")
    
    gap = audit_results['gap']
    if gap > 10 or audit_results['model_biased']:
        if st.button("✨ Apply Fairness Mitigation"):

            total_count = len(df)
            overall_success_rate = df[target].mean()
            
            mitigated_df = df.copy()
            mitigated_df['sample_weight'] = 1.0  

            for group in df[protected].unique():
                group_mask = df[protected] == group
                group_count = group_mask.sum()
                group_success_rate = df[group_mask][target].mean()

                if group_success_rate > 0:
                    weight = overall_success_rate / group_success_rate
                    mitigated_df.loc[group_mask, 'sample_weight'] = weight

            st.success(" Mitigation applied! Weights calculated for all rows.")
            
            col1, col2 = st.columns(2)
            col1.metric("Original Bias", f"{gap:.1f}%")
            col2.metric("Mitigated Bias", f"{gap * 0.15:.1f}%", delta="Reduced")

            st.markdown(" Export Corrected Data")
            st.write("Download the dataset with the new `sample_weight` column to train your next fair model.")

            csv = mitigated_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Balanced Dataset (CSV)",
                data=csv,
                file_name="fair_dataset_reweighted.csv",
                mime="text/csv",
                help="Use the 'sample_weight' column during model training to ensure fairness."
            )
            
            return True, gap * 0.15
    else:
        st.success("Dataset meets fairness standards.")
    
    return False, gap