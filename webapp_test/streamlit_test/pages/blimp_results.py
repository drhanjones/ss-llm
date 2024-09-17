import streamlit as st
import pandas as pd

st.title('Results for BLIMP data')
st.sidebar.title('Options')
st.sidebar.markdown('This page shows the results for the BLIMP data')


results_xlsx_path = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/rundata.xlsx'
run_details_df = pd.read_excel(results_xlsx_path, sheet_name='Run Details')
blimp_results_df = pd.read_excel(results_xlsx_path, sheet_name='BLIMP')

blimp_columns = list(blimp_results_df.columns)
blimp_columns = [col for col in blimp_columns if col not in ["run_id", "blimp_exists"]]
print(blimp_columns)
st.dataframe(run_details_df.merge(blimp_results_df, on='run_id', how="right")[["run_id", "n_head", "n_layer",
                                                                               "mask_type", "mask_decay_rate",
                                                                               "curriculum_type", *blimp_columns]])


