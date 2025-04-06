import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import text

st.title('Data Management Webapp')
st.sidebar.title('Options')
st.sidebar.markdown('This page shows the results for the data')

# #blimp_results = st.Page("pages/blimp_results.py", title="BLIMP Results")
# #pg = st.navigation([blimp_results])

# results_xlsx_path = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/rundata.xlsx'
# results_df = pd.read_excel(results_xlsx_path, sheet_name='Loss')
# run_details_df = pd.read_excel(results_xlsx_path, sheet_name='Run Details')
# #st.write(results_df)

# st.dataframe(results_df.merge(run_details_df, on='run_id', how="right")[["run_id", "train_loss", "val_loss", "mask_type", "mask_decay_rate", "curriculum_type"]])

#Things to do:

#Model List - Show list of models 
#Model Details - Show details of the model - Check if database has details for given model

model_management = st.Page("pages/model_management.py", title="Model Management")
story_management = st.Page("pages/story_management.py", title="Story Management")
model_compare_delta = st.page("pages/model_compare_delta.py", title="Training Curve Comparison Delta")

pg = st.navigation([model_management, story_management, model_compare_delta])


pg.run()









