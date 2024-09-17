import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to load data from the Excel file
@st.cache
def load_data(file):
    xls = pd.ExcelFile(file)
    run_ids = pd.read_excel(xls, 'Run IDs')
    run_details = pd.read_excel(xls, 'Run Details')
    loss = pd.read_excel(xls, 'Loss')
    blimp = pd.read_excel(xls, 'BLIMP')
    reading_time = pd.read_excel(xls, 'Reading Time')
    return run_ids, run_details, loss, blimp, reading_time


# Function to join datasets
def join_data(run_details, loss, blimp, reading_time):
    try:
        merged_data = run_details.merge(loss, on='run_id', how='left')
        merged_data = merged_data.merge(blimp, on='run_id', how='left')
        merged_data = merged_data.merge(reading_time, on='run_id', how='left')
    except Exception as e:
        print(merged_data.columns)
        print("Exception: ", e)
    return merged_data

# Function to clean and select relevant columns for aggregation
def clean_data(df, columns):
    return df[columns]


def serve_blimp_results(curriculum_data_blimp, masking_data_blimp):

    #choice of blimp total average, blimp avg or individual blimp scores

    blimp_choice = st.selectbox("Select Choice of granularity", ["Total Average", "Group Average", "Individual Scores"])

    masking_data_blimp['x_category'] = masking_data_blimp.apply(
        lambda x: "No Masking" if x['mask_type'] == 'Non'
        else "Linear Decay" if x['mask_type'] == 'linear'
        else "Weak Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 0.5
        else "Intermediate Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 1
        else "Strong Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 2
        else "Stronger Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 3
        else "Strongest Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 4
        else "Weak Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 0.5
        else "Intermediate Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 1
        else "Strong Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 2
        else "Weak Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 0.5
        else "Intermediate Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 1
        else "No Category assigned", axis=1)

    masking_data_blimp["plot_color"] = masking_data_blimp.apply(lambda x: "gray" if x['mask_type'] == 'Non'
    else "blue" if x['mask_type'] == 'linear'
    else "green" if x['mask_type'] == 'exponential_new'
    else "red" if x['mask_type'] == 'logarithmic'
    else "yellow" if x['mask_type'] == 'sigmoid'
    else "black", axis=1)

    disp_masking_blimp = masking_data_blimp[
        masking_data_blimp['mask_type'].isin(['Non', 'linear', 'exponential_new', 'logarithmic', 'sigmoid'])]
    disp_masking_blimp = disp_masking_blimp[disp_masking_blimp['n_layer'] == 6]
    disp_masking_blimp["x_category"] = pd.Categorical(disp_masking_blimp["x_category"].isin(
        ["No Masking", "Linear Decay", "Weak Exponential Decay", "Intermediate Exponential Decay", "Strong Exponential Decay",
            "Stronger Exponential Decay", "Strongest Exponential Decay",
         "Weak Logarithmic Decay", "Intermediate Logarithmic Decay", "Strong Logarithmic Decay", "Weak Sigmoid Decay",
         "Intermediate Sigmoid Decay"]))
    disp_masking_blimp["x_category"] = pd.Categorical(disp_masking_blimp["x_category"],
                                                      categories=["No Masking", "Linear Decay",
                                                                  "Weak Exponential Decay",
                                                                  "Intermediate Exponential Decay",
                                                                  "Strong Exponential Decay",
                                                                    "Stronger Exponential Decay",
                                                                    "Strongest Exponential Decay",
                                                                  "Weak Logarithmic Decay",
                                                                  "Intermediate Logarithmic Decay",
                                                                  "Strong Logarithmic Decay", "Weak Sigmoid Decay",
                                                                  "Intermediate Sigmoid Decay"], ordered=True)
    disp_masking_blimp = disp_masking_blimp.sort_values('x_category')

    if blimp_choice == "Total Average":

        st.subheader("Total Average BLIMP Scores by Mask Type")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=disp_masking_blimp, x='x_category', y='total_avg', hue='echoic_memory')
        plt.xticks(rotation=90)
        #plt.ylim(60, 75)
        st.pyplot(plt)

    elif blimp_choice == "Group Average":
        st.subheader("Group Average BLIMP Scores by Mask Type")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=masking_data_blimp, x='x_category', y='blimp_avg', hue='echoic_memory')
        plt.xticks(rotation=90)
        plt.ylim(60,75)
        st.pyplot(plt)
def serve_loss_results(curriculum_data_loss, masking_data_loss):

    print("available columns: ", masking_data_loss.columns)

    #3 types of comparison
    #1. Effect of Size
    #Effect of Echoic Memory
    #Effect of Mask Type

    #print(type of different mask types)
    #print("Mask Types: ", masking_data_loss['mask_type'].unique())
    #print("Mask Decay Rates: ", masking_data_loss['mask_decay_rate'].unique())
    #print(masking_data_loss)

    st.subheader("Effect of Mask Type and strength on Loss")

    st.header("Masking Experiments Analysis - Loss")
    #grouped_masking_loss = masking_data_loss.groupby(['n_layer', 'n_head', 'mask_type']).mean().reset_index()
    #Filter for echoic memory = 1,
    #Filter for mask_type = None, linear, exponential_new, logarithmic, sigmoid
    #Filter for n_layer = 6

    #Create new column for x axis category where -
    # Mask Type = None means X_category is "No Masking"
    # Mask Type = linear means X_category is "Linear Decay"
    # Mask Type = exponential_new and mask_decay_rate = 0.5 means X_category is "Weak Exponential Decay"
    # Mask Type = exponential_new and mask_decay_rate = 1 means X_category is "Intermediate Exponential Decay"
    # Mask Type = exponential_new and mask_decay_rate = 2 means X_category is "Strong Exponential Decay"
    # Mask Type = logarithmic and mask_decay_rate = 0.5 means X_category is "Weak Logarithmic Decay"
    # Mask Type = logarithmic and mask_decay_rate = 1 means X_category is "Intermediate Logarithmic Decay"
    # Mask Type = logarithmic and mask_decay_rate = 2 means X_category is "Strong Logarithmic Decay"
    # Mask Type = sigmoid and mask_decay_rate = 0.5 means X_category is "Weak Sigmoid Decay"
    # Mask Type = sigmoid and mask_decay_rate = 1 means X_category is "Intermediate Sigmoid Decay"

    masking_data_loss['x_category'] = masking_data_loss.apply(
        lambda x: "No Masking" if x['mask_type'] == 'Non'
        else "Linear Decay" if x['mask_type'] == 'linear'
        else "Weak Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 0.5
        else "Intermediate Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 1
        else "Strong Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 2
        else "Stronger Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 3
        else "Strongest Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 4
        else "Weak Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 0.5
        else "Intermediate Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 1
        else "Strong Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 2
        else "Weak Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 0.5
        else "Intermediate Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 1
        else "No Category assigned", axis=1)

    masking_data_loss["plot_color"] = masking_data_loss.apply(lambda x: "gray" if x['mask_type'] == 'Non'
                                                                else "blue" if x['mask_type'] == 'linear'
                                                                else "green" if x['mask_type'] == 'exponential_new'
                                                                else "red" if x['mask_type'] == 'logarithmic'
                                                                else "yellow" if x['mask_type'] == 'sigmoid'
                                                                else "black", axis=1)



    #print("Masking Loss: ", masking_data_loss)
    #disp_masking_loss = masking_data_loss[masking_data_loss['echoic_memory'] == 5]
    #print("Masking Loss 1: ", disp_masking_loss)
    disp_masking_loss = masking_data_loss[masking_data_loss['mask_type'].isin(['Non', 'linear', 'exponential_new', 'logarithmic', 'sigmoid'])]

    disp_masking_loss = disp_masking_loss[disp_masking_loss['n_layer'] == 6]

    #x_category in order ["No Masking", "Linear Decay",
    # "Weak Exponential Decay", "Intermediate Exponential Decay", "Strong Exponential Decay",
    # "Weak Logarithmic Decay", "Intermediate Logarithmic Decay", "Strong Logarithmic Decay",
    # "Weak Sigmoid Decay", "Intermediate Sigmoid Decay"]
    #Filter only for the above categories
    disp_masking_loss = disp_masking_loss[disp_masking_loss['x_category'].isin(["No Masking", "Linear Decay",
                                                                                "Weak Exponential Decay", "Intermediate Exponential Decay",
                                                                                "Strong Exponential Decay",
                                                                                "Stronger Exponential Decay", "Strongest Exponential Decay",
                                                                                "Weak Logarithmic Decay", "Intermediate Logarithmic Decay", "Strong Logarithmic Decay", "Weak Sigmoid Decay", "Intermediate Sigmoid Decay"])]
    disp_masking_loss['x_category'] = pd.Categorical(disp_masking_loss['x_category'], categories=["No Masking", "Linear Decay",
                                                                                                  "Weak Exponential Decay", "Intermediate Exponential Decay",
                                                                                                  "Strong Exponential Decay", "Stronger Exponential Decay", "Strongest Exponential Decay",

                                                                                                  "Weak Logarithmic Decay", "Intermediate Logarithmic Decay", "Strong Logarithmic Decay", "Weak Sigmoid Decay", "Intermediate Sigmoid Decay"], ordered=True)
    disp_masking_loss = disp_masking_loss.sort_values('x_category')
    print(disp_masking_loss)

    st.subheader("Training Loss for different Mask Types")
    #Plot with x axis as x category and y axis as train loss with multiple bars per each x category
    plt.figure(figsize=(10, 6))
    sns.barplot(data=disp_masking_loss, x='x_category', y='train_loss', hue='echoic_memory')
    #make x axis labels vertical
    plt.xticks(rotation=90)
    plt.ylim(2.75,3)

    st.pyplot(plt)

    st.subheader("Validation Loss for different Mask Types")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=disp_masking_loss, x='x_category', y='val_loss', hue='echoic_memory')
    #make x axis labels vertical
    plt.xticks(rotation=90)
    plt.ylim(3.5,4.2)
    st.pyplot(plt)



    st.header("Curriculum Learning Analysis - Loss")

    grouped_curriculum_loss = curriculum_data_loss.groupby(
        ['n_layer', 'n_head', 'curriculum_type']).mean().reset_index()

    st.subheader("Training Loss by Layers, Heads and Curriculum Type")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_curriculum_loss, x='n_layer', y='train_loss', hue='curriculum_type')
    st.pyplot(plt)

    st.subheader("Validation Loss by Layers, Heads and Curriculum Type")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_curriculum_loss, x='n_layer', y='val_loss', hue='curriculum_type')
    st.pyplot(plt)


def serve_rt_results(curriculum_data_reading_time, masking_data_reading_time):
    st.header("Curriculum Learning Analysis - Reading Time")
    grouped_curriculum_reading_time = curriculum_data_reading_time.groupby(
        ['n_layer', 'n_head', 'curriculum_type']).mean().reset_index()



    st.header("Masking Experiments Analysis - Reading Time")
    #grouped_masking_reading_time = masking_data_reading_time.groupby(
    #    ['n_layer', 'n_head', 'mask_type']).mean().reset_index()

    masking_data_reading_time['x_category'] = masking_data_reading_time.apply( lambda x: "No Masking" if x['mask_type'] == 'Non'
        else "Linear Decay" if x['mask_type'] == 'linear'
        else "Weak Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 0.5
        else "Intermediate Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 1
        else "Strong Exponential Decay" if x['mask_type'] == 'exponential_new' and x['mask_decay_rate'] == 2
        else "Weak Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 0.5
        else "Intermediate Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 1
        else "Strong Logarithmic Decay" if x['mask_type'] == 'logarithmic' and x['mask_decay_rate'] == 2
        else "Weak Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 0.5
        else "Intermediate Sigmoid Decay" if x['mask_type'] == 'sigmoid' and x['mask_decay_rate'] == 1
        else "No Category assigned", axis=1)

    disp_masking_reading_time = masking_data_reading_time[masking_data_reading_time['mask_type'].isin(['Non', 'linear', 'exponential_new', 'logarithmic', 'sigmoid'])]
    disp_masking_reading_time = disp_masking_reading_time[disp_masking_reading_time['n_layer'] == 6]
    disp_masking_reading_time["x_category"] = pd.Categorical(disp_masking_reading_time["x_category"], categories=["No Masking", "Linear Decay", "Weak Exponential Decay", "Intermediate Exponential Decay", "Strong Exponential Decay", "Weak Logarithmic Decay", "Intermediate Logarithmic Decay", "Strong Logarithmic Decay", "Weak Sigmoid Decay", "Intermediate Sigmoid Decay"], ordered=True)
    disp_masking_reading_time = disp_masking_reading_time.sort_values('x_category')

    st.subheader("Correlation Surprisal by Mask Type")
    plt.figure(figsize=(10, 6))
    #sns.barplot(data=grouped_masking_reading_time, x='n_layer', y='corr_surprisal', hue='mask_type')
    sns.barplot(data=disp_masking_reading_time, x='x_category', y='corr_surprisal', hue='echoic_memory')
    plt.xticks(rotation=90)
    plt.ylim(0.06,0.14)


    st.pyplot(plt)


    st.subheader("Correlation Surprisal by Layers, Heads and Curriculum Type")
    plt.figure(figsize=(10, 6))
    #sns.barplot(data=grouped_curriculum_reading_time, x='n_layer', y='corr_surprisal', hue='curriculum_type')
    #st.pyplot(plt)


# Streamlit app
st.title("Model Run Analysis")
#Choice of file upload or use default file
default_file = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/rundata.xlsx'
use_default = st.selectbox("Use Default File?", [False, True])
if use_default:
    uploaded_file = default_file
else:
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    run_ids, run_details, loss, blimp, reading_time = load_data(uploaded_file)
    st.success("Data loaded successfully!")

    # Separate data into curriculum learning and masking experiments
    curriculum_data = run_details[run_details['curriculum_learning'] == True]
    masking_data = run_details[run_details['curriculum_learning'] == False]

    # Join data
    curriculum_data = join_data(curriculum_data, loss, blimp, reading_time)
    masking_data = join_data(masking_data, loss, blimp, reading_time)

    #BLIMP relevant columns - anaphor_agreement	argument_structure	binding	control_raising	determiner_noun_agreement	ellipsis	filler_gap	irregular_forms	island_effects	npi_licensing	quantifiers	subject_verb_agreement	hypernym	qa_congruence_easy	qa_congruence_tricky	subject_aux_inversion	turn_taking	blimp_avg	supplement_avg	total_avg

    #Loss relevant columns - train_loss, val_loss
    #Reading Time relevant columns - corr_surprisal

    #Filter Columns common - n_layer, n_head (maybe combine both into one)
    #Echoic Memory - echoic_memory
    #Masking - mask_type, mask_decay_rate
    #Curriculum Learning - curriculum_type

    #Make 3 possible dataframes - one for BLIMP, one for Loss, one for Reading Time

    # Clean and select relevant columns for aggregation for blimp, loss and reading time, with run_id, common filter columns, echoic memory and curriculum or mask type depending on the experiment

    curriculum_columns_blimp = ['run_id', 'n_layer', 'n_head', 'curriculum_type', 'echoic_memory', 'anaphor_agreement', 'argument_structure', 'binding', 'control_raising', 'determiner_noun_agreement', 'ellipsis', 'filler_gap', 'irregular_forms', 'island_effects', 'npi_licensing', 'quantifiers', 'subject_verb_agreement', 'hypernym', 'qa_congruence_easy', 'qa_congruence_tricky', 'subject_aux_inversion', 'turn_taking', 'blimp_avg', 'supplement_avg', 'total_avg']
    masking_columns_blimp = ['run_id', 'n_layer', 'n_head', 'mask_type', 'mask_decay_rate', 'echoic_memory', 'anaphor_agreement', 'argument_structure', 'binding', 'control_raising', 'determiner_noun_agreement', 'ellipsis', 'filler_gap', 'irregular_forms', 'island_effects', 'npi_licensing', 'quantifiers', 'subject_verb_agreement', 'hypernym', 'qa_congruence_easy', 'qa_congruence_tricky', 'subject_aux_inversion', 'turn_taking', 'blimp_avg', 'supplement_avg', 'total_avg']

    curriculum_columns_loss = ['run_id', 'n_layer', 'n_head', 'curriculum_type', 'echoic_memory', 'train_loss', 'val_loss']
    masking_columns_loss = ['run_id', 'n_layer', 'n_head', 'mask_type', 'mask_decay_rate', 'echoic_memory', 'train_loss', 'val_loss']

    curriculum_columns_reading_time = ['run_id', 'n_layer', 'n_head', 'curriculum_type', 'echoic_memory', 'corr_surprisal']
    masking_columns_reading_time = ['run_id', 'n_layer', 'n_head', 'mask_type', 'mask_decay_rate', 'echoic_memory', 'corr_surprisal']

    # Clean and select relevant columns
    curriculum_data_blimp = clean_data(curriculum_data, curriculum_columns_blimp)
    masking_data_blimp = clean_data(masking_data, masking_columns_blimp)

    curriculum_data_loss = clean_data(curriculum_data, curriculum_columns_loss)
    masking_data_loss = clean_data(masking_data, masking_columns_loss)

    curriculum_data_reading_time = clean_data(curriculum_data, curriculum_columns_reading_time)
    masking_data_reading_time = clean_data(masking_data, masking_columns_reading_time)

    # Selection box for metric type
    metric_type = st.selectbox("Select Metric Type", ["BLIMP", "Loss", "Reading Time"])

    if metric_type == "BLIMP":
        serve_blimp_results(curriculum_data_blimp, masking_data_blimp)

    elif metric_type == "Loss":
        serve_loss_results(curriculum_data_loss, masking_data_loss)

    elif metric_type == "Reading Time":
        serve_rt_results(curriculum_data_reading_time, masking_data_reading_time)