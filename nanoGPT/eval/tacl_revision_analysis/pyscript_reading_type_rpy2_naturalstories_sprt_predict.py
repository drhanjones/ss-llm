from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import pandas as pd
import tqdm
import os
import sqlite3
import platform

importr("DBI")
importr("lme4")
importr("DT")

from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import conversion

import json
import argparse



parser = argparse.ArgumentParser(description="Fit surprisal models with specified equation pair")
parser.add_argument(
    "--equation_pair_index",
    type=int,
    help="Index of the equation pair to use (required)"
)
args = parser.parse_args()

# Check if the equation_pair_index is provided
if args.equation_pair_index is None:
    raise ValueError("equation_pair_index must be provided.")
elif not (args.equation_pair_index >= 0):
    raise ValueError("equation_pair_index must be between greater than or equal to 0.")
equation_pair_index = args.equation_pair_index

def rpy2py(x):
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.rpy2py(x)


def py2rpy(x):
    with localconverter(default_converter + pandas2ri.converter):
        return conversion.py2rpy(x)


if "Darwin" in platform.system():
    ROOT_PATH = "/Users/abishekthamma/Projects/Masters Thesis/ss-llm"
    RESULTS_DB_PATH = f"{ROOT_PATH}/nanoGPT/results/results.db"
elif "pop-os" in platform.node():
    raise NotImplementedError("This code is not intended to be run on pop-os machine yet.")
    ROOT_PATH = "/home/athamma1/Projects/ss-llm"
    RESULTS_DB_PATH = f"{ROOT_PATH}/nanoGPT/results/results.db"
elif "tux14psy" in platform.node():
    ROOT_PATH = "/home/athamma/Projects/ss-llm/ss-llm"
    RESULTS_DB_PATH = f"{ROOT_PATH}/nanoGPT/results/results.db"
elif "Linux" in platform.system():
    ROOT_PATH = "/gpfs/home5/athamma1/Projects/ss-llm"
    RESULTS_DB_PATH = f"{ROOT_PATH}/results.db"



def extract_model_summary(model_name, summary_name):
    # 1. Fixed Effects
    fixed_effects = ro.r(f"as.data.frame({summary_name}$coefficients)")
    fixed_effects_df = rpy2py(fixed_effects)
    # print(fixed_effects_df)
    fixed_effects_df.columns = ["Estimate", "Std. Error", "t value"]  # , 'Pr(>|t|)']

    # 2. Random Effects
    random_effects = ro.r(f"as.data.frame(VarCorr({model_name}))")
    random_effects_df = rpy2py(random_effects)
    # print("\nRandom Effects (Variance and Std. Dev by Group):\n", random_effects_df)

    # 3. Residuals
    residuals = ro.r(f"as.data.frame({summary_name}$residuals)")
    residuals_df = rpy2py(residuals)
    # print("\nResiduals:\n", residuals_df)

    # 4. Model Fit Statistics
    aic = ro.r(f"AIC({model_name})")[0]
    bic = ro.r(f"BIC({model_name})")[0]
    log_likelihood = ro.r(f"logLik({model_name})")[0]
    warnings = ro.r("warnings()")
    fit_stats_df = pd.DataFrame(
        {
            "AIC": [aic],
            "BIC": [bic],
            "Log-Likelihood": [log_likelihood],
            "Warnings": [warnings],
        }
    )

    # 5. Variance-Covariance Matrix of Random Effects
    var_cov_matrix = ro.r(f"as.data.frame({summary_name}$varcor)")
    var_cov_matrix_df = rpy2py(var_cov_matrix)

    return fixed_effects_df, random_effects_df, fit_stats_df, var_cov_matrix_df


def predict_rt(fit_name):
    ro.r(f'''
    baseline_df$PredictedLogRT <- predict({fit_name})
    ''')

    # # Extract the predicted RT values from the R data frame
    # predicted_rt = ro.r(f'as.data.frame(baseline_df)')
    
    #Column names of interest is "RTUID", "WorkerID", "StoryWordID", "LogRT", predicted_RT"

    ro.r('''predicted_rt <- baseline_df[, c("RTUID", "WorkerID", "StoryWordID", "LogRT", "PredictedLogRT")]''')

    #Print Head of the predicted RT values
    
    # Convert the predicted RT values to a pandas DataFrame
    predicted_rt_df = rpy2py(ro.r('predicted_rt'))
    
    return predicted_rt_df


def execute_r_lmer_model(fit_formula, data_frame_name, fit_name, fit_summary_name):
    ro.r(f"""
    {fit_name} <- lmer({fit_formula}, data={data_frame_name}, REML=F)
    {fit_summary_name} <- summary({fit_name})
    """)

    fixed_effects, random_effects, fit_stats, var_cov_matrix = extract_model_summary(
        fit_name, fit_summary_name
    )

    predict_rt_df = predict_rt(fit_name)
    return fixed_effects, random_effects, fit_stats, var_cov_matrix, predict_rt_df


# Given a model id, load the dataset for it in r, transform the data and run the model and return the results


# Replace SPRTNaturalStories with RTDundeeCorpus
def fit_surprisal_model(filter_model_id):
    ro.r(f'''
           RESULTS_DB_PATH <- "{RESULTS_DB_PATH}"
            results_db <- dbConnect(RSQLite::SQLite(), RESULTS_DB_PATH)

            baseline_df <- dbGetQuery(results_db,"
                                        WITH FilteredModel AS (
                                    SELECT StoryWordID, SurprisalScore
                                    FROM ModelSurprisalScores
                                    WHERE ModelID = {filter_model_id}
                                ),
                                RESULT_TABLE AS (
                                    SELECT SPRTNaturalStories.RTUID, 
                                        SPRTNaturalStories.WorkerID, 
                                        SPRTNaturalStories.StoryWordID, 
                                        SPRTNaturalStories.RT, 
                                        WordDetails.Word as WordCategory, 
                                        WordDetails.CharacterLength, 
                                        WordDetails.WordUID as WordCategoryID,
                                        WordDetails.LogFrequencies as LogFrequencies,
                                        Story.POSTag as POSTag,
                                        FilteredModel.SurprisalScore as SurprisalScore,

                                        LAG(WordDetails.CharacterLength, 1) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev_len,
                                        LAG(WordDetails.LogFrequencies, 1) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev_freq,
                                        LAG(FilteredModel.SurprisalScore, 1) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev_surp,

                                        LAG(WordDetails.CharacterLength, 2) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev2_len,
                                        LAG(WordDetails.LogFrequencies, 2) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev2_freq,
                                        LAG(FilteredModel.SurprisalScore, 2) OVER (
                                            PARTITION BY SPRTNaturalStories.WorkerID, Story.StoryID
                                            ORDER BY Story.WordID
                                        ) as prev2_surp


                                    FROM SPRTNaturalStories 
                                    JOIN Story on SPRTNaturalStories.StoryWordID = Story.StoryWordID 
                                    JOIN WordDetails on WordDetails.WordUID = Story.WordUID 
                                    JOIN FilteredModel on FilteredModel.StoryWordID = SPRTNaturalStories.StoryWordID
                                )
                                SELECT * FROM RESULT_TABLE
                                WHERE prev2_len IS NOT NULL;
            ")
            '''
            )
    
    
    ro.r('''
        baseline_df$LogRT <- log(baseline_df$RT)

        baseline_df$CharacterLength_c <- scale(baseline_df$CharacterLength)
        #Should I scale Log Frequencies(?)
        baseline_df$LogFrequencies_c <- scale(baseline_df$LogFrequencies)

        baseline_df$POSTag <- as.factor(baseline_df$POSTag)
        baseline_df$WorkerID <- as.factor(baseline_df$WorkerID)
        baseline_df$WordCategory <- as.factor(baseline_df$WordCategory)
        baseline_df$WordCategoryID <- as.factor(baseline_df$WordCategoryID)
        
         
        baseline_df$prev_len_c <- scale(baseline_df$prev_len)
        baseline_df$prev_freq_c <- scale(baseline_df$prev_freq)
        baseline_df$prev_surp_c <- scale(baseline_df$prev_surp)

        baseline_df$prev2_len_c <- scale(baseline_df$prev2_len)
        baseline_df$prev2_freq_c <- scale(baseline_df$prev2_freq)
        baseline_df$prev2_surp_c <- scale(baseline_df$prev2_surp)
         
        baseline_df$SurprisalScore_c <- scale(baseline_df$SurprisalScore)
        data_size <- nrow(baseline_df)
        ''')
    

    data_frame_name = "baseline_df"

    fit_name_m = "fit_model"
    fit_summary_name_m = "fit_model_summary"

    fixed_effects_m, random_effects_m, fit_stats_m, var_cov_matrix_m, predict_rt_df = (
        execute_r_lmer_model(
            fit_formula_m, "baseline_df", fit_name_m, fit_summary_name_m
        )
    )

    return fixed_effects_m, random_effects_m, fit_stats_m, var_cov_matrix_m, predict_rt_df


def compile_results_as_df(
    model_id,
    fixed_effects_df,
    random_effects_df,
    fit_stats_df,
    var_cov_matrix_df,
    fit_stats_b,
):
    data_size = ro.r("data_size")[0]
    results_df_row = {
        "ModelID": model_id,
        "condition_model_formula": fit_formula_m,
        "Log-Likelihood": fit_stats_df["Log-Likelihood"].values[0],
        "Coefficient for Surprisal Score": fixed_effects_df.loc[
            "SurprisalScore_c", "Estimate"
        ],
        "Delta Log-Likelihood": fit_stats_df["Log-Likelihood"].values[0] - fit_stats_b["Log-Likelihood"].values[0],
        "AIC": fit_stats_df["AIC"].values[0],
        "BIC": fit_stats_df["BIC"].values[0],
        "Data Size": data_size,
        "baseline_model_formula": fit_formula_b,
        "Fixed Effects": fixed_effects_df.to_html(),
        "Random Effects": random_effects_df.to_html(),
        "Variance-Covariance Matrix": var_cov_matrix_df.to_html(),
    }

    # results_df = pd.DataFrame(results_df_row, index=[0])

    return results_df_row


def save_json_lines_predict(
        model_id, save_jsonl, equation_pair_index=None
):
    # Read the existing CSV file


    if equation_pair_index is None:
        raise ValueError(
            "equation_pair_index must be provided when save_as is 'json'"
        )

    json_file_name = (
        f"{model_id}_equation_pair_{equation_pair_index}.jsonl"
    )

    os.makedirs(
        f"naturalstories_predict/models_results_equation_pair_{equation_pair_index}", exist_ok=True
    )

    json_file_path = os.path.join(
        "naturalstories_predict",
        f"models_results_equation_pair_{equation_pair_index}", json_file_name
    )
    with open(json_file_path, "w") as json_file:
        for entry in save_jsonl:
            json_file.write(json.dumps(entry) + "\n")

    

from pathlib import Path

def try_claim(work_id: str, lock_dir="claims_predict") -> bool:
    Path(lock_dir).mkdir(exist_ok=True)
    lock_path = Path(lock_dir) / f"{work_id}_eqid_{equation_pair_index}_naturalstories.lock"

    try:
        # O_EXCL makes creation atomic: fails if file already exists
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_claim(work_id: str, lock_dir="claims_predict") -> None:
    lock_path = Path(lock_dir) / f"{work_id}_eqid_{equation_pair_index}_naturalstories.lock"
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def create_connection_cursor(db_file):
    """
    Create a database connection to the SQLite database specified by the db_file

    Args:
        db_file (str): database file

    Returns:
        Connection object or None
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    return conn, c


# if "pop-os" in platform.node():
#     ROOT = r"/Users/abishekthamma/Projects/Masters Thesis/ss-llm/nanoGPT"
# else:
#     ROOT = r"/Users/abishekthamma/Projects/Masters Thesis/ss-llm/nanoGPT"

# RESULTS_ROOT = os.path.join(ROOT, "results")
# SQL_DB = os.path.join(RESULTS_ROOT, "results.db")

SQL_DB = RESULTS_DB_PATH

conn, c = create_connection_cursor(SQL_DB)

MODEL_LIST_QUERY = """
SELECT  DISTINCT ModelSurprisalScores.ModelID, Model.OutputFolderName, Model.BatchSize, Model.Dataset, Model.Seed, Model.MaskType 
FROM ModelSurprisalScores
JOIN Model ON ModelSurprisalScores.ModelID = Model.ModelID

WHERE 
((Model.NumLayers = 6 AND Model.MaskType="exponential_new" AND Model.MaskDecayRate=2 AND Model.EchoicMemory=10) 
OR (Model.NumLayers = 6 AND Model.MaskType="Non" AND Model.CurriculumLearning=0 ))
AND 
(Model.ModelID not in (5496427, 8456913, 5734459,  6617787))
AND Model.BatchSize=32
AND Model.Dataset in ("babylm_full_bpe_8k", "babylm_full_bpe_100M_8k")
ORDER BY Seed, BatchSize, Dataset, MaskType

"""

model_id_list = sorted(pd.read_sql_query(MODEL_LIST_QUERY, conn)["ModelID"].unique().tolist())

# model_id_list = model_id_list[:1]

print(len(model_id_list), model_id_list)
fit_stats_b = None


data_frame_name = "baseline_df"
fit_name_b = "fit_baseline"
fit_summary_name_b = "fit_summary_baseline"
fit_stats_b = None

# Below Block will be used for creating pairs of equations for fit_formula_b and fit_formula_m
# fit_formula_b = baseline
# fit_formula_m = condition_model

controls = "CharacterLength_c + LogFrequencies_c + prev_len_c + prev_freq_c + prev2_len_c + prev2_freq_c"
surprisal_terms = "SurprisalScore_c + prev_surp_c + prev2_surp_c"

model_pairs_list = [
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 | WorkerID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 | WorkerID)",
        "Notes": "Simplest model that only includes random intercept for WorkerID and looks at fixed effects of surprisal terms",
    },
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 | WorkerID)",
        "fit_formula_m": f"LogRT ~ {controls} + (1 + {surprisal_terms} | WorkerID)",
        "Notes": "Model that includes random slopes for surprisal terms by WorkerID, but doesn't look at global fixed effects of surprisal terms",
    },
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 | WorkerID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 + {surprisal_terms} | WorkerID)",
        "Notes": "Model that includes random slopes for surprisal terms by WorkerID, and also looks at global fixed effects of surprisal terms. Possibly the most correct model without item intercepts",
    },
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 | WorkerID) + (1 | WordCategoryID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 + {surprisal_terms} | WorkerID) + (1 | WordCategoryID)",
        "Notes": "Model that includes random intercepts for WordCategory, in addition to WorkerID, and looks at fixed effects of surprisal terms along with random slopes for surprisal terms by WorkerID. Although, I am not sure if this is a meaningful model because random intercepts for items mean controls become meaningless? Because what does it mean to have a separate intercept for each item and also control for length and frequency of the same item (and previous items, sure)? But after a lot of discussion, it seems to make sense and might be the best way to model this data.",
    },
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 + {controls} | WorkerID) + (1 | WordCategoryID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 + {controls} + {surprisal_terms} | WorkerID) + (1 | WordCategoryID)",
        "Notes": "Model that includes random intercepts for WordCategory, in addition to WorkerID, and looks at fixed effects of surprisal terms along with random slopes for surprisal terms and controls by WorkerID. Corresponding to the comment - all variables random slopes",
    },
    #
    #
    # Below is the pairs from after discussion and from mail (But without item intercepts)
    #
    # Pair 5
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 + {controls} | WorkerID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 + {controls} + {surprisal_terms} | WorkerID)",
        "Notes": "Most complicated model, Part 1",
    },
    # Pair 6
    {
        "fit_formula_b": f"LogRT ~ {controls} + (1 | WorkerID)",
        "fit_formula_m": f"LogRT ~ {controls} + {surprisal_terms} + (1 + {surprisal_terms} | WorkerID) ",
        "Notes": "Most complicated model, Part 2",
    },
    #
    #
    #   Same but with item intercepts
    #
    #
    # Pair 7
    {
        "fit_formula_b": f"LogRT ~ (1 | WorkerID) + (1 | WordCategoryID)",
        "fit_formula_m": f"LogRT ~ {surprisal_terms} + (1 + {surprisal_terms} | WorkerID) + (1 | WordCategoryID)",
        "Notes": "Secondary Check, item effect model",
    },
]

# equation_pair_index = 5  # Change this index to change the model pair being used

write_path = (
    f"surprisal_analysis_results_naturalstories_equation_pair_{equation_pair_index}.csv"
)

# results_df = pd.DataFrame()

fit_formula_b = model_pairs_list[equation_pair_index]["fit_formula_b"]
fit_formula_m = model_pairs_list[equation_pair_index]["fit_formula_m"]


print("Fitting baseline model...", fit_formula_b)
print("Number of models to fit:", fit_formula_m)


def verify_model_already_processed(model_id):

    #First check if the models is being processed by another process
    if not try_claim(model_id):
        return [model_id]
    

    json_file_path = os.path.join(
        "naturalstories_predict",
        f"models_results_equation_pair_{equation_pair_index}",
        f"{model_id}_equation_pair_{equation_pair_index}.json",
    )
    if os.path.exists(json_file_path):
        return [model_id]
    else:
        return []


for model_id in tqdm.tqdm(model_id_list):
    if model_id in verify_model_already_processed(model_id):
        continue
    try:
        fixed_effects_m, random_effects_m, fit_stats_m, var_cov_matrix_m, predict_rt_df = (
            fit_surprisal_model(model_id)
        )
        predict_rt_df["ModelID"] = model_id
        predict_rt_df = predict_rt_df[["ModelID", "RTUID", "WorkerID", "StoryWordID", "LogRT", "PredictedLogRT"]]
        predict_rt_df["EquationPairIndex"] = equation_pair_index

        predict_rt_df_json = predict_rt_df.to_dict(orient="records")

        save_json_lines_predict(
            model_id, predict_rt_df_json, equation_pair_index=equation_pair_index
        )        

        release_claim(model_id)
        break
    except Exception as e:
        print(f"Error for model id {model_id}: {e}")
        continue
