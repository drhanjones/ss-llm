#%%
library(DBI)
library(lme4)

RESULTS_DB_PATH <- "/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/results.db"
results_db <- dbConnect(RSQLite::SQLite(), RESULTS_DB_PATH)


#%%


#Notes
# 2 types of modelling approaches - Individual subject level fit (OSh Approach) vs Average subject level fit (Wilcox)


# Steps to do  - 

# Sanity Checks - start with simple model use only character length as predictor and/or Word frequencies
# Step 2 - Add more predictors - POS tags, Word Categories
# Baseline - LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID) + (1 | POSTag)
# Baseline - LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID) + (1 | WordCategoryID)



# Step 3 - Delta Loglikelihood with Surprisal as predictor
# Oh and Schuler Eq -
# Baseline - LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID) + (1 | POSTag)
# Model - LogRT ~ CharacterLength_c + LogFrequencies_c + SurprisalScore_c + (1 | WorkerID) + (1 | POSTag)

# Wilcox et al Eq -
# Baseline -  LogRT ~ CharacterLength_c + prev_len_c + prev2_len_c + prev3_len_c + LogFrequencies_c + prev_freq_c + prev2_freq_c + prev3_freq_c
# Model - LogRT ~ CharacterLength_c + prev_len_c + prev2_len_c + prev3_len_c + LogFrequencies_c + prev_freq_c + prev2_freq_c + prev3_freq_c + SurprisalScore_c + prev_surp_c + prev2_surp_c + prev3_surp_c

#%%

#%%

#Individual Subject Level Fit
baseline_df_ind = dbGetQuery(results_db, "
SELECT RTDundeeCorpus.RTUID,
RTDundeeCorpus.WorkerID, 
RTDundeeCorpus.StoryWordID, 
RTDundeeCorpus.GazeDuration, 
RTDundeeCorpus.IgnoreRow,
WordDetails.Word as WordCategory, 
WordDetails.CharacterLength, 
WordDetails.WordUID as WordCategoryID,
WordDetails.LogFrequencies as LogFrequencies,
Story.POSTag as POSTag
FROM RTDundeeCorpus 
JOIN Story on RTDundeeCorpus.StoryWordID = Story.StoryWordID 
JOIN WordDetails on WordDetails.WordUID = Story.WordUID 
WHERE RTDundeeCorpus.IgnoreRow == 0
ORDER BY WorkerID
")

#Convert to log scale
baseline_df_ind$LogRT <- log(baseline_df_ind$GazeDuration)

baseline_df_ind$WorkerID <- as.factor(baseline_df_ind$WorkerID)
baseline_df_ind$WordCategoryID <- as.factor(baseline_df_ind$WordCategoryID)
baseline_df_ind$POSTag <- as.factor(baseline_df_ind$POSTag)

baseline_df_ind$CharacterLength_c <- scale(baseline_df_ind$CharacterLength)
baseline_df_ind$LogFrequencies_c <- scale(baseline_df_ind$LogFrequencies)

#%%

#Average Subject Level Fit - Uses Previous 2 words length and frequency as predictors (along with surprisal)

# baseline_df_avg = dbGetQuery(results_db, "
# SELECT RTDundeeCorpus.RTUID,
# RTDundeeCorpus.WorkerID, 
# RTDundeeCorpus.StoryWordID, 
# RTDundeeCorpus.GazeDuration, 
# RTDundeeCorpus.IgnoreRow,
# WordDetails.Word as WordCategory, 
# WordDetails.CharacterLength, 
# WordDetails.WordUID as WordCategoryID,
# WordDetails.LogFrequencies as LogFrequencies,
# Story.POSTag as POSTag
# FROM RTDundeeCorpus 
# JOIN Story on RTDundeeCorpus.StoryWordID = Story.StoryWordID 
# JOIN WordDetails on WordDetails.WordUID = Story.WordUID 
# WHERE RTDundeeCorpus.IgnoreRow == 0
# ORDER BY WorkerID
# ")

#%%

#Simple Model - LogRT ~ CharacterLength_c 

simple_model_1 <- lmer(LogRT ~ CharacterLength_c + (1 | WorkerID), data = baseline_df_ind)
summary(simple_model_1)
logLik(simple_model_1)

simple_model_2 <- lmer(LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID), data = baseline_df_ind)
summary(simple_model_2)
logLik(simple_model_2)

simple_model_3 <- lmer(LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID) + (1 | POSTag), data = baseline_df_ind)
summary(simple_model_3)
logLik(simple_model_3)

#Result - LogLik decreases as we add more predictors - so the model is getting better
#Other sanity checks - As character length increases, RT increases
#As LogFrequencies increases, RT decreases 
#%%

# Data with model 
RESULTS_DB_PATH <- "/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/results.db"
results_db <- dbConnect(RSQLite::SQLite(), RESULTS_DB_PATH)

baseline_df_wsurp <- dbGetQuery(results_db,
"WITH FilteredModel AS (
SELECT StoryWordID, SurprisalScore
FROM ModelSurprisalScores
WHERE ModelID = 8465733
)
SELECT RTDundeeCorpus.RTUID, 
RTDundeeCorpus.WorkerID, 
RTDundeeCorpus.StoryWordID, 
RTDundeeCorpus.GazeDuration,

WordDetails.Word as WordCategory, 
WordDetails.CharacterLength, 
WordDetails.WordUID as WordCategoryID,
WordDetails.LogFrequencies as LogFrequencies,

Story.POSTag as POSTag,

FilteredModel.SurprisalScore as SurprisalScore

FROM RTDundeeCorpus
JOIN Story on RTDundeeCorpus.StoryWordID = Story.StoryWordID 
JOIN WordDetails on WordDetails.WordUID = Story.WordUID 
JOIN FilteredModel on FilteredModel.StoryWordID = RTDundeeCorpus.StoryWordID
WHERE RTDundeeCorpus.IgnoreRow=0
    
")


