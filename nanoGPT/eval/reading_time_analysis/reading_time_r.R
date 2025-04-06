library(DBI)
library(lme4)

RESULTS_DB_PATH <- "/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/results.db"
results_db <- dbConnect(RSQLite::SQLite(), RESULTS_DB_PATH)

baseline_df <- dbGetQuery(results_db,"SELECT SPRTNaturalStories.RTUID, 
SPRTNaturalStories.WorkerID, 
SPRTNaturalStories.StoryWordID, 
SPRTNaturalStories.RT, 
WordDetails.Word as WordCategory, 
WordDetails.CharacterLength, 
WordDetails.WordUID as WordCategoryID,
WordDetails.LogFrequencies as LogFrequencies,
Story.POSTag as POSTag
FROM SPRTNaturalStories 
JOIN Story on SPRTNaturalStories.StoryWordID = Story.StoryWordID 
JOIN WordDetails on WordDetails.WordUID = Story.WordUID 
")

baseline_df$LogRT <- log(baseline_df$RT)

baseline_df$WorkerID <- as.factor(baseline_df$WorkerID)
baseline_df$WordCategoryID <- as.factor(baseline_df$WordCategoryID)
baseline_df$POSTag <- as.factor(baseline_df$POSTag)
baseline_df$CharacterLength_c <- scale(baseline_df$CharacterLength)

#Should I scale Log Frequencies(?)
baseline_df$LogFrequencies_c <- scale(baseline_df$LogFrequencies)
print.data.frame(baseline_df[1:5,])

exp_index <- sample(1:nrow(baseline_df), nrow(baseline_df)*0.5)

baseline_df$exp <- 0
baseline_df$exp[exp_index] <- 1
baseline_df$exp <- as.factor(baseline_df$exp)

exploratory_df <- baseline_df[baseline_df$exp == 1,]
test_df <- baseline_df[baseline_df$exp == 0,]






fit_formula <- "LogRT ~ 1  + ( 1 | WorkerID) + (1 | WordCategoryID)"
fit <- lmer(fit_formula, data=exploratory_df, REML=F)
summary(fit)

logLik(fit)



fit_formula_b <- "LogRT ~ 1  + ( 1 | WorkerID) + (1 | POSTag)"
fit_b <- lmer(fit_formula_b, data=exploratory_df, REML=F)
summary(fit_b)

logLik(fit_b)

fit_formula_2 <- "LogRT ~ CharacterLength_c + (1 + CharacterLength_c | WorkerID) + (1 | WordCategoryID)"    
fit_2 <- lmer(fit_formula_2, data=exploratory_df, REML=F)
summary(fit_2)


fit_formula_2b <-  "LogRT ~ CharacterLength_c + (1 + CharacterLength_c || WorkerID) + (1 | POSTag)" 
fit_2b <- lmer(fit_formula_2b, data=exploratory_df, REML=F)
summary(fit_2b)
logLik(fit_2b)




fit_formula_3 <- "LogRT ~ CharacterLength_c + (1 + CharacterLength_c | WorkerID) + (1 | POSTag) "
fit_3 <- lmer(fit_formula_3, data=exploratory_df, REML=F) # control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B'))
summary(fit_3)




# Start with random intercepts only
m1 <- lmer("LogRT ~ CharacterLength_c + (1 | WorkerID) + (1 | POSTag)", data = exploratory_df)
summary(m1)
logLik(m1)

# If m1 converges, add one random slope
m2 <- lmer("LogRT ~ CharacterLength_c + (1 + CharacterLength_c || WorkerID) + (1 | POSTag)", data = exploratory_df)
summary(m2)
logLik(m2)

# If m2 converges, try correlated random slopes
m3 <- lmer("LogRT ~ CharacterLength_c + (1 + CharacterLength_c | WorkerID) + (1 | WordCategoryID)", data = exploratory_df)
summary(m3)
logLik(m3)



# If all above converge, add second fixed effect
m4 <- lmer("LogRT ~ CharacterLength_c + LogFrequencies_c + (1 + CharacterLength_c | WorkerID) + (1 | WordCategoryID)", data = exploratory_df)
summary(m4)
logLik(m4)

# Compare models using anova()
anova(m1, m2, m3)



# Start with random intercepts only
m5 <- lmer("LogRT ~ CharacterLength_c + (1 | WorkerID)", data = exploratory_df)
summary(m5)
logLik(m5)

# Start with random intercepts only
m5b <- lmer("LogRT ~ LogFrequencies_c + (1 | WorkerID)", data = exploratory_df, REML = F)
summary(m5b)
logLik(m5b)

library("performance")

r2(m5b)
summary(m5b)


# Start with random intercepts only
m6 <- lmer("LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID)", data = exploratory_df)
summary(m6)
logLik(m6)
AIC(m6)
BIC(m6)


# Start with random intercepts only
m7 <- lmer("LogRT ~ CharacterLength_c + LogFrequencies_c + (1 | WorkerID) + (1 | POSTag)", data = exploratory_df)
summary(m7)
logLik(m7)
AIC(m7)
BIC(m7)


# Start with random intercepts only
m8 <- lmer("LogRT ~ CharacterLength_c + LogFrequencies_c + (1 + LogFrequencies_c | WorkerID) + (1 | POSTag)", data = exploratory_df)
summary(m8)
logLik(m8)


AIC(m8)
BIC(m8)

##############


RESULTS_DB_PATH <- "/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/results/results.db"

results_db <- dbConnect(RSQLite::SQLite(), RESULTS_DB_PATH)

baseline_df <- dbGetQuery(results_db,"With RESULT_TABLE AS (
                    WITH SPRT_AVG AS (
                    SELECT SPRTNaturalStories.StoryWordID, avg(SPRTNaturalStories.RT) as Avg_RT
                    FROM SPRTNaturalStories
                    GROUP BY SPRTNaturalStories.StoryWordID
                    )

                    SELECT ModelSurprisalScores.ModelID, ModelSurprisalScores.StoryWordID,
                    Story.CorpusID, Story.StoryID,
                    WordDetails.CharacterLength, WordDetails.LogFrequencies,  
                    ModelSurprisalScores.SurprisalScore,
                    SPRT_AVG.Avg_RT,

                    LAG(WordDetails.CharacterLength, 1) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev_len,
                    LAG(WordDetails.LogFrequencies, 1) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev_freq,
                    LAG(ModelSurprisalScores.SurprisalScore, 1) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev_surp,

                    LAG(WordDetails.CharacterLength, 2) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev2_len,
                    LAG(WordDetails.LogFrequencies, 2) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev2_freq,
                    LAG(ModelSurprisalScores.SurprisalScore, 2) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev2_surp,

                    LAG(WordDetails.CharacterLength, 3) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev3_len,
                    LAG(WordDetails.LogFrequencies, 3) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev3_freq,
                    LAG(ModelSurprisalScores.SurprisalScore, 3) OVER (PARTITION BY Story.StoryID ORDER BY Story.WordID) as prev3_surp

                    from ModelSurprisalScores
                    JOIN Story on Story.StoryWordID = ModelSurprisalScores.StoryWordID
                    JOIN WordDetails on WordDetails.WordUID = Story.WordUID
                    JOIN SPRT_AVG on SPRT_AVG.StoryWordID = ModelSurprisalScores.StoryWordID
                    Where ModelSurprisalScores.ModelID = 6892214 
                    ) 
                    SELECT * FROM RESULT_TABLE 
                    WHERE prev3_len IS NOT NULL
                
            ")

baseline_df$LogRT <- log(baseline_df$Avg_RT)

baseline_df$CharacterLength_c <- scale(baseline_df$CharacterLength)
#Should I scale Log Frequencies(?)
baseline_df$LogFrequencies_c <- scale(baseline_df$LogFrequencies)

baseline_df$prev_len_c <- scale(baseline_df$prev_len)
baseline_df$prev_freq_c <- scale(baseline_df$prev_freq)
baseline_df$prev_surp_c <- scale(baseline_df$prev_surp)

baseline_df$prev2_len_c <- scale(baseline_df$prev2_len)
baseline_df$prev2_freq_c <- scale(baseline_df$prev2_freq)
baseline_df$prev2_surp_c <- scale(baseline_df$prev2_surp)

baseline_df$prev3_len_c <- scale(baseline_df$prev3_len)
baseline_df$prev3_freq_c <- scale(baseline_df$prev3_freq)
baseline_df$prev3_surp_c <- scale(baseline_df$prev3_surp)


baseline_df$SurprisalScore_c <- scale(baseline_df$SurprisalScore)
data_size <- nrow(baseline_df)

fit_formula_b = "LogRT ~ CharacterLength_c + prev_len_c + prev2_len_c + prev3_len_c + LogFrequencies_c + prev_freq_c + prev2_freq_c + prev3_freq_c"
fit_formula_m = "LogRT ~ CharacterLength_c + prev_len_c + prev2_len_c + prev3_len_c + LogFrequencies_c + prev_freq_c + prev2_freq_c + prev3_freq_c + SurprisalScore_c + prev_surp_c + prev2_surp_c + prev3_surp_c"

#fit_formula_b = "LogRT ~ CharacterLength_c*LogFrequencies_c + prev_len_c*prev_freq_c + prev2_len_c*prev2_freq_c + prev3_len_c*prev3_freq_c"
#fit_formula_m = "LogRT ~ SurprisalScore_c + prev_surp_c + prev2_surp_c + prev3_surp_c + CharacterLength_c*LogFrequencies_c + prev_len_c*prev_freq_c + prev2_len_c*prev2_freq_c + prev3_len_c*prev3_freq_c"

model_b = lm(fit_formula_b, data= baseline_df)
summary(model_b)
logLik(model_b)

model_m = lm(fit_formula_m, data= baseline_df)
summary(model_m)
logLik(model_m)

logLik(model_m)-logLik(model_b)

