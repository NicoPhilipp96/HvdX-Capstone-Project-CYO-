#load libraries
library(tidyverse)
library(nbastatR)
library(tidymodels)
library(rpart)
library(rpart.plot)
library(ROCR)
library(caret)
library(tinytex)
library(pROC)

Sys.setenv(VROOM_CONNECTION_SIZE = 500000)# Set the connection buffer size


  #gather data from the nbastatr package
  nbastatR::bref_teams_stats(seasons = 2017,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat1 <- dataBREFPerGameTeams
  
  nbastatR::bref_teams_stats(seasons = 2018,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat2 <- dataBREFPerGameTeams
  
  nbastatR::bref_teams_stats(seasons = 2019,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat3 <- dataBREFPerGameTeams
  
  nbastatR::bref_teams_stats(seasons = 2021,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat4 <- dataBREFPerGameTeams
  
  nbastatR::bref_teams_stats(seasons = 2022,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat5 <- dataBREFPerGameTeams
  
  nbastatR::bref_teams_stats(seasons = 2023,
                             return_message = FALSE,
                             assign_to_environment = TRUE,
                             nest_data = FALSE,
                             join_data = TRUE,
                             widen_data = TRUE)
  dat6 <- dataBREFPerGameTeams
  
  #combine into one dataframe
  combined_df <- rbind(dat1, dat2, dat3, dat4, dat5, dat6)


#####select metrics of interest
  combined_df_clean <- combined_df %>%
    select(yearSeason, nameTeam, isPlayoffTeam, astPerGameTeam, blkPerGameTeam, drbPerGameTeam, orbPerGameTeam, pctFG2PerGameTeam, pctFG3PerGameTeam, pctFTPerGameTeam, ptsPerGameTeam)


###playoff vs. non playoff counts
playoffs_counts <- table(combined_df_clean$isPlayoffTeam)
print(playoffs_counts)

###start decision tree model process

#80 percent train, 20 percent test
set.seed(123)

trainIndex <- createDataPartition(combined_df_clean$isPlayoffTeam, p = 0.8, list = FALSE, times = 1)

train_df <- combined_df_clean[trainIndex,]
test_df <- combined_df_clean[-trainIndex,]

###Defining our predictor (x) and response (y) variables
x <- combined_df_clean %>% select(c(yearSeason, nameTeam, astPerGameTeam, blkPerGameTeam, drbPerGameTeam, orbPerGameTeam, pctFG2PerGameTeam, pctFG3PerGameTeam, pctFTPerGameTeam, ptsPerGameTeam))

y <- combined_df_clean$isPlayoffTeam

###Extracting Training Data (X_train, y_train) and Validation Data (X_val, y_val)

X_train <- train_df %>% select(c(yearSeason, nameTeam, astPerGameTeam, blkPerGameTeam, drbPerGameTeam, orbPerGameTeam, pctFG2PerGameTeam, pctFG3PerGameTeam, pctFTPerGameTeam, ptsPerGameTeam))

y_train <- train_df$isPlayoffTeam

###

X_val <- test_df %>% select(c(yearSeason, nameTeam, astPerGameTeam, blkPerGameTeam, drbPerGameTeam, orbPerGameTeam, pctFG2PerGameTeam, pctFG3PerGameTeam, pctFTPerGameTeam, ptsPerGameTeam))

y_val <- test_df$isPlayoffTeam


#define the parameters for model training.
set.seed(123)

fitControl <- trainControl(method = 'loocv',
                           number = 10,
                           savePredictions = 'final') #Save final prediction for each fold



#Training the decision tree model
#set the max depth of the tree, and minimum instances for the tree to split
set.seed(123)

tree <- train(factor(isPlayoffTeam) ~.,
              data = train_df %>% select(-c("nameTeam", "yearSeason")),
              method = 'rpart',
              metric = 'Accuracy',
              trControl = fitControl,
              control = rpart.control(minsplit = 1,
                                      maxdepth = 4))

#This code segment is evaluating the performance of a decision tree model on different datasets (training, validation, and the entire dataset) and aims to assess if the model is overfitting or underfitting

#train - test - split

#determining if overfit(accuracy < train ) or underfit(accurary>train)

#1) setting up model for the trained dataframe 
tree_predict_train <- tree %>% predict(X_train, method = 'prop')
#2) testing model on the validating dataframe - train/test split.
tree_predict_val <- tree %>% predict(X_val, method = 'prop')
#3) testing model on the whole dataframeix
tree_predict <- tree %>% predict(x, method = 'prop')

###

#1) testing model on the trained dataframe - theoretically the best fit bc no new data. Over/underfit is with respect to this model's results
confusionMatrix(factor(tree_predict_train), factor(y_train), positive = "TRUE")

#2) testing model on the validating dataframe - train/test split. 
confusionMatrix(factor(tree_predict_val), factor(y_val))

#3) testing model on the whole dataframe - report the model's score metrics from this confusionmatrix. Provides an overall assessment of the model's performance
confusionMatrix(factor(tree_predict), factor(y), positive = "TRUE")


###plot decision tree
rattle::fancyRpartPlot(tree$finalModel, type = 4, caption = "Playoff Model")


#Receiver Operating Characteristic (ROC) curve analysis to evaluate the performance of a binary classification model.

# Get predicted probabilities for the positive class (isPlayoffTeam = TRUE)
set.seed(123)

predictions <- predict(tree, newdata = X_val, type = "prob")[, "TRUE"]

#create prediction object
pred <- prediction(predictions, y_val)

# Compute the AUC
auc <- performance(pred, "auc")@y.values[[1]]

# Print the AUC
print(auc)

#calculate performance measures
perf <- performance(pred, "tpr", "fpr")

#plot roc curve
plot(perf, main = "ROC Curve", col = "blue")

# Add diagonal reference line
abline(a = 0, b = 1, col = "red", lty = 2)


#########################################################

###To fullfil the two model requirement, we want to see how a more simple, logistic regression model compares to the decision tree model?
# Logistic Regression Model
set.seed(123)
logistic_model <- glm(isPlayoffTeam ~ ., data = train_df, family = "binomial")

# Predictions on Training Set
logistic_predict_train <- predict(logistic_model, newdata = X_train, type = "response")

# Predictions on Validation Set
logistic_predict_val <- predict(logistic_model, newdata = X_val, type = "response")

# Predictions on the Whole Dataset
logistic_predict <- predict(logistic_model, newdata = x, type = "response")

# Model Evaluation for Logistic Regression
confusionMatrix(factor(logistic_predict_train > 0.5), factor(y_train), positive = "TRUE")
confusionMatrix(factor(logistic_predict_val > 0.5), factor(y_val))
confusionMatrix(factor(logistic_predict > 0.5), factor(y), positive = "TRUE")

# ROC Curve for Logistic Regression
roc_curve <- roc(y_val, logistic_predict_val)
auc <- auc(roc_curve)

# Plotting ROC Curve
plot(roc_curve, main = "ROC Curve for Logistic Regression", col = "blue")

legend("bottomright", legend = paste("AUC =", round(auc, 2)), col = "blue", lty = 1, cex = 0.8)
