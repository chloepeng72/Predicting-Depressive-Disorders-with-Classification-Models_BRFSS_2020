# CS 699 
# Term Project
# April 3, 2024
# Jack Perreault and Chloe Yihe Peng


library(tidyverse)
library(class)
library(dplyr)
library(caret)
library(rpart)
library(rsample)
library(cluster)
#install.packages("rlang")
library(rlang)


data <- read.csv("MET CS 699 Data Mining/Term Project/project_dataset_5K.csv")
#data <- read.csv("project_dataset_5K.csv")
summary(data)


# Drop specified columns
drop.cols <- c('IDATE', 'IDAY', 'IMONTH', 'IYEAR', 'SEQNO', 'X_PSU')
data_clean <- dplyr::select(data, -dplyr::one_of(drop.cols))


# Further data clean-up Jack's code
data_clean <- data_clean[, colSums(is.na(data_clean)) < 0.25 * nrow(data_clean)]

# Convert 'Class' from 'Y', 'N' to 1, 0
data_clean$Class <- ifelse(data_clean$Class == "Y", 1, 0)

# Normalize the scale of the numerical data
data_clean_numeric <- data_clean %>% select_if(is.numeric)

#data_clean_scaled <- as.data.frame(scale(data_clean_numeric))
#data_clean_scaled$Class <- data_clean$Class # Adding 'Class' column back

# Recoding specified variables
#data_clean_scaled$DISPCODE <- ifelse(data_clean_scaled$DISPCODE == 1100, "Y", "N")


# Save the preprocessed data to a CSV file
write.csv(data_clean_numeric, file = "preprocessed_data.csv", row.names = FALSE)
set.seed(123)
# save initial_train.csv and initial_test.csv
split <- initial_split(data_clean_numeric, prop = 0.66, strata = "Class")
data_clean_numeric$Class = factor(data_clean_numeric$Class)
train <- training(split)
test <- testing(split)

train$Class = factor(train$Class)
test$Class = factor(test$Class)

write.csv(train, file = "initial_train.csv", row.names = FALSE)
write.csv(test, file = "initial_test.csv", row.names = FALSE)

##########################################################################
# unsupervised  ML (analysis without considering any target variables)
library(stats)

# replace NAs with column means
data_clean_numeric <- apply(data_clean_numeric, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))

# take out rows with Infinite values
data_clean_numeric <- data_clean_numeric[!apply(data_clean_numeric, 1, function(x) any(is.infinite(x))), ]

# lastly convert it b= back to data frame just in case!!!
data_clean_numeric <- as.data.frame(data_clean_numeric)

############# PCA
pca_result <- prcomp(data_clean_numeric[, -which(names(data_clean_numeric) == "Class")], 
                     center = TRUE, 
                     scale. = TRUE)
summary(pca_result)
plot(pca_result, type = "l")


######## elbow clustering test to see.

# first for the sake of this method, replace NAs with the mean of the variables
data_clean_no_na <- apply(data_clean_numeric, 2, function(x) ifelse(is.na(x), mean(x, na.rm = TRUE), x))


# then calculate within-cluster sum of squares for different cluster numbers (k)
wss <- sapply(1:10, function(k) kmeans(data_clean_no_na, centers = k)$tot.withinss)

# Plot the elbow method to help determine the optimal number of clusters
plot(1:10, wss, type = "b", xlab = "Number of Clusters", ylab = "Within-cluster Sum of Squares")
title("Elbow Method for Optimal Number of Clusters")


########################################################### Three attribute selection methods
library(FSelector)
library(Boruta)
library(RWeka)
#install.packages("FSinR")
library(FSinR)

# information gain
info.gain <- information.gain(Class ~ ., data_clean_numeric)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$`Info Gain`), ]
sorted.info.gain
# top 10
sorted.info.gain[1:10, ]


# Calculate the gainRatio
gain_ratio <- gain.ratio(Class ~ ., data = data_clean_numeric)

gain_ratio_df <- cbind(Attributes = rownames(gain_ratio), data.frame(gain_ratio, row.names=NULL))
names(gain_ratio_df) <- c("Attribute", "Gain Ratio")

# Sort the gain ratios in descending order
sorted_gain_ratio <- gain_ratio_df[order(-gain_ratio_df$`Gain Ratio`), ]
print(sorted_gain_ratio)
# top 10
top_10_gain_ratio <- sorted_gain_ratio[1:10, ]
print(top_10_gain_ratio)



# calculate the Gini index

# Function to calculate Gini Index for Class vs the result of features
calculateAndPrintTop10GiniGains <- function(data, target) {
  features <- setdiff(names(data), target) # All column names except the target
  giniResults <- list()
  
  for(feature in features) {
    totalInstances <- nrow(data) # Total number of instances
    
    # Calculate the counts and proportions of each level of the target variable
    targetTable <- table(data[[target]])
    targetProportions <- targetTable / totalInstances
    overallGini <- 1 - sum(targetProportions^2)
    
    # Split the data based on the feature and calculate Gini index for each subgroup
    splitData <- split(data, data[[feature]])
    weightedGini <- 0
    
    for(split in names(splitData)) {
      splitSize <- nrow(splitData[[split]])
      splitTable <- table(splitData[[split]][[target]])
      splitProportions <- splitTable / splitSize
      splitGini <- 1 - sum(splitProportions^2)
      weightedGini <- weightedGini + (splitSize / totalInstances) * splitGini
    }
    
    # Calculate the Gini gain
    giniGain <- overallGini - weightedGini
    
    # Store the results in the list
    giniResults[[feature]] <- giniGain
  }
  
  # Convert the results to a data frame for easy sorting
  giniDataFrame <- data.frame(Feature = names(giniResults), GiniGain = unlist(giniResults), row.names = NULL)
  
  # Sort by GiniGain in descending order
  sortedGiniDataFrame <- giniDataFrame[order(-giniDataFrame$GiniGain), ]
  
  # Return the top 10 features based on Gini gain
  return(sortedGiniDataFrame[1:10, ])
}


# Apply the function
top10GiniGains <- calculateAndPrintTop10GiniGains(data_clean_numeric, "Class")
print(top10GiniGains)

########################################################## 3 methods of balancing the dataset

### Method 1: bootstrapped
Class_1 <- data_clean_numeric[data_clean_numeric$Class == 1, ]

# Calculate how many more instances I need, i decided on the threshold to be 3600 for better TPRs
needed_instances <- 3600 - nrow(Class_1)

# Sample with replacement
bootstrapped_Class_1 <- Class_1[sample(nrow(Class_1), size = needed_instances, replace = TRUE), ]

# Combine the bootstrapped instances with the original Class 1 instances
augmented_Class_1 <- rbind(Class_1, bootstrapped_Class_1)

# Now combine back with the Class 0 instances to form the new, more balanced dataset
Class_0 <- data_clean_numeric[data_clean_numeric$Class == 0, ]
balanced_data <- rbind(Class_0, augmented_Class_1)

# Shuffle the balanced dataset 
set.seed(123)
balanced_data <- balanced_data[sample(nrow(balanced_data)), ]

table(balanced_data$Class)


### Method 2 SMOTE:
#install.packages("DMwR2")
library(DMwR2)

#install.packages("smotefamily")
library(smotefamily)

# Applying SMOTE
# Here, I decided k to be 7, since the result from elbow method shows that is the optimal number of cluster of the dataset. 
data_clean_numeric$Class <- as.factor(data_clean_numeric$Class)

X <- data_clean_numeric[, -which(names(data_clean_numeric) == "Class")]
Y <- data_clean_numeric$Class
balanced_data_1 <- SMOTE(X, Y, K = 7, dup_size = 3)

# convert it back to a df
balanced_df <- balanced_data_1$data

# shuffle the balanced dataset
set.seed(123)
shuffled_indices <- sample(nrow(balanced_df))
balanced_df_shuffled <- balanced_df[shuffled_indices, ]

balanced_df_shuffled$class <- as.factor(balanced_df_shuffled$class)
names(balanced_df_shuffled)[names(balanced_df_shuffled) == "class"] <- "Class"

summary(balanced_df_shuffled)


##################### Method 3  JACK's CODE TO GET BORUTA AND INFO GAIN
library(Boruta)
sum(is.na(data_clean_numeric))
sapply(data_clean_numeric, function(x) sum(is.na(x)))
data_clean_numeric_filled <- data_clean_numeric

# Loop through each column in the dataframe
for (col_name in names(data_clean_numeric_filled)) {
  # Check if the column is numeric to avoid trying to calculate mean of non-numeric columns
  if (is.numeric(data_clean_numeric_filled[[col_name]])) {
    # Replace NA values with the mean of the column, excluding NA values in the mean calculation
    data_clean_numeric_filled[[col_name]][is.na(data_clean_numeric_filled[[col_name]])] <- mean(data_clean_numeric_filled[[col_name]], na.rm = TRUE)
  }
}

data.boruta = Boruta(Class ~ ., data = data_clean_numeric_filled)
data.boruta

cols = c("Class", getSelectedAttributes(data.boruta, withTentative = FALSE))
data_boruta = data_clean_numeric_filled[cols]

data_clean_numeric_filled
info.gain <- information.gain(Class ~ ., data_clean_numeric_filled)
info.gain <- cbind(rownames(info.gain), data.frame(info.gain, row.names=NULL))
names(info.gain) <- c("Attribute", "Info Gain")
sorted.info.gain <- info.gain[order(-info.gain$`Info Gain`), ]
sorted.info.gain
# top 10
top10 = sorted.info.gain[1:10, 'Attribute']
top10_final = c(top10, "Class")
top10_numeric_filled = data_clean_numeric_filled[top10_final]

###############################################################################################################################
###############################################################################################################################
# classification models


################################################ Log Regression(Jack)
TPRTNR = function(TP, TN, FP, FN){
  TPR = TP/(TP + FN)
  TNR = TN/ (TN + FP)
  return(list(TPR = TPR, TNR = TNR))
}
library(stats)

set.seed(123)
library(pROC)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
top10_numeric_filled$Class = factor(top10_numeric_filled$Class)
train <- training(split)
test <- testing(split)

sum(as.numeric(train$Class == 1))/nrow(train)

logmodel = glm(Class ~ ., data = train, family = "binomial")

y_pred = predict(logmodel, test, type = "response")
roc_obj = roc(response = test$Class, predictor = y_pred)
optimal_cutoff = coords(roc_obj, "best", ret = "threshold")

y_pred
y_pred = as.factor(ifelse(y_pred >= as.numeric(optimal_cutoff), 1, 0))
y_pred
test$Class = as.factor(test$Class)
confusionMatrix(y_pred, test$Class)
TPRTNR(173, 1062, 315, 151)


################## With Boruta GOOD RESULTS
set.seed(123)
library(pROC)
split <- initial_split(data_boruta, prop = 0.66, strata = "Class")
data_boruta$Class = factor(data_boruta$Class)
train <- training(split)
test <- testing(split)

sum(as.numeric(train$Class == 1))/nrow(train)

logmodel = glm(Class ~ ., data = train, family = "binomial")


y_pred = predict(logmodel, test, type = "response")
roc_obj = roc(response = test$Class, predictor = y_pred)
optimal_cutoff = coords(roc_obj, "best", ret = "threshold")

y_pred
y_pred = as.factor(ifelse(y_pred >= as.numeric(optimal_cutoff), 1, 0))
y_pred
test$Class = as.factor(test$Class)
confusionMatrix(y_pred, test$Class)
TPRTNR(214, 1050, 327, 110)
library(caret)


performance_measures_log_reg <- confusionMatrix(y_pred, test$Class)

# calculate performance measures
# in this case: 1 is postive and predicted 1 is tp, that is why tp is row 2 column 2.
cm_log_reg <- performance_measures_log_reg$table
tp = cm_log_reg[2,2]
fp = cm_log_reg[2,1]
tn = cm_log_reg[1,1]
fn = cm_log_reg[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}


# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == 0)
n1 <- sum(test$Class == 1)

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")


########################################################################  C5.0

### c5.0 with imbalanced data (top10_numeric_filled)
library(C50)
set.seed(123)
library(pROC)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
top10_numeric_filled$Class = factor(top10_numeric_filled$Class)
train <- training(split)
test <- testing(split)

C5.tree <- C5.0(Class ~ ., data = train)

# test 
pred <- predict(C5.tree, newdata = test, type = "class")


performance_measures_c50 <- confusionMatrix(data=pred,reference = test$Class)
performance_measures_c50

### c5.0 with data: balanced_data
library(C50)
set.seed(123)
library(pROC)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
balanced_data$Class = factor(balanced_data$Class)
train <- training(split)
test <- testing(split)

C5.tree <- C5.0(Class ~ ., data = train)

# test 
pred <- predict(C5.tree, newdata = test, type = "class")


performance_measures_c50 <- confusionMatrix(data=pred,reference = test$Class)
performance_measures_c50



### c50 with balanced data(best)
library(C50)
set.seed(123)
library(pROC)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
balanced_df_shuffled$Class = factor(balanced_df_shuffled$Class)
train <- training(split)
test <- testing(split)

C5.tree <- C5.0(Class ~ ., data = train)

# test 
pred <- predict(C5.tree, newdata = test, type = "Class")


performance_measures_c50 <- confusionMatrix(data=pred,reference = test$Class)
performance_measures_c50


# calculate performance measures
# in this case: 1 is postive and predicted 1 is tp, that is why tp is row 2 column 2.
cm_c50 <- performance_measures_c50$table
tp = cm_c50[2,2]
fp = cm_c50[2,1]
tn = cm_c50[1,1]
fn = cm_c50[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")



##################################################################### J48
library(RWeka)
modelLookup("J48")

### J48 with top10_numeric_filled
set.seed(123)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
top10_numeric_filled$Class = factor(top10_numeric_filled$Class)
train <- training(split)
test <- testing(split)

# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

model <- caret::train(Class ~ ., data = train, method = "J48", trControl = train_control)
model
#plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$Class)

performance_measures_j48 <- confusionMatrix(test_pred, test$Class)
performance_measures_j48





### J48 with balanced_data
set.seed(123)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
balanced_data$Class = factor(balanced_data$Class)
train <- training(split)
test <- testing(split)

# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

model <- caret::train(Class ~ ., data = train, method = "J48", trControl = train_control)
model
#plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$Class)

performance_measures_j48 <- confusionMatrix(test_pred, test$Class)
performance_measures_j48




### J48 with balanced_df_shuffled(best)
set.seed(123)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
balanced_df_shuffled$Class = factor(balanced_df_shuffled$Class)
train <- training(split)
test <- testing(split)

# repeat 10-fold cross-validation 5 times
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, 
                              summaryFunction = defaultSummary)

model <- caret::train(Class ~ ., data = train, method = "J48", trControl = train_control)
model
#plot(model)
test_pred <- predict(model, newdata = test)
confusionMatrix(test_pred, test$Class)

performance_measures_j48 <- confusionMatrix(test_pred, test$Class)
performance_measures_j48

# calculate performance measures
# in this case: 1 is postive and predicted 1 is tp, that is why tp is row 2 column 2.

cm_j48 <- performance_measures_j48$table
tp = cm_j48[2,2]
fp = cm_j48[2,1]
tn = cm_j48[1,1]
fn = cm_j48[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")



################################################################### mlpML
#install.packages("RSNNS")
library(RSNNS)
modelLookup('mlpML')

### mlpML with top10_numeric_filled
set.seed(123)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(top10_numeric_filled$Class))
test$Class <- factor(test$Class, levels = levels(top10_numeric_filled$Class))


ctrl = trainControl(method = 'cv', number = 10)

mlp_grid <- expand.grid(layer1 = c(5, 10, 15),
                        layer2 = c(5, 10, 15),
                        layer3 = c(5, 10, 15))



### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = mlp_grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlp <- caret::confusionMatrix(pr, test$Class, positive = '1')
cm_mlp


### mlpML with balanced_data
set.seed(123)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_data$Class))
test$Class <- factor(test$Class, levels = levels(balanced_data$Class))


ctrl = trainControl(method = 'cv', number = 10)

mlp_grid <- expand.grid(layer1 = c(5, 10, 15),
                        layer2 = c(5, 10, 15),
                        layer3 = c(5, 10, 15))



### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = mlp_grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlp <- caret::confusionMatrix(pr, test$Class, positive = '1')
cm_mlp




### mlpML with balanced_df_shuffled(best)
set.seed(123)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_df_shuffled$Class))
test$Class <- factor(test$Class, levels = levels(balanced_df_shuffled$Class))


ctrl = trainControl(method = 'cv', number = 10)

mlp_grid <- expand.grid(layer1 = c(5, 10, 15),
                        layer2 = c(5, 10, 15),
                        layer3 = c(5, 10, 15))



### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = mlp_grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlp <- caret::confusionMatrix(pr, test$Class, positive = '1')
cm_mlp


# calculate performance measures 
cm_mlp <- cm_mlp$table
tp = cm_mlp[2,2]
fp = cm_mlp[2,1]
tn = cm_mlp[1,1]
fn = cm_mlp[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation

# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) /(n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")




################################################# gradient boosting machine
#install.packages("gbm")
library(gbm)

### with top10_numeric_filled
# Adjust factor levels to be valid R variable names!!
top10_numeric_filled$Class <- factor(top10_numeric_filled$Class)
levels(top10_numeric_filled$Class) <- make.names(levels(top10_numeric_filled$Class), unique = TRUE)


set.seed(123)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
top10_numeric_filled$Class = factor(top10_numeric_filled$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(top10_numeric_filled$Class))
test$Class <- factor(test$Class, levels = levels(top10_numeric_filled$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(0.001, 0.01, 0.1),
                       n.minobsinnode = 3)


# Train the model
set.seed(123)
gbmFit <- caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)


gbmFit
#plot(gbmFit)

pred <- predict(gbmFit, test)
cm_gradient <- caret::confusionMatrix(pred, test$Class)
cm_gradient


### with balanced_data
# Adjust factor levels to be valid R variable names!!
balanced_data$Class <- factor(balanced_data$Class)
levels(balanced_data$Class) <- make.names(levels(balanced_data$Class), unique = TRUE)


set.seed(123)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
balanced_data$Class = factor(balanced_data$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_data$Class))
test$Class <- factor(test$Class, levels = levels(balanced_data$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(0.001, 0.01, 0.1),
                       n.minobsinnode = 3)


# Train the model
set.seed(123)
gbmFit <- caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)


gbmFit
#plot(gbmFit)

pred <- predict(gbmFit, test)
cm_gradient <- caret::confusionMatrix(pred, test$Class)
cm_gradient




### with balanced_df_shuffled(best model)
# Adjust factor levels to be valid R variable names!!
balanced_df_shuffled$Class <- factor(balanced_df_shuffled$Class)
levels(balanced_df_shuffled$Class) <- make.names(levels(balanced_df_shuffled$Class), unique = TRUE)


set.seed(123)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
balanced_df_shuffled$Class = factor(balanced_df_shuffled$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_df_shuffled$Class))
test$Class <- factor(test$Class, levels = levels(balanced_df_shuffled$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl <- trainControl(method = "CV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

gbmGrid <- expand.grid(interaction.depth = c(1, 3, 5, 7, 9),
                       n.trees = (2:10) * 10,
                       shrinkage = c(0.001, 0.01, 0.1),
                       n.minobsinnode = 3)


# Train the model
set.seed(123)
gbmFit <- caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "gbm",
                       tuneGrid = gbmGrid,
                       metric = "ROC",
                       verbose = FALSE,
                       trControl = ctrl)


gbmFit
#plot(gbmFit)

pred <- predict(gbmFit, test)
cm_gradient <- caret::confusionMatrix(pred, test$Class)
cm_gradient


# calculate performance measures 
cm_gradient <- cm_gradient$table
tp = cm_gradient[2,2]
fp = cm_gradient[2,1]
tn = cm_gradient[1,1]
fn = cm_gradient[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")


#############################################################   XGBoost
#install.packages("xgboost")
library(xgboost)

### with top10_numeric_filled
# Adjust factor levels to be valid R variable names!!
top10_numeric_filled$Class <- factor(top10_numeric_filled$Class)
levels(top10_numeric_filled$Class) <- make.names(levels(top10_numeric_filled$Class), unique = TRUE)


set.seed(123)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
top10_numeric_filled$Class = factor(top10_numeric_filled$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(top10_numeric_filled$Class))
test$Class <- factor(test$Class, levels = levels(top10_numeric_filled$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7))

set.seed(123)
xgbModel <- caret::train(x = train[, -class_column_index], # Exclude 'Class' column
                         y = train[, class_column_index],   # Specify the 'Class' predicted column
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel

#plot(xgbModel)

pred <- predict(xgbModel, test)
cm_xgb <- caret::confusionMatrix(pred, test$Class)
cm_xgb



### with balanced_data
# Adjust factor levels to be valid R variable names!!
balanced_data$Class <- factor(balanced_data$Class)
levels(balanced_data$Class) <- make.names(levels(balanced_data$Class), unique = TRUE)


set.seed(123)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
balanced_data$Class = factor(balanced_data$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_data$Class))
test$Class <- factor(test$Class, levels = levels(balanced_data$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7))

set.seed(123)
xgbModel <- caret::train(x = train[, -class_column_index], # Exclude 'Class' column
                         y = train[, class_column_index],   # Specify the 'Class' predicted column
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel

#plot(xgbModel)

pred <- predict(xgbModel, test)
cm_xgb <- caret::confusionMatrix(pred, test$Class)
cm_xgb




### with balanced_df_shuffled (best model)
# Adjust factor levels to be valid R variable names!!
balanced_df_shuffled$Class <- factor(balanced_df_shuffled$Class)
levels(balanced_df_shuffled$Class) <- make.names(levels(balanced_df_shuffled$Class), unique = TRUE)


set.seed(123)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
balanced_df_shuffled$Class = factor(balanced_df_shuffled$Class)
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_df_shuffled$Class))
test$Class <- factor(test$Class, levels = levels(balanced_df_shuffled$Class))


### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


xgb_control = trainControl(
  method = "cv", number = 10,
  summaryFunction = defaultSummary)

xgbGrid <- expand.grid(
  nrounds = seq(from = 50, to = 300, by = 50),
  eta = c(0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  gamma = c(0, 1, 2),
  colsample_bytree = 1,
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.7))

set.seed(123)
xgbModel <- caret::train(x = train[, -class_column_index], # Exclude 'Class' column
                         y = train[, class_column_index],   # Specify the 'Class' predicted column
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "Spec",
                         verbose = FALSE,
                         trControl = xgb_control)
xgbModel

#plot(xgbModel)

pred <- predict(xgbModel, test)
cm_xgb <- caret::confusionMatrix(pred, test$Class)
cm_xgb

# calculate performance measures 
cm_xgb <- cm_xgb$table
tp = cm_xgb[2,2]
fp = cm_xgb[2,1]
tn = cm_xgb[1,1]
fn = cm_xgb[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)

# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")


##################################################################### mlpWeightDecayML
library(rsample)

### With top10_numeric_filled
set.seed(123)
split <- initial_split(top10_numeric_filled, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are appropriate data type
train$Class <- factor(train$Class, levels = levels(top10_numeric_filled$Class))
test$Class <- factor(test$Class, levels = levels(top10_numeric_filled$Class))

### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl = trainControl(method = 'cv', number = 10)
mlp_grid = expand.grid(layer1 = c(1,3),
                       layer2 = c(1,3),
                       layer3 = c(1,3),
                       decay = seq(0.01, 0.05, 0.1))
grid <- expand.grid(layer1 = 1:3, layer2 = 1:3, layer3 = 1:3,
                    decay = c(0, .001))

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpWeightDecayML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlpWeight <- caret::confusionMatrix(pr, test$Class)
cm_mlpWeight


### With balanced_data
set.seed(123)
split <- initial_split(balanced_data, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are appropriate data type
train$Class <- factor(train$Class, levels = levels(balanced_data$Class))
test$Class <- factor(test$Class, levels = levels(balanced_data$Class))

### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl = trainControl(method = 'cv', number = 10)
mlp_grid = expand.grid(layer1 = c(1,3),
                       layer2 = c(1,3),
                       layer3 = c(1,3),
                       decay = seq(0.01, 0.05, 0.1))
grid <- expand.grid(layer1 = 1:3, layer2 = 1:3, layer3 = 1:3,
                    decay = c(0, .001))

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpWeightDecayML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlpWeight <- caret::confusionMatrix(pr, test$Class)
cm_mlpWeight



### With balanced_df_shuffled(best model)
set.seed(123)
split <- initial_split(balanced_df_shuffled, prop = 0.66, strata = "Class")
train <- training(split)
test <- testing(split)

# make sure the train and test data are approiate data type
train$Class <- factor(train$Class, levels = levels(balanced_df_shuffled$Class))
test$Class <- factor(test$Class, levels = levels(balanced_df_shuffled$Class))

### first find the index where Class is
class_column_index <- which(names(train) == "Class")
print(class_column_index)


ctrl = trainControl(method = 'cv', number = 10)
mlp_grid = expand.grid(layer1 = c(1,3),
                       layer2 = c(1,3),
                       layer3 = c(1,3),
                       decay = seq(0.01, 0.05, 0.1))
grid <- expand.grid(layer1 = 1:3, layer2 = 1:3, layer3 = 1:3,
                    decay = c(0, .001))

mlp_fit = caret::train(x = train[, -class_column_index],  # Exclude 'Class' column
                       y = train[, class_column_index],   # Specify the 'Class' predicted column
                       method = "mlpWeightDecayML", 
                       preProc =  c('center', 'scale'),
                       trControl = ctrl,
                       tuneGrid = grid)

mlp_fit
#plot(mlp_fit)

pr <- predict(mlp_fit, newdata = test)
pr

cm_mlpWeight <- caret::confusionMatrix(pr, test$Class)
cm_mlpWeight


# calculate performance measures 
cm_mlpWeight <- cm_mlpWeight$table
tp = cm_mlpWeight[2,2]
fp = cm_mlpWeight[2,1]
tn = cm_mlpWeight[1,1]
fn = cm_mlpWeight[1,2]

calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

# Calculate measures for class 0
performance_measures_class0 <- calculate_measures(tn, fn, tp, fp)
print(performance_measures_class0)

# Calculate measures for class 1
performance_measures_class1 <- calculate_measures(tp, fp, tn, fn)
print(performance_measures_class1)


# Weighted Average Calculation
# let the number of instances in class 0 be n0, and in class 1 be n1
n0 <- sum(test$Class == "X0")
n1 <- sum(test$Class == "X1")

weighted_average <- function(performance_measures_class0, performance_measures_class1, n0, n1) {
  weighted_avg <- (n0 * performance_measures_class0$values + n1 * performance_measures_class1$values) / (n0 + n1)
  measures <- performance_measures_class0$measures
  weighted_average_df <- data.frame(measures, weighted_avg)
  return(weighted_average_df)
}

weighted_average <- weighted_average(performance_measures_class0, performance_measures_class1, n0, n1)
print(weighted_average)

# ROC Curve
library(pROC)

roc_data <- roc(test$Class, as.numeric(pred))
plot(roc_data, col = "blue", main = "ROC Curve")
