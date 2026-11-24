#loading initial library + reading in / filtering code
library(tidyverse)
library(naivebayes)

data <- read.csv("Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE)
colSums(is.na(data))
summary(data)

data2 <- data |>
  mutate(
    # Numerical variables (excluding id and the competition target addicted_label):
    # target is.na() subsets directly and replace with the computed column mean
    across(
      where(is.numeric) & !c(id, addicted_label),
      function(col) {
        col[is.na(col)] <- mean(col, na.rm = TRUE)
        col
      }
    ),
    # Categorical variables: convert any blank strings to NA first, then replace NA
    # subsets with the most frequent category (mode) via table() and which.max()
    across(
      where(is.character),
      function(col) {
        col[trimws(col) == ""] <- NA
        col[is.na(col)] <- names(which.max(table(col)))
        col
      }
    )
  )

#to show all numerical outliers
boxplot(data2$daily_screen_time_hours, data2$social_media_hours, 
        data2$gaming_hours, data2$work_study_hours, data2$sleep_hours, 
        data2$notifications_per_day, data2$app_opens_per_day, 
        data2$weekend_screen_time)

#to zoom in on categories with outliers
boxplot(data2$social_media_hours, 
        data2$work_study_hours,
        data2$weekend_screen_time)

#running through outliers and pruning them
x <- data2$social_media_hours
Q1 <- quantile(data2$social_media_hours, 0.25)
Q3 <- quantile(data2$social_media_hours, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data2 <- data2[(x >= lb & x <= ub),]

#outlier 2

x <- data2$work_study_hours
Q1 <- quantile(data2$work_study_hours, 0.25)
Q3 <- quantile(data2$work_study_hours, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data2 <- data2[(x >= lb & x <= ub),]

#outlier 3

x <- data2$weekend_screen_time
Q1 <- quantile(data2$weekend_screen_time, 0.25)
Q3 <- quantile(data2$weekend_screen_time, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data2 <- data2[(x >= lb & x <= ub),]

#stealing code from michael (sorry bro)
#splits data and sets seed
model_data <- data2 |>
  filter(!is.na(addicted_label)) |>
  mutate(addicted_label = factor(addicted_label, levels = c(1, 0)))

unlabeled_data <- data2 |>
  filter(is.na(addicted_label)) |>
  select(-addicted_label)

set.seed(3220)

#creates model and prediction then creates confusion matrix
bayesModel <- naivebayes::naive_bayes(addicted_label ~ ., data = model_data)
bayesPrediction <- predict(bayesModel, newdata = unlabeled_data)

idx <- sample(seq_len(nrow(model_data)), size = floor(0.8 * nrow(model_data)))

bayesModel <- naivebayes::naive_bayes(addicted_label ~ ., data = model_data)
test_predictions <- predict(bayesModel, newdata = model_data)

bayesTable <- table(actual = test$addicted_label, predicted = test_predictions)
