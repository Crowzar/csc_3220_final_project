library(tidyverse)
library(janitor)
library(e1071) # Provides svm() backed by libsvm, the standard SVM implementation in R
library(caret) # Used for confusionMatrix() to compute F1, sensitivity, specificity, and balanced accuracy
library(pROC)  # Used to compute ROC curves and ROC-AUC on imbalanced classes (~71% Addicted)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE, na.strings = c("", " ", "NA"))

colSums(is.na(data)) #This is to check for any NA's in the data

# The following line will no longer be used for the data set due to the teacher telling us that dropping this amount of our data would not be a good idea, and to instead impute the data using mean or most common category
# data2 <- data |> filter(if_all(-addicted_label, ~ !is.na(.) & trimws(as.character(.)) != "")) This will help clean the data to remove NA's and empty spaces, excluding the Addicted label column since the data was split already when we got it from the competition

# This will be the new data with imputed data instead of removing NA and blank rows
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

colSums(is.na(data2)) #This is to verify the NA's were removed

get_dupes(data2) #This will use the janitor packages to check for any duplicate data

summary(data2)

nrow(data2)

