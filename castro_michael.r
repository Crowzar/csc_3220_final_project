library(tidyverse)
library(janitor)
library(tidymodels)
library(LiblineaR)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE, na.strings = c("", " ", "NA"))

colSums(is.na(data)) # This is to check for any NA's in the data

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

colSums(is.na(data2)) # This is to verify the NA's were removed

data2 |> select(-id) |> get_dupes() # This will use the janitor package to check for any duplicate data (excluding unique row id)

summary(data2)

nrow(data2)

# This is where building and training the SVM model will begin

# Separate labeled rows (for training/evaluation) from unlabeled competition rows (for prediction)
# unlabeled_data will be used for testing and the model_data with labels will be used to train the data
model_data <- data2 |>
  filter(!is.na(addicted_label)) |>
  mutate(addicted_label = factor(addicted_label, levels = c(1, 0)))

unlabeled_data <- data2 |>
  filter(is.na(addicted_label)) |>
  select(-addicted_label)

set.seed(3220)

# commenting out the following data since I changed stategy to use the already split data
# data_split <- initial_split(model_data, prop = 0.75, strata = addicted_label)
# started with prop at 0.75 and got an accuracy of 0.835
# next tried prop at 0.8 and the accuracy only increased to 0.836
# a prop of 0.6 produced an accuracy of 0.835 as well.
# tried a prop of 0.9 and the accuracy went to 0.839

train_data <- model_data

test_data <- unlabeled_data

# Linear SVM preprocessing recipe
svm_recipe <- recipe(addicted_label ~ ., data = train_data) |>
  step_rm(id) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# Linear SVM model specification
svm_spec <- svm_linear(mode = "classification", engine = "LiblineaR", cost = 1) #swapping kernlab for LiblineaR engine for it scales with large row counts

svm_wf <- workflow() |> add_recipe(svm_recipe) |> add_model(svm_spec)

svm_fit <- fit(svm_wf, data = train_data)

# Predict addicted_label on the unlabeled test rows
test_preds <- augment(svm_fit, new_data = test_data)

# Preview the first 6 predicted classes alongside each row's ID to verify predictions generated properly
head(test_preds |> select(id, .pred_class))

set.seed(3220)

folds <- vfold_cv(train_data, v = 5, strata = addicted_label)
# 10 folds is taking a long time with the amount of data, tried 5 folds next

svm_cv <- fit_resamples(
  svm_wf,
  resamples = folds,
  # Evaluation metrics (positive class = 1 / addicted):
  #   accuracy     - Overall proportion of correct predictions
  #   precision    - Precision: proportion of predicted addicted users (1) who are actually addicted
  #   recall       - Recall (sensitivity): proportion of actual addicted users (1) correctly predicted as addicted
  #   spec         - Specificity: proportion of actual non-addicted users (0) correctly predicted as not addicted
  metrics = metric_set(accuracy, precision, recall, spec),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(svm_cv)

# Feature correlations with addicted_label and with each other
# Uses the baked recipe features (numeric + dummy-encoded categoricals) and converts
# addicted_label back to 1 (addicted) / 0 (not addicted) so positive correlation = more addicted
cor_data <- svm_recipe |>
  prep() |>
  bake(new_data = NULL) |>
  mutate(addicted_label = as.numeric(as.character(addicted_label)))

cor_matrix <- cor(cor_data, use = "pairwise.complete.obs")

# Ranked correlations of each feature with being addicted (addicted_label)
addicted_cor <- tibble(
  feature = colnames(cor_matrix),
  correlation = cor_matrix[, "addicted_label"]
) |>
  filter(feature != "addicted_label") |>
  arrange(desc(abs(correlation)))

addicted_cor

# Full correlation matrix (features with addicted_label and with each other)
round(cor_matrix, 3)

# Correlation heatmap of all features and addicted_label
cor_matrix |>
  as.data.frame() |>
  rownames_to_column("feature_1") |>
  pivot_longer(-feature_1, names_to = "feature_2", values_to = "correlation") |>
  mutate(
    feature_1 = factor(feature_1, levels = colnames(cor_matrix)),
    feature_2 = factor(feature_2, levels = rev(colnames(cor_matrix)))
  ) |>
  ggplot(aes(x = feature_1, y = feature_2, fill = correlation)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", correlation)), size = 2.8) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Feature Correlation Matrix (Including Addicted Label)",
    x = NULL,
    y = NULL,
    fill = "Correlation"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Out-of-fold confusion matrix from 5-fold cross-validation
svm_conf_mat <- collect_predictions(svm_cv) |>
  conf_mat(truth = addicted_label, estimate = .pred_class)

svm_conf_mat

autoplot(svm_conf_mat, type = "heatmap") # This will produce a graphic of the confusion matrix