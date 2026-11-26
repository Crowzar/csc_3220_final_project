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
  #   f_meas       - F1 score: harmonic mean of precision and recall
  metrics = metric_set(accuracy, precision, recall, spec, f_meas),
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

# Output folder for SVM figures
fig_dir <- if (dir.exists("csc_3220_final_project")) {
  file.path("csc_3220_final_project", "svm_figures")
} else {
  "svm_figures"
}
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Correlation heatmap of all features and addicted_label
cor_heatmap <- cor_matrix |>
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

cor_heatmap
ggsave(
  filename = file.path(fig_dir, "01_correlation_matrix.png"),
  plot = cor_heatmap,
  width = 10,
  height = 8.5,
  dpi = 300,
  bg = "white"
)

# Out-of-fold confusion matrix from 5-fold cross-validation
svm_conf_mat <- collect_predictions(svm_cv) |>
  conf_mat(truth = addicted_label, estimate = .pred_class)

svm_conf_mat

svm_cm_plot <- autoplot(svm_conf_mat, type = "heatmap") + # This will produce a graphic of the confusion matrix
  labs(title = "Confusion Matrix - Linear SVM (5-Fold CV)")

svm_cm_plot
ggsave(
  filename = file.path(fig_dir, "02_confusion_matrix.png"),
  plot = svm_cm_plot,
  width = 6.5,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# Named lookup vector to turn yardstick metric IDs into clean plot labels
metric_labels <- c(
  accuracy  = "Accuracy",
  precision = "Precision",
  recall    = "Recall (Sensitivity)",
  spec      = "Specificity",
  f_meas    = "F1 Score"
)

# Summarize mean and standard error for each metric across the 5 CV folds,
# and lock the factor order so bars appear in the order defined above
cv_metrics_summary <- collect_metrics(svm_cv) |>
  mutate(
    metric_label = factor(
      metric_labels[.metric],
      levels = unname(metric_labels)
    )
  )

# Build bar chart of mean CV scores with +/- 1 standard error bars and numeric labels
svm_metrics_plot <- ggplot(cv_metrics_summary, aes(x = metric_label, y = mean)) +
  geom_col(fill = "#2C7FB8", width = 0.65) +
  # Clamp error bars to [0, 1] so they never extend past valid percentage bounds
  geom_errorbar(
    aes(ymin = pmax(0, mean - std_err), ymax = pmin(1, mean + std_err)),
    width = 0.2,
    linewidth = 0.6
  ) +
  # Place exact 3-decimal score above each bar
  geom_text(
    aes(label = sprintf("%.3f", mean)),
    vjust = -0.8,
    size = 3.8,
    fontface = "bold"
  ) +
  # Format y-axis as percentages with extra headroom above 100% for text labels
  scale_y_continuous(limits = c(0, 1.08), labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Linear SVM Performance Metrics (5-Fold CV)",
    subtitle = "Bars show mean across 5 folds; error bars show +/- 1 SE",
    x = NULL,
    y = "Score"
  ) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

# Display in Plots pane and save as a 300-DPI PNG
svm_metrics_plot
ggsave(
  filename = file.path(fig_dir, "03_cv_metrics_summary.png"),
  plot = svm_metrics_plot,
  width = 8,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# summarize = FALSE returns each fold's individual score instead of the 5-fold average
cv_metrics_by_fold <- collect_metrics(svm_cv, summarize = FALSE) |>
  mutate(
    metric_label = factor(
      metric_labels[.metric],
      levels = unname(metric_labels)
    )
  )

# Plot each metric across Fold1-Fold5; flat lines indicate consistent generalization
svm_folds_plot <- ggplot(
  cv_metrics_by_fold,
  aes(x = id, y = .estimate, color = metric_label, group = metric_label)
) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Linear SVM Metric Stability Across 5 CV Folds",
    x = "Cross-Validation Fold",
    y = "Score",
    color = "Metric"
  ) +
  theme_minimal(base_size = 11)

# Display and export the fold-by-fold stability plot
svm_folds_plot
ggsave(
  filename = file.path(fig_dir, "04_cv_metrics_by_fold.png"),
  plot = svm_folds_plot,
  width = 8.5,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# Extract the learned linear hyperplane weights from the fitted LiblineaR model;
# drop the intercept ("Bias") and tag each weight's sign for color-coding
svm_weights <- tidy(svm_fit) |>
  filter(term != "Bias") |>
  mutate(direction = if_else(estimate >= 0, "Positive", "Negative"))

# Horizontal bar chart sorted by absolute weight magnitude (since all predictors
# were standardized with step_normalize(), magnitudes are directly comparable)
svm_weights_plot <- ggplot(
  svm_weights,
  aes(x = reorder(term, abs(estimate)), y = estimate, fill = direction)
) +
  geom_col() +
  geom_hline(yintercept = 0, color = "gray40", linewidth = 0.5) + # Zero reference line
  coord_flip() + # Flip axes so feature names are easy to read horizontally
  scale_fill_manual(
    values = c("Positive" = "#B2182B", "Negative" = "#2166AC"),
    name = "Weight Sign"
  ) +
  labs(
    title = "Linear SVM Feature Weights (Standardized Predictors)",
    subtitle = "Larger absolute weight = stronger influence on the SVM decision boundary",
    x = NULL,
    y = "SVM Weight"
  ) +
  theme_minimal(base_size = 11)

# Display and export the feature weights plot
svm_weights_plot
ggsave(
  filename = file.path(fig_dir, "05_svm_feature_weights.png"),
  plot = svm_weights_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)

# Diverging bar chart of `addicted_cor` (computed earlier), ranked by strength of correlation
addicted_cor_plot <- addicted_cor |>
  mutate(direction = if_else(correlation >= 0, "Positive", "Negative")) |>
  ggplot(aes(x = reorder(feature, abs(correlation)), y = correlation, fill = direction)) +
  geom_col() +
  # Place correlation value just outside the tip of each bar (right for +, left for -)
  geom_text(
    aes(
      label = sprintf("%.2f", correlation),
      hjust = if_else(correlation >= 0, -0.15, 1.15)
    ),
    size = 3.2
  ) +
  coord_flip() +
  scale_fill_manual(
    values = c("Positive" = "#B2182B", "Negative" = "#2166AC"),
    name = "Direction"
  ) +
  scale_y_continuous(limits = c(-1, 1)) + # Fix correlation axis to full [-1, 1] range
  labs(
    title = "Feature Correlations with Smartphone Addiction",
    x = NULL,
    y = "Pearson Correlation with Addicted Label"
  ) +
  theme_minimal(base_size = 11)

# Display and export the feature correlation bar chart
addicted_cor_plot
ggsave(
  filename = file.path(fig_dir, "06_feature_correlations_addicted.png"),
  plot = addicted_cor_plot,
  width = 9,
  height = 6,
  dpi = 300,
  bg = "white"
)

# Count predicted 1s vs. 0s on the unlabeled test set and format "count (pct%)" labels
test_pred_summary <- test_preds |>
  count(.pred_class) |>
  mutate(
    pct = n / sum(n),
    label = sprintf("%s (%.1f%%)", scales::comma(n), pct * 100),
    class_label = if_else(.pred_class == "1", "Addicted (1)", "Not Addicted (0)")
  )

# Bar chart comparing predicted class counts (useful sanity check against training balance)
svm_test_dist_plot <- ggplot(
  test_pred_summary,
  aes(x = class_label, y = n, fill = class_label)
) +
  geom_col(width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = label), vjust = -0.5, size = 3.8, fontface = "bold") +
  # Format counts with commas and add 12% top padding so labels don't clip
  scale_y_continuous(
    labels = scales::comma,
    expand = expansion(mult = c(0, 0.12))
  ) +
  scale_fill_manual(
    values = c("Addicted (1)" = "#B2182B", "Not Addicted (0)" = "#2166AC")
  ) +
  labs(
    title = "Predicted Class Distribution on Unlabeled Test Data (Linear SVM)",
    x = "Predicted Class",
    y = "Number of Observations"
  ) +
  theme_minimal(base_size = 11)

# Display and export the test prediction distribution chart
svm_test_dist_plot
ggsave(
  filename = file.path(fig_dir, "07_test_prediction_distribution.png"),
  plot = svm_test_dist_plot,
  width = 7,
  height = 5.5,
  dpi = 300,
  bg = "white"
)
