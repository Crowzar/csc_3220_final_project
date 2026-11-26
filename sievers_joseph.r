library(tidyverse)
library(tidymodels)
library(janitor)
library(rstatix)
library(rpart)
library(rpart.plot)

# Use readr::read_csv() instead of base read.csv() (~3x faster on ~988k rows)
raw_data <- read_csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis.csv", show_col_types = FALSE)

# Quick summary (min, quartiles, median, mean, max, NA count) of the age column
summary(raw_data$age)

clean_data <- raw_data |>
  mutate(
    # Numerical predictors (ignoring id and addicted_label):
    # replace any NA values with the column's mean
    across(
      where(is.numeric) & !c(id, addicted_label),
      function(col) {
        col[is.na(col)] <- mean(col, na.rm = TRUE)
        col
      }
    ),
    # Categorical predictors: convert any blank/whitespace strings to NA first,
    # then replace NA values with the most frequent category (mode)
    across(
      where(is.character),
      function(col) {
        col[trimws(col) == ""] <- NA
        col[is.na(col)] <- names(which.max(table(col)))
        col
      }
    )
  )

# Scatterplot comparing gaming hours vs. work/study hours
# Sample 5,000 rows so ggplot doesn't hang/timeout rendering ~988k overlapping points
set.seed(3220)
clean_data |>
  slice_sample(n = 5000) |>
  ggplot(aes(x = gaming_hours, y = work_study_hours)) +
  geom_point(alpha = 0.4, size = 1) +
  labs(
    title = "Gaming Hours vs. Work/Study Hours (5,000-Row Sample)",
    x = "Hours Gaming",
    y = "Hours on Work/Study"
  )

# Cross-tabulations of categorical predictors against addicted_label (excluding NAs)
table(clean_data$stress_level, clean_data$addicted_label, useNA = "no")
table(clean_data$gender, clean_data$addicted_label, useNA = "no")
table(clean_data$academic_work_impact, clean_data$addicted_label, useNA = "no")

# Pairwise Pearson correlations between selected numeric screen-time/lifestyle variables
cor(clean_data$weekend_screen_time, clean_data$social_media_hours)
cor(clean_data$daily_screen_time_hours, clean_data$social_media_hours)
cor(clean_data$daily_screen_time_hours, clean_data$sleep_hours)
cor(clean_data$social_media_hours, clean_data$sleep_hours)
cor(clean_data$social_media_hours, clean_data$work_study_hours)

# Create output directory for exported figures if it does not already exist
fig_dir <- if (dir.exists("csc_3220_final_project")) {
  file.path("csc_3220_final_project", "classificationTree_figures")
} else {
  "classificationTree_figures"
}
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# Compute correlation matrix across all numeric columns (excluding id);
# pairwise.complete.obs handles the unlabeled (NA) rows in addicted_label
corr_mat <- clean_data |>
  select(where(is.numeric), -id) |>
  cor(use = "pairwise.complete.obs")

# Convert the correlation matrix into a long data frame for ggplot2
corr_df <- as.data.frame(as.table(corr_mat)) |>
  rename(var1 = Var1, var2 = Var2, corr = Freq)

# Build a color-coded correlation heatmap with rounded values in each cell
corr_plot <- ggplot(corr_df, aes(x = var1, y = fct_rev(var2), fill = corr)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", corr)), size = 3) +
  scale_fill_gradient2(
    low = "#2166AC",    # Blue for negative correlations
    mid = "white",      # White for zero correlation
    high = "#B2182B",   # Red for positive correlations
    midpoint = 0,
    limits = c(-1, 1),
    name = "Correlation"
  ) +
  coord_fixed() +       # Keep heatmap cells square
  labs(
    title = "Correlation Matrix of Numeric Variables",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1), # Rotate x labels so they don't overlap
    panel.grid = element_blank()
  )

# Export the correlation matrix heatmap
ggsave(
  filename = file.path(fig_dir, "01_correlation_matrix.png"),
  plot = corr_plot,
  width = 9,
  height = 7.5,
  dpi = 300,
  bg = "white"
)

# Use the competition's existing split without altering addicted_label values:
# - train_data: rows where addicted_label is known (0 or 1), cast to factor for classification
#   with level "1" first so yardstick treats 1 (addicted) as the positive class
# - test_data:  rows where addicted_label is unknown (NA), used for final predictions
train_data <- clean_data |>
  filter(!is.na(addicted_label)) |>
  mutate(addicted_label = factor(addicted_label, levels = c(1, 0)))

test_data <- clean_data |>
  filter(is.na(addicted_label)) |>
  select(-addicted_label)

# Define preprocessing recipe on training data:
# - predict addicted_label using all other columns
# - mark `id` as an identifier so it is excluded from model predictors
rec <- recipe(addicted_label ~ ., data = train_data) |>
  update_role(id, new_role = "id")

# Specify a classification tree using the rpart engine:
# - tree_depth = 4 limits the tree to at most 4 levels of splits
# - min_n = 4 requires at least 4 observations in a node to attempt a split
# - xval = 0 skips rpart's redundant 10-fold internal CV (we use tree_depth = 4 directly)
# - maxsurrogate = 0 skips surrogate split search since predictors have no NAs left
tree_addicted <- decision_tree(
  mode = "classification",
  tree_depth = 4,
  min_n = 4
) |>
  set_engine("rpart", xval = 0, maxsurrogate = 0)

# Bundle the preprocessing recipe and model specification into a single workflow
wflow_tree <- workflow() |>
  add_recipe(rec) |>
  add_model(tree_addicted)

# Fit the classification tree on the labeled training set
fit_tree <- fit(wflow_tree, data = train_data)

# Print the fitted tree rules and node statistics to the console
fit_tree

# Predict addicted_label on the unlabeled test set
test_preds <- augment(fit_tree, new_data = test_data)
head(test_preds |> select(id, .pred_class, .pred_1, .pred_0))
table(test_preds$.pred_class)

# Extract the underlying rpart model object needed by rpart.plot
tree_engine <- fit_tree |> extract_fit_engine()

# Export the tree diagram to PNG (extra = 104 shows predicted class, both class
# probabilities P(1) and P(0), and % of training rows in each node)
png(
  filename = file.path(fig_dir, "02_classification_tree.png"),
  width = 1800,
  height = 1200,
  res = 200
)
rpart.plot::rpart.plot(
  tree_engine,
  extra = 104,
  roundint = FALSE, # Keep decimal split thresholds instead of rounding to integers
  main = "Smartphone Addiction Classification Tree"
)
dev.off() # Close the PNG device to finalize the saved file

# Print variable importance scores (higher = larger contribution to reducing impurity)
tree_engine$variable.importance

# Put variable importance into a tidy tibble for plotting
vip_df <- tibble(
  variable = names(tree_engine$variable.importance),
  importance = as.numeric(tree_engine$variable.importance)
)

# Horizontal bar chart ranking predictors by importance
vip_plot <- ggplot(vip_df, aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "#2C7FB8") +
  coord_flip() +
  labs(
    title = "Variable Importance - Classification Tree",
    x = NULL,
    y = "Importance"
  ) +
  theme_minimal(base_size = 11)

# Export the variable importance plot
ggsave(
  filename = file.path(fig_dir, "03_variable_importance.png"),
  plot = vip_plot,
  width = 8,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# Because test_data is unlabeled, evaluate out-of-sample performance using 5-fold CV
# (lower v to 3 if you want faster CV execution)
set.seed(3220)
folds <- vfold_cv(train_data, v = 5, strata = addicted_label)

# Classification metrics (positive class = "1", the first factor level):
# - accuracy:  overall fraction of correct predictions
# - sens:      sensitivity / recall (% of actual 1s correctly caught)
# - spec:      specificity (% of actual 0s correctly identified)
# - precision: % of predicted 1s that are truly 1
# - f_meas:    F1 score (harmonic mean of precision and sensitivity)
# - roc_auc:   area under the ROC curve (1.0 = perfect, 0.5 = random guessing)
tree_metrics <- metric_set(accuracy, sens, spec, precision, f_meas, roc_auc)

tree_cv <- fit_resamples(
  wflow_tree,
  resamples = folds,
  metrics = tree_metrics,
  control = control_resamples(save_pred = TRUE)
)

# Print cross-validated evaluation metrics
collect_metrics(tree_cv)

# Collect out-of-fold predictions for confusion matrix and ROC curve
cv_preds <- collect_predictions(tree_cv)

# Compute and print the out-of-fold confusion matrix
cm <- conf_mat(cv_preds, truth = addicted_label, estimate = .pred_class)
cm

# Export the confusion matrix heatmap
cm_plot <- autoplot(cm, type = "heatmap") +
  labs(title = "Confusion Matrix - Classification Tree (5-Fold CV)")

ggsave(
  filename = file.path(fig_dir, "04_confusion_matrix.png"),
  plot = cm_plot,
  width = 6.5,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

# Export the out-of-fold ROC curve
roc_plot <- roc_curve(cv_preds, truth = addicted_label, .pred_1) |>
  autoplot() +
  labs(title = "ROC Curve - Smartphone Addiction Classification Tree (5-Fold CV)")

ggsave(
  filename = file.path(fig_dir, "05_roc_curve.png"),
  plot = roc_plot,
  width = 7,
  height = 5.5,
  dpi = 300,
  bg = "white"
)
