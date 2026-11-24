library(tidyverse)
library(tidymodels)
library(janitor)
library(rstatix)
library(rpart)
library(rpart.plot)
raw_data <- read.csv("Smartphone_Usage_And_Addiction_Analysis.csv")
summary(raw_data$age)

clean_data <- raw_data |>
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


clean_data |> ggplot(aes(x=gaming_hours, y=work_study_hours)) + geom_point(cex=1) +
  labs(x= "Hours Gaming", y="Hours on Work/Study")

table(clean_data$stress_level, clean_data$addicted_label, useNA = "no")
table(clean_data$gender, clean_data$addicted_label, useNA = "no")
table(clean_data$academic_work_impact, clean_data$addicted_label, useNA = "no")

cor(clean_data$weekend_screen_time, clean_data$social_media_hours)
cor(clean_data$daily_screen_time_hours, clean_data$social_media_hours)
cor(clean_data$daily_screen_time_hours, clean_data$sleep_hours)
cor(clean_data$social_media_hours, clean_data$sleep_hours)
cor(clean_data$social_media_hours, clean_data$work_study_hours)

#Make Classification Tree
set.seed(3220)
new_clean_data <- drop_na(clean_data) |> mutate(addicted_label = as.factor(addicted_label))
rec <- recipe(addicted_label ~ ., new_clean_data) |> 
  step_naomit(addicted_label)


data_split <- initial_split(new_clean_data, prop=.6, strata=addicted_label)
tree_addicted <- decision_tree(
  mode = "classification", tree_depth = 4,
  min_n = 4
)

wflow_tree <- workflow() |>
  add_recipe(rec) |>
  add_model(tree_addicted)

fit_tree <- fit(wflow_tree, new_clean_data)

fit_tree
fit_tree |> extract_fit_engine() |> # The plot function expects an rpart model.
  rpart.plot::rpart.plot(roundint=FALSE)

#Figure out what the model means and how to measure it
