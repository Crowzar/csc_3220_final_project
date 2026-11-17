#loading initial library + reading in / filtering code
library(tidyverse)

data <- read.csv("Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE)
colSums(is.na(data))
summary(data)

data_pruned <- data |> filter(if_all(-addicted_label, ~ !is.na(.) & trimws(as.character(.)) != ""))
colSums(is.na(data_pruned)) #this decision was made because we still have 400k entries here

#to show all numerical outliers
boxplot(data_pruned$daily_screen_time_hours, data_pruned$social_media_hours, 
        data_pruned$gaming_hours, data_pruned$work_study_hours, data_pruned$sleep_hours, 
        data_pruned$notifications_per_day, data_pruned$app_opens_per_day, 
        data_pruned$weekend_screen_time)

#to zoom in on categories with outliers
boxplot(data_pruned$social_media_hours, 
        data_pruned$work_study_hours,
        data_pruned$weekend_screen_time)

#running through outliers and pruning them
x <- data_pruned$social_media_hours
Q1 <- quantile(data_pruned$social_media_hours, 0.25)
Q3 <- quantile(data_pruned$social_media_hours, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data_pruned <- data_pruned[(x >= lb & x <= ub),]

#outlier 2

x <- data_pruned$work_study_hours
Q1 <- quantile(data_pruned$work_study_hours, 0.25)
Q3 <- quantile(data_pruned$work_study_hours, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data_pruned <- data_pruned[(x >= lb & x <= ub),]

#outlier 3

x <- data_pruned$weekend_screen_time
Q1 <- quantile(data_pruned$weekend_screen_time, 0.25)
Q3 <- quantile(data_pruned$weekend_screen_time, 0.75)
IQR <- Q3 - Q1

lb <- Q1 - 1.5 * IQR
ub <- Q3 + 1.5 * IQR

data_pruned <- data_pruned[(x >= lb & x <= ub),]

