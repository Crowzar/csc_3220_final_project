library(tidyverse)
library(janitor)
library(rstatix)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE)

colSums(is.na(data))

data2 <- data |> filter(if_all(-addicted_label, ~ !is.na(.) & trimws(as.character(.)) != ""))

colSums(is.na(data2))

get_dupes(data2)

summary(data2)

nrow(data2)
