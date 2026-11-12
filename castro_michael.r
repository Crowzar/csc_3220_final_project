library(tidyverse)
library(janitor)
library(rstatix)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis.csv", header = TRUE)

colSums(is.na(data))

get_dupes(data)

identify_outliers(data)

summary(data)
