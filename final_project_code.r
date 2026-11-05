library(tidyverse)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv", header = TRUE)

colSums(is.na(data))

summary(data)
