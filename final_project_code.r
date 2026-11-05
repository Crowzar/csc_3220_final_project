library(tidyverse)

data <- read.csv("csc_3220_final_project/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv", header = TRUE)

colSums(is.na(data))

summary(data)

data2 <- data
for (i in 1:ncol(data)){
    data2[ ,i][is.na(data2[ ,i])] <- mean(data2[ ,i], na.rm = TRUE)
}

colSums(is.na(data2))

summary(data2)
