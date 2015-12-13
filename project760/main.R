census.data <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census.data.v2.csv", header=TRUE, na.strings="NA")
census.test <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census.test.v2.csv", header=TRUE, na.strings="NA")



train_data = census.data[3:42]
test_data = census.test[3:42]

train_labels = as.factor(census.data[[1]])
test_labels = as.factor(census.test[[1]])

tiny.set <- census.data[c(1,2,3,4,5), 35:42]

base_train <- census.data[1:38];
base_test <- census.test[1:38];

# KNN
# 42, 42, 40, 39 - big losers. 38, 37, 36 are oook

knn_train <- na.omit(base_train)[3:38]
knn_test <- na.omit(base_test)[3:38]
knn_train_labels <- as.factor(na.omit(base_train)[[1]])

library(class)
knn_test_labels = knn(knn_train, knn_test, knn_train_labels, k=1)
