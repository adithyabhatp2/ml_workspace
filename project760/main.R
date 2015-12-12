census.data <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census-income-reordered-withHeader.data.csv")
census.test <- read.csv("D:/Software/gitRepository/ml_workspace/project760/input/census-income-reordered-withHeader.test.csv")

train_data = census.data[3:42]
test_data = census.test[3:42]

train_labels = as.factor(census.data[[1]])
test_labels = as.factor(census.test[[1]])

#census_pred = knn(train_data, test_data, train_labels, k=1)
