
# title: "Mini Project 1: Breast Cancer Prediction"
# author: "Shilpa Gopal"
# language: R 

# Mini Project 1: Breast Cancer Prediction"


# Loading Packages
## 1. Loading the dataset and explore its structure

library("ggplot2")
library("caTools")
library("corrplot")
library("dplyr")
library("corrplot")
library("caret")


# Read Dataset
main_dataset <- read.csv("/Users/sheme/sheme/Codes/R_File/multimedia_mini_project/breast_cancer.csv", header = TRUE)

# Viewing first few rows of Dataset
head(main_dataset)

# Display the structure of the dataset
## 2. Calculate summary statistics (mean, median, minimum, and maximum) for numerical columns
str(main_dataset)

# Displaying Dimension of Dataset
dim(main_dataset)

# Summary of the Dataset
## 2. Calculate summary statistics (mean, median, minimum, and maximum) for numerical columns
summary(main_dataset)

# Remove missing value NAs (if applicable)
## Check for missing values(NA) and handling them
main_dataset <- main_dataset[-33]
summary(main_dataset)

# Frequency of Cancer Diagnosis
# Number of women affected with benign and malignant tumor
main_dataset %>% count(diagnosis)

# Percentage of women affected with benign and malignant tumor
main_dataset %>% count(diagnosis) %>% 
  group_by(diagnosis) %>%
  summarize(perc_dx = round((n / 569) * 100, 2))

# Data Visualization
# Frequency of cancer diagnosis using tabular calculation
diagnosis.table <- table(main_dataset$diagnosis)
colors <- terrain.colors(2)
diagnosis.prop.table <- prop.table(diagnosis.table) * 100
diagnosis.prob.df <- as.data.frame(diagnosis.prop.table)
pielabels <- sprintf("%s - %3.1f%s", diagnosis.prob.df[,1], diagnosis.prop.table, "%")
pie(diagnosis.prop.table, labels = pielabels, clockwise = TRUE, col = colors, border = "gainsboro", radius = 0.8, main = "Frequency of Cancer Diagnosis")
legend(1, .4, legend = diagnosis.prob.df[,1], cex = 0.7, fill = colors)

# Correlation plot 
## 5. Compute and visualize correlation.
# Correlation plot - relationship with variator 
c <- cor(main_dataset[,3:31])
corrplot(c, order = "hclust", tl.cex = 0.7)


# Columns comparions 
# Comparing the radius column, area column, concavity column of benign and malignant stage
ggplot(main_dataset, aes(x=diagnosis, y=radius_mean, fill="pink")) + geom_boxplot(fill = "yellow") + ggtitle("Radius of Benign Vs Malignant")
ggplot(main_dataset, aes(x=diagnosis, y=area_mean)) + geom_boxplot() + ggtitle("Area of Benign and Malignant")
ggplot(main_dataset, aes(x=diagnosis, y=concavity_mean)) + geom_boxplot() + ggtitle("Concavity of Benign and Malignant")

# Observation from the box plot - malignant cells have higher radius, area, concavity mean than benign cells

# 6. Barplot comparision
# Barplot for analyzing the tumors of the affected women
ggplot(main_dataset, aes(x=diagnosis, fill = texture_mean)) + geom_bar() + ggtitle("Women affected in Benign and Malignant Tumor")

# Women affected at higher levels based on mean from the analysis of boxplot
sel_data <- main_dataset[main_dataset$radius_mean > 10 & main_dataset$radius_mean < 15 & main_dataset$compactness_mean > 0.1,]

ggplot(sel_data, aes(x=diagnosis, y=radius_mean, fill = diagnosis)) + geom_col() + ggtitle("Women affected at higher level based on mean")

# Density plot based on texture mean
ggplot(main_dataset, aes(x=texture_mean, fill = as.factor(diagnosis))) + geom_density(alpha = 0.4) + ggtitle("Texture mean for Benign Vs Malignant")

# Barplot for area_se
ggplot(main_dataset, aes(x=area_se > 15, fill = diagnosis)) + geom_bar(position = "fill") + ggtitle("Area se for Benign Vs Malignant")

# Checking distribution of Data via histograms
## 7. Using histograms visualize the distribution of numerical features.
ggplot(main_dataset, aes(x=concavity_mean, fill = diagnosis)) + geom_histogram(binwidth = 10) + ggtitle("Concavity mean for Benign Vs Malignant")
ggplot(main_dataset, aes(x = texture_se)) + geom_histogram(binwidth = 10) + facet_wrap(~ diagnosis) + ggtitle("Texture se mean for Benign and Malignant")
ggplot(main_dataset, aes(x = perimeter_mean)) + geom_histogram(binwidth = 10) + facet_wrap(~ diagnosis) + ggtitle("Perimeter mean for Benign and Malignant")


# Train the Algorithm
## Split the data into training and testing sets using Logistic Regression
main_dataset$diagnosis <- factor(main_dataset$diagnosis, levels = c("B", "M"))

split <- sample.split(main_dataset$diagnosis, SplitRatio = 0.65)
main_dataset <- main_dataset[-33]
training_set <- subset(main_dataset, split == TRUE)
test_set <- subset(main_dataset, split == FALSE)

# Normalization process
## 8. Perform feature scaling or normalization.
training_set[,3:32] <- scale(training_set[,3:32])
test_set[,3:32] <- scale(test_set[,3:32])

# 9. Train and evaluate machine learning models for breast cancer diagnosis prediction.
# Create training and testing sets
set.seed(1234)
data_index <- createDataPartition(main_dataset$diagnosis, p = 0.7, list = FALSE)
train_data <- main_dataset[data_index, -1]
test_data <- main_dataset[-data_index, -1]

# Building Model
## 10. Predict using the trained random forest model on the test set
fitControl <- trainControl(
  method="cv",
  number = 5,
  preProcOptions = list(thresh = 0.99), # threshold for PCA preprocess
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Random Forest model
model_rf <- train(
  diagnosis ~ .,
  train_data,
  method = "ranger",
  metric = "ROC",
  preProcess = c('center', 'scale'),
  trControl = fitControl
)

# Predict using the trained random forest model on the test set
pred_rf <- predict(model_rf, test_data)

# Convert the actual column to a factor with levels "B" and "M"
results_df <- tibble(
  predicted = pred_rf,
  actual = test_data$diagnosis
)
results_df$actual <- factor(results_df$actual, levels = c("B", "M"))

# Calculate confusion matrix
## 11. Calculate confusion matrix
cm_rf <- confusionMatrix(data = results_df$predicted, reference = results_df$actual, positive = "M")

# printing
print(cm_rf)

