install.packages("psych")
install.packages("car")
install.packages("MASS")
install.packages("tidyverse")
install.packages("ggpubr")
install.packages("rstatix")

seller_nps <- read.csv(file = 'C:/Users/csteeves/Desktop/Seller Analysis/seller nps combined.csv')

head(seller_nps)
summary(seller_nps)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(dplyr)

seller_res <- cor.test(seller_nps$nps, seller_nps$account_age, method = "pearson")
seller_res

seller_res2 <-cor.test(seller_nps$nps, seller_nps$account_age,  method = "spearman")
seller_res2

# Create a Q-Q plot
seller_nps <- data.frame (seller_nps)
seller_nps_part <- select (seller_nps, nps, account_age)
qqnorm(seller_nps_part$nps, pch = 1, frame = FALSE)
qqline(seller_nps_part$nps, col = "red", lwd = 2)  # Add a reference line
qqnorm(seller_nps_part$account_age, pch = 1, frame = FALSE)
qqline(seller_nps_part$account_age, col = "red", lwd = 2)


ggscatter(seller_nps_part, x = "account_age", y = "nps", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Account Age (months)", ylab = "NPS (1-10)")


# Fit a linear model
y<- seller_nps_part$nps
x<- seller_nps_part$account_age

model <- lm(y ~ x, data = seller_nps_part)

# View the summary of the model
summary(model)

# Plot the data and the regression line
plot(x, y, main = "Simple Linear Regression", xlab = "X", ylab = "Y")
abline(model, col = "red")
#######################################################################
buyer_nps <- read.csv(file = 'C:/Users/csteeves/Desktop/Seller Analysis/buyer nps combined.csv')

head(buyer_nps)
summary(buyer_nps)


buyer_res <- cor.test(buyer_nps$nps, buyer_nps$account_age, method = "pearson")
buyer_res

buyer_res2 <-cor.test(buyer_nps$nps, buyer_nps$account_age,  method = "spearman")
buyer_res2

# Create a Q-Q plot
buyer_nps <- data.frame (buyer_nps)
buyer_nps_part <- select (buyer_nps, nps, account_age)
qqnorm(buyer_nps_part$nps, pch = 1, frame = FALSE)
qqline(buyer_nps_part$nps, col = "red", lwd = 2)  # Add a reference line
qqnorm(buyer_nps_part$account_age, pch = 1, frame = FALSE)
qqline(buyer_nps_part$account_age, col = "red", lwd = 2)


ggscatter(buyer_nps_part, x = "account_age", y = "nps", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "Account Age (months)", ylab = "NPS (1-10)")


# Fit a linear model
y<- buyer_nps_part$nps
x<- buyer_nps_part$account_age

model <- lm(y ~ x, data = buyer_nps_part)

# View the summary of the model
summary(model)

# Plot the data and the regression line
plot(x, y, main = "Simple Linear Regression", xlab = "X", ylab = "Y")
abline(model, col = "red")




##### Random Forest
nps <- read.csv(file = 'C:/Users/csteeves/Desktop/Seller Analysis/Buyer NPS ML.csv')
str(nps)
nps$nps_group <- factor(nps$nps_group)
str(nps)
# Load necessary packages
library(randomForest)
library(datasets)

# Load the Iris dataset
data(nps)

# View the first few rows of the dataset
head(nps)

# Set the seed for reproducibility
set.seed(42)

# Split the data into training and testing sets
sample_size <- floor(0.7 * nrow(nps))
train_indices <- sample(seq_len(nrow(nps)), size = sample_size)

train_data <- nps[train_indices, ]
test_data <- nps[-train_indices, ]

# Fit the random forest model
rf_model <- randomForest(nps_group ~ ., data = train_data, ntree = 100, mtry = 2, importance = TRUE)

# Print the model summary
print(rf_model)

# Evaluate variable importance
importance_values <- importance(rf_model)
print(importance_values)
varImpPlot(rf_model)

# Make predictions on the test set
predictions <- predict(rf_model, test_data)

# Create a confusion matrix to evaluate accuracy
confusion_matrix <- table(test_data$nps_group, predictions)
print(confusion_matrix)


# Get importance values
importance_values <- importance(rf_model)

# View importance values
print(importance_values)

# Visualize importance
varImpPlot(rf_model)