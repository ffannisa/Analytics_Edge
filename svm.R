# Load the required libraries
library(e1071)

# Read the training and test datasets
training_data <- read.csv("train1.csv")
test_data1 <- read.csv("test1.csv")

# Preprocess the data (remove irrelevant columns, scaling, encoding, etc.)
# The preprocessing steps will depend on the nature of your data. Ensure that the preprocessing applied to the training data is also applied to the test data.
cols_to_remove <- c(84, 86, 88, 90, 91, 93, 94, 96, 98, 100, 101, 103, 105, 107, 109)

# Remove the specified columns

training_data <- training_data[, -cols_to_remove]
training_data <- training_data[, -(1:3)]

test_data <- test_data1[, -cols_to_remove]
test_data <- test_data[, -(1:3)]

training_data$Ch1 <- as.numeric(training_data$Ch1)
training_data$Ch2 <- as.numeric(training_data$Ch2)
training_data$Ch3 <- as.numeric(training_data$Ch3)
training_data$Ch4 <- as.numeric(training_data$Ch4)

# Combine the response variables into a single target variable representing the chosen choice (1, 2, 3, or 4)
training_data$chosen_choice <- 0
training_data$chosen_choice[training_data$Ch1 == 1] <- 1
training_data$chosen_choice[training_data$Ch2 == 1] <- 2
training_data$chosen_choice[training_data$Ch3 == 1] <- 3
training_data$chosen_choice[training_data$Ch4 == 1] <- 4

# Convert the chosen_choice column to factor
training_data$chosen_choice <- as.factor(training_data$chosen_choice)


# Train the SVM model on the training portion
svm_model <- svm(training_data$chosen_choice ~ ., data = training_data[, 1:91], probability = TRUE)

predictions <- predict(svm_model, newdata = test_data[, 1:91], probability = TRUE)


# Extract the predicted probabilities for each class
predicted_probs <- attr(predictions, "probabilities")

# Create a new data frame for the CSV output
output_data <- data.frame(
  No = test_data1$No,  # Assuming the "No" column is present in the test data
  Ch1 = predicted_probs[, 2],  # Ch1 predictions (Column 1 of predicted probabilities)
  Ch2 = predicted_probs[, 1],  # Ch2 predictions (Column 2 of predicted probabilities)
  Ch3 = predicted_probs[, 3],  # Ch3 predictions (Column 3 of predicted probabilities)
  Ch4 = predicted_probs[, 4]   # Ch4 predictions (Column 4 of predicted probabilities)
)

# Write the data frame to a CSV file
write.csv(output_data, file = "svm.csv", row.names = FALSE)