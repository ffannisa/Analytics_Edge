# Load the required libraries
library(keras)

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


# Convert the response variables to numeric format (1 for chosen, 0 for not chosen)
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

# Define the neural network architecture
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(91)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 4, activation = "softmax")  # Output layer with 4 units for 4 choices and softmax activation

# Compile the model
model %>% compile(
  loss = "sparse_categorical_crossentropy",  # Sparse categorical cross-entropy for multi-class classification
  optimizer = "adam",
  metrics = c("accuracy")
)


# Prepare the input and output data for training
x_train <- as.matrix(training_data[, 1:91])  # Attributes from column 1 to 91
y_train <- as.numeric(training_data$chosen_choice) - 1  # Subtract 1 to convert the target to 0-based index

# Train the model on the entire training dataset
history <- model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 50,
  batch_size = 32
)

# Now, use the trained model to make predictions on the test data
x_test <- as.matrix(test_data[, 1:91])  # Attributes from column 1 to 91

# Make predictions
predictions <- model %>% predict(x_test)

# The predictions will be in probability form (since the output layer uses softmax activation)
# You can convert the probabilities back to the original classes (1, 2, 3, 4) using the "which.max" function
predicted_classes <- apply(predictions, 1, which.max)

# Create a new data frame to store the test predictions
test_predictions <- data.frame(No = test_data1$No, Ch1 = 0, Ch2 = 0, Ch3 = 0, Ch4 = 0)

# Assign the predicted classes to the corresponding columns in the test predictions data frame
test_predictions$Ch1[predicted_classes == 1] <- 1
test_predictions$Ch2[predicted_classes == 2] <- 1
test_predictions$Ch3[predicted_classes == 3] <- 1
test_predictions$Ch4[predicted_classes == 4] <- 1

# Save the predictions to a new CSV file
write.csv(test_predictions, "nn.csv", row.names = FALSE)