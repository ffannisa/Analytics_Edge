svm <- read.csv("svm.csv")
nn <- read.csv("nn.csv")
xgboost <- read.csv("xgboost.csv")

# Define the weights for each model
weight_svm <- 0.2
weight_nn <- 0.2
weight_xgboost <- 0.6

# Calculate the weighted average probabilities for each class
weighted_avg_probs <- data.frame(
  No = svm$No,  # Assuming "No" column is present in the predictions
  Ch1 = (svm$Ch1 * weight_svm) + (nn$Ch1 * weight_nn) + (xgboost$Ch1 * weight_xgboost),
  Ch2 = (svm$Ch2 * weight_svm) + (nn$Ch2 * weight_nn) + (xgboost$Ch2 * weight_xgboost),
  Ch3 = (svm$Ch3 * weight_svm) + (nn$Ch3 * weight_nn) + (xgboost$Ch3 * weight_xgboost),
  Ch4 = (svm$Ch4 * weight_svm) + (nn$Ch4 * weight_nn) + (xgboost$Ch4 * weight_xgboost)
)

# Write the weighted average probabilities to a new CSV file
write.csv(weighted_avg_probs, file = "weighted_average_predictions.csv", row.names = FALSE)