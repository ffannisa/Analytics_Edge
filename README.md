# The Analytics Edge - Kaggle Data Competition
# Predicting Car Safety features Package

Result: A mLogloss of 1.25 in the private leaderboard and 1.26 in the public leaderboard. Manage to be able to be in the top 3 groups based on leaderboard.

We have 4 files in our directory: the prediction done in SVM (under svm.R), Neural Network (under neuralnetwork.R), XGBoost (under xgboost.R) and lastly the weighted average of all the mentioned predictions (under weighted average.R).

We used machine learning algorithms such as SVM, neural network, and XGBoost. Combining their predictions by putting different weightage on the ML model prediction.

To run the probability prediction for submission:

*User should run svm.R, neuralnetwork.R and xgboost.R.

*Once all 3 files are runned, 3 csv files should be available under the user's directory.

*Now, user can run the file named weighted average.R, this file will result in the final csv for submission.

(under the name weighted_average_predictions.csv)
