library(xgboost)
library(caTools)  # For sample.split function
library(dplyr)
library(caret)
# read training dataset



train<- read.csv("train1.csv")
test<- read.csv("test1.csv")

cols_to_remove <- c(1,2,3,113,112,111,110,115,114,84, 86, 88, 90, 91, 93, 94, 96, 98, 100, 101, 103, 105, 107)

train$Choice_number <- ifelse(train$Ch1 == 1, 1,
                       ifelse(train$Ch2 == 1, 2,
                              ifelse(train$Ch3 == 1, 3,
                                     ifelse(train$Ch4 == 1, 4, NA))))
train$Choice_name <- ifelse(train$Ch1 == 1, "Ch1",
                              ifelse(train$Ch2 == 1, "Ch2",
                                     ifelse(train$Ch3 == 1, "Ch3",
                                            ifelse(train$Ch4 == 1, "Ch4", NA))))

test$Choice <- ifelse(test$Ch1 == 1, "Ch1", 
                       ifelse(test$Ch2 == 1, "Ch2", 
                              ifelse(test$Ch3 == 1, "Ch3", 
                                     ifelse(test$Ch4 == 1, "Ch4", NA))))

#train$milesind <- ifelse(train$milesind == 2, 75, 
                              #ifelse(train$milesind == 8, 375, 
                                     #ifelse(train$milesind == 4, 175, 
                                            ifelse(train$milesind == 1, 45,
                                                   ifelse(train$milesind == 3, 125,
                                                          ifelse(train$milesind == 6, 275,
                                                                 ifelse(train$milesind == 7, 325, 0)))))))






train$Choice_name<- as.factor(train$Choice_name)
label<- as.numeric(train[, "Choice_number"])-1
Trained_train<- train[, -cols_to_remove]

#categorical_cols <- c("segment", "year", "miles", "night", "gender", "age", "educ", "region", "Urb", "income", "ppark")
#encoded_df1 <- as.data.frame(model.matrix(~.-1, data = Trained_train[,  categorical_cols]))
#encoded_df2 <- as.data.frame(model.matrix(~.-1, data = test_data[,  categorical_cols]))
#cols <- c("segment", "year", "miles", "night", "gender", "age", "educ", "region", "Urb", "income", "ppark", "nightind", "segmentind", "yearind", "genderind", "ageind", "educind", "regionind", "Urbind", "incomeind", "pparkind")
#Trained_train <- cbind(Trained_train[, !colnames(Trained_train) %in% cols], encoded_df1)
#test_data <- cbind(test_data[, !colnames(test_data) %in% cols], encoded_df2)

n=nrow(train)
train.index=sample(n,floor(0.75*n))
train.data= as.matrix(Trained_train[train.index,])
train.label= label[train.index]
test.data=as.matrix (Trained_train[-train.index,])
test.label= label[-train.index]

xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test= xgb.DMatrix(data=test.data,label=test.label)


param <- list(
  objective = "multi:softprob",  # For regression task (other objectives for classification, etc.)
  max_depth = 3,                  # Maximum depth of the trees
  eta = 0.06,
  gamma= 3,
  subsample= 0.75,
  colsample_bytree= 0.75 ,
  eval_metric="mlogloss",
  num_class= 4,
  booster="gbtree"

)

xgb.fit= xgb.train(params=param,
                   data=xgb.train,
                   nrounds = 250,
                   nthreads=1,
                   early_stopping_rounds = 50,
                   watchlist = list(val1=xgb.train,val2=xgb.test),
                   verbose = 0
                   )

xgb.fit

xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(train$Choice_name)

xgb.pred$prediction = colnames(xgb.pred)[max.col(xgb.pred)]
xgb.pred$label = levels(train$Choice_name)[test.label+1]

result = sum(xgb.pred$prediction== xgb.pred$label)/ nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))



test_data = as.matrix(test[,-cols_to_remove])
#test_data.label = as.numeric(test[,"Choice"])-1 
xgb.test.data = xgb.DMatrix(data= test_data)

xgb.pred.test = predict( xgb.fit, xgb.test.data, reshape=T, output_margin=FALSE)
xgb.pred.test = as.data.frame(xgb.pred.test)
colnames(xgb.pred.test) = c("Ch1","Ch2","Ch3","Ch4")
#xgb.pred.test$prediction = colnames(xgb.pred.test)[max.col(xgb.pred.test)]

submission <- data.frame(No = test$No, xgb.pred.test)

write.csv(submission, file = "xgboost.csv", row.names = FALSE)
