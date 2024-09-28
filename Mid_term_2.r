#### Preparatory Steps ####

## To clear the working station
rm(list = ls())

# Import necessary library
library(xgboost)
library(glmnet)
library(data.table)
library(ggplot2)


########################################################################
#### Data Generating Process ####
# Set seed to create random numbers that can be reproduced
set.seed(1)

# Training sample
n_train <- 10000

# Validation set
n_val <- 5000

# Test sample
n_test <- 100000

# Coefficients
betas <- c(2, 1.5, 1, 0.5)
p <- length(betas)

# Generating the predictors in a matrix form
x_train <- matrix(rnorm(n_train * p), nrow = n_train, ncol = p)
x_val <- matrix(rnorm(n_val * p), nrow = n_val, ncol = p)
x_test <- matrix(rnorm(n_test * p), nrow = n_test, ncol = p)

# Generate error terms
epsilon_train <- rnorm(n_train)
epsilon_val <- rnorm(n_val)
epsilon_test <- rnorm(n_test)

# Calculating the target variables
y_train <- x_train %*% betas + epsilon_train
y_val <- x_val %*% betas + epsilon_val
y_test <- x_test %*% betas + epsilon_test

# Signal to Noise Ratio
SNR = sum(betas^2) / (sum(betas^2) + 1^2)     # A measure of the signal-to-noise ratio, var(E(y|x)) / var(y)
print(SNR)

########################################################################
########## TASK 1 ##########

#### Fitting XGBoost ####
# We first implement DMatrix in order to optimize XGBoost
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dval <- xgb.DMatrix(data = x_val, label = y_val)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# A high number of rounds to begin with
num_rounds <- 1000

# Fitting the model on the training set, and using validation for num_trees
watch_list <- list(test = dtest, train = dtrain)

# Use CV on the dval data set
bst_cv = xgb.cv(data = dval,
                objective = "reg:squarederror",
                eta = 0.1,
                nfold = 5,
                max_depth = 4,
                nrounds = num_rounds,
                watchlist = watch_list,
                early_stopping_rounds = 20,
                verbose = 0 )

# Optimal number of trees
best_ntrees <- as.numeric(bst_cv[8])
print(paste("Optimal number of trees: ", best_ntrees))

bst_model = xgboost(data = dtrain, 
                    max.depth = 4,
                    objective = "reg:squarederror",
                    eta = 0.1,
                    nrounds = as.numeric(bst_cv[8]),
                    early_stopping_rounds = 20,
                    verbose = 0 )

# Predicting on the test_set
y_test_pred_xgb <- predict(bst_model, newdata = x_test)

# Compute R^2
rss_xgb <- mean((y_test - y_test_pred_xgb)^2)
# tss_xgb <- sum((y_test - mean(y_test))^2)
tss_xgb <- var(y_test)
r2_xgb <- round(1 - (rss_xgb / tss_xgb), 6)
print(paste("R2 for XGBoost:", r2_xgb))


#### Fitting OLS ####
# Naming columns for x_train and x_test
colnames(x_train) <- paste0("x", 1:ncol(x_train))
colnames(x_test) <- colnames(x_train)

# Converting matrices to data frames for the lm() function
df_train <- data.frame(y_train = y_train, x_train)
df_test <- data.frame(x_test) 

# Fitting the OLS model
ols_model <- lm(y_train ~ 0 + x1 + x2 + x3 + x4, data  = df_train)
sum_ols = summary(ols_model)
print(sum_ols)

# Predicting with OLS
y_pred_ols <- predict(ols_model, newdata = df_test)

# Compute R2 for OLS
rss_ols <- mean((y_test - y_pred_ols)^2)
tss_ols <- var(y_test)
r2_ols <- round(1 - (rss_ols / tss_ols), 6)
print(paste("R2 for OLS:", r2_ols))

# Compare R2 values
print(paste("Comparison of R2 values - XGBoost: ", r2_xgb, ", OLS: ", r2_ols))


########################################################################
########## TASK 2 ##########

# Gathering results in a table
results <- data.table(max_depth = integer(), rmse = numeric())

# For loop over max_depth values
for(depth in 1:4){
  bst_cv_max = xgb.cv(
    data = dval,
    objective = "reg:squarederror",
    eta = 0.1,
    nfold = 5,
    max_depth = depth,
    nrounds = num_rounds,
    watchlist = watch_list,
    early_stopping_rounds = 20,
    verbose = 0 )
  
  # Train the model on the optimal number of trees from CV
  model = xgboost(data = dtrain, 
                  max.depth = depth,
                  objective = "reg:squarederror",
                  eta = 0.1,
                  nrounds = as.numeric(bst_cv_max[8]),
                  early_stopping_rounds = 20,
                  verbose = 0 )
  # Predict test set
  test_pred <- predict(model, newdata = dtest)
  
  # Calculate RMSE for test set
  rmse <- sqrt(mean((y_test - test_pred)^2))
  
  # Store the result
  results <- rbind(results, data.table(max_depth = depth, rmse = rmse))
}
print(results)


########################################################################
########## TASK 3 ##########

# Setting the exponential transformation
x_train_e <- exp(x_train)
x_val_e <- exp(x_val)
x_test_e <- exp(x_test)

# And for the y's
y_train_e <- exp(y_train)
y_val_e <- exp(y_val)
y_test_e <- exp(y_test)

# Repeating the same task as above

## TASK 3.1 ##

#### Fitting XGBoost ####
# We first implement DMatrix in order to optimize XGBoost
dtrain_e <- xgb.DMatrix(data = x_train_e, label = y_train_e)
dval_e <- xgb.DMatrix(data = x_val_e, label = y_val_e)
dtest_e <- xgb.DMatrix(data = x_test_e, label = y_test_e)

# Fitting the model on the training set, and using validation for num_trees
watch_list_e <- list(test = dtest_e, train = dtrain_e)

# Use CV on the dval data set
bst_cv_e = xgb.cv(data = dval_e,
                  objective = "reg:tweedie",
                  eval_metric = "tweedie-nloglik@1.1",
                  eta = 0.1,
                  nfold = 5,
                  max_depth = 4,
                  nrounds = num_rounds,
                  watchlist = watch_list_e,
                  early_stopping_rounds = 20,
                  verbose = 0 )

# Optimal number of trees
best_ntrees_e <- as.numeric(bst_cv_e[8])
print(paste("Optimal number of trees: ", best_ntrees_e))

bst_model_exp = xgboost(data = dtrain_e, 
                        max.depth = 4,
                        objective = "reg:tweedie",
                        eval_metric = "tweedie-nloglik@1.1",
                        eta = 0.1,
                        nrounds = as.numeric(bst_cv_e[8]),
                        early_stopping_rounds = 20,
                        verbose = 0 )

# Predicting on the test_set
y_test_pred_xgb_e <- predict(bst_model_exp, newdata = x_test_e)

# Compute R^2
rss_xgb_e <- mean((y_test_e - y_test_pred_xgb_e)^2)
#tss_xgb_e <- sum((y_test_e - mean(y_test_e))^2)
tss_xgb_e <- var(y_test_e)
r2_xgb_e <- round(1 - (rss_xgb_e / tss_xgb_e), 6)
print(paste("R2 for XGBoost:", r2_xgb_e))


#### Fitting OLS ####
# Naming columns for x_train and x_test
colnames(x_train_e) <- paste0("x", 1:ncol(x_train_e))
colnames(x_test_e) <- colnames(x_train_e)

# Converting matrices to data frames for the lm() function
df_train_e <- data.frame(y_train_e = y_train_e, x_train_e)
df_test_e <- data.frame(x_test_e) 

# Fitting the OLS model
ols_model_exp <- lm(y_train_e ~ 0 + x1 + x2 + x3 + x4, data = df_train_e)
sum_ols_exp = summary(ols_model_exp)
print(sum_ols_exp)

# Predicting with OLS
y_pred_ols_e <- predict(ols_model_exp, newdata = df_test_e)

# Compute R2 for OLS
rss_ols_e <- mean((y_test_e - y_pred_ols_e)^2)
# tss_ols_e <- sum((y_test_e - mean(y_test_e))^2)
tss_ols_e <- var(y_test_e)
r2_ols_e <- round(1 - (rss_ols_e / tss_ols_e), 6)
print(paste("R2 for OLS:", r2_ols_e))

# Compare R2 values
print(paste("Comparison of R2 values - XGBoost: ", r2_xgb_e, ", OLS: ", r2_ols_e))


########## TASK 3.2 ##########

# Gathering results in a table
results_e <- data.table(max_depth = integer(), rmse = numeric())

# For loop over max_depth values
for(depth in 1:4){
  bst_cv_max = xgb.cv(
    data = dval_e,
    objective = "reg:tweedie",
    eval_metric = "tweedie-nloglik@1.1",
    eta = 0.1,
    nfold = 5,
    max_depth = depth,
    nrounds = num_rounds,
    watchlist = watch_list_e,
    early_stopping_rounds = 20,
    verbose = 0 )
  
  # Train the model on the optimal number of trees from CV
  model = xgboost(data = dtrain_e, 
                  max.depth = depth,
                  objective = "reg:tweedie",
                  eval_metric = "tweedie-nloglik@1.1",
                  eta = 0.1,
                  nrounds = as.numeric(bst_cv_max[8]),
                  early_stopping_rounds = 20,
                  verbose = 0 )
  
  # Predict test set
  test_pred <- predict(model, dtest_e)
  
  # Calculate RMSE for test set
  rmse <- sqrt(mean((y_test_e - test_pred)^2))
  
  # Store the result
  results_e <- rbind(results_e, data.table(max_depth = depth, rmse = rmse))
}
print(results_e)


########################################################################
########## TASK 4 ##########

# First we need to make a copy of the original dataset, so we do not change it
x_train_100 = x_train
x_val_100 = x_val
x_test_100 = x_test

# Changing the first 10 elements of x1 in the training, val and test set to -100
x_train_100[1:10, 1] <- -100
x_val_100[1:10, 1] <- -100
x_test_100[1:10, 1] <- -100

# Setting the exponential transformation
x_train_e_ch <- exp(x_train_100)
x_val_e_ch <- exp(x_val_100)
x_test_e_ch <- exp(x_test_100)

# And for the y's
y_train_e_ch <- exp(y_train)
y_val_e_ch <- exp(y_val)
y_test_e_ch <- exp(y_test)

# Repeating the same task as above

## TASK 4.1 ##
#### Fitting XGBoost ####
# We first implement DMatrix in order to optimize XGBoost
dtrain_e_ch <- xgb.DMatrix(data = x_train_e_ch, label = y_train_e_ch)
dval_e_ch <- xgb.DMatrix(data = x_val_e_ch, label = y_val_e_ch)
dtest_e_ch <- xgb.DMatrix(data = x_test_e_ch, label = y_test_e_ch)

# Fitting the model on the training set, and using validation for num_trees
watch_list_e_ch <- list(test = dtest_e_ch, train = dtrain_e_ch)

# Use CV on the dval data set
bst_cv_e_ch = xgb.cv(data = dval_e_ch,
                     objective = "reg:tweedie",
                     eval_metric = "tweedie-nloglik@1.1",
                     eta = 0.1,
                     nfold = 5,
                     max_depth = 4,
                     nrounds = num_rounds,
                     watchlist = watch_list_e_ch,
                     early_stopping_rounds = 20,
                     verbose = 0 )

# Optimal number of trees
best_ntrees_e_ch <- as.numeric(bst_cv_e_ch[8])
print(paste("Optimal number of trees: ", best_ntrees_e_ch))

# Fitting the model
bst_model_exp_ch = xgboost(data = dtrain_e_ch, 
                           max.depth = 4,
                           objective = "reg:tweedie",
                           eval_metric = "tweedie-nloglik@1.1",
                           eta = 0.1,
                           nrounds = as.numeric(bst_cv_e_ch[8]),
                           early_stopping_rounds = 20,
                           verbose = 0 )

# Predicting on the test_set
y_test_pred_xgb_e_ch <- predict(bst_model_exp_ch, newdata = x_test_e_ch)

# Compute R^2
rss_xgb_e_ch <- mean((y_test_e_ch - y_test_pred_xgb_e_ch)^2)
#tss_xgb_e_ch <- sum((y_test_e - mean(y_test_e))^2)
tss_xgb_e_ch <- var(y_test_e_ch)
r2_xgb_e_ch <- round(1 - (rss_xgb_e_ch / tss_xgb_e_ch), 6)
print(paste("R2 for XGBoost:", r2_xgb_e_ch))


#### Fitting OLS ####
# Naming columns for x_train and x_test
colnames(x_train_e_ch) <- paste0("x", 1:ncol(x_train_e))
colnames(x_test_e_ch) <- colnames(x_train_e_ch)

# Converting matrices to data frames for the lm() function
df_train_e_ch <- data.frame(y_train_e = y_train_e_ch, x_train_e_ch)
df_test_e_ch <- data.frame(x_test_e_ch) 

# Fitting the OLS model
ols_model_exp_ch <- lm(y_train_e ~ 0 + x1 + x2 + x3 + x4, data  = df_train_e_ch)
sum_ols_exp_ch = summary(ols_model_exp_ch)
print(sum_ols_exp_ch)

# Predicting with OLS
y_pred_ols_e_ch <- predict(ols_model_exp_ch, newdata = df_test_e_ch)

# Compute R2 for OLS
rss_ols_e_ch <- mean((y_test_e_ch - y_pred_ols_e_ch)^2)
#tss_ols_e_ch <- sum((y_test_e - mean(y_test_e))^2)
tss_ols_e_ch <- var(y_test_e_ch)
r2_ols_e_ch <- round(1 - (rss_ols_e_ch / tss_ols_e_ch), 6)
print(paste("R2 for OLS:", r2_ols_e_ch))

# Compare R2 values
print(paste("Comparison of R2 values - XGBoost: ", r2_xgb_e_ch, ", OLS: ", r2_ols_e_ch))


########## TASK 4.2 ##########
# Gathering results in a table
results_e_ch <- data.table(max_depth = integer(), rmse = numeric())

# For loop over max_depth values
for(depth in 1:4){
  bst_cv_max = xgb.cv(
    data = dval_e_ch,
    objective = "reg:tweedie",
    eval_metric = "tweedie-nloglik@1.1",
    eta = 0.1,
    nfold = 5,
    max_depth = depth,
    nrounds = num_rounds,
    watchlist = watch_list_e_ch,
    early_stopping_rounds = 20,
    verbose = 0 )
  
  # Train the model on the optimal number of trees from CV
  model = xgboost(data = dtrain_e_ch, 
                  max.depth = depth,
                  objective = "reg:tweedie",
                  eval_metric = "tweedie-nloglik@1.1",
                  eta = 0.1,
                  nrounds = as.numeric(bst_cv_max[8]),
                  early_stopping_rounds = 20,
                  verbose = 0 )
  
  # Predict test set
  test_pred <- predict(model, dtest_e_ch)
  
  # Calculate RMSE for test set
  rmse <- sqrt(mean((y_test_e_ch - test_pred)^2))
  
  # Store the result
  results_e_ch <- rbind(results_e_ch, data.table(max_depth = depth, rmse = rmse))
}
print(results_e_ch)


########################################################################
########## TASK 5 ##########

# Introducing the fifth feature x5 = x2 - x3 + u, with u ~ N(0, 0.001^2)
u_train <- rnorm(n_train, mean = 0, sd = 0.001)
u_val <- rnorm(n_val, mean = 0, sd = 0.001)
u_test <- rnorm(n_test, mean = 0, sd = 0.001)

x5_train <- x_train[,2] - x_train[,3] + u_train
x5_val <- x_val[,2] - x_val[,3] + u_val
x5_test <- x_test[,2] - x_test[,3] + u_test

# Append x5 to the existing / new datasets
x_train_5 <- cbind(x_train, x5_train)
x_val_5 <- cbind(x_val, x5_val)
x_test_5 <- cbind(x_test, x5_test)
head(x_test_5)

# Setting the exponential transformation
x_train_e_5 <- exp(x_train_5)
x_val_e_5 <- exp(x_val_5)
x_test_e_5 <- exp(x_test_5)

# Need to change the feature names in order to avoid confusion
colnames(x_train_e_5)[5] <- "x5"
colnames(x_test_e_5)[5]<- "x5"

# And for the y's
y_train_e_5 <- exp(y_train)
y_val_e_5 <- exp(y_val)
y_test_e_5 <- exp(y_test)

# Repeating the same task as above

## TASK 5.1 ##

#### Fitting XGBoost ####
# We first implement DMatrix in order to optimize XGBoost
dtrain_e_5 <- xgb.DMatrix(data = x_train_e_5, label = y_train_e_5)
dval_e_5 <- xgb.DMatrix(data = x_val_e_5, label = y_val_e_5)
dtest_e_5 <- xgb.DMatrix(data = x_test_e_5, label = y_test_e_5)


# Fitting the model on the training set, and using validation for num_trees
watch_list_e_5 <- list(test = dtest_e_5, train = dtrain_e_5)

# Use CV on the dval data set
bst_cv_e_5 = xgb.cv(data = dval_e_5,
                    objective = "reg:tweedie",
                    eval_metric = "tweedie-nloglik@1.1",
                    eta = 0.1,
                    nfold = 5,
                    max_depth = 4,
                    nrounds = num_rounds,
                    watchlist = watch_list_e_5,
                    early_stopping_rounds = 20,
                    verbose = 0 )

# Optimal number of trees
best_ntrees_e_5 <- as.numeric(bst_cv_e_5[8])
print(paste("Optimal number of trees: ", best_ntrees_e_5))

bst_model_exp_5 = xgboost(data = dtrain_e_5, 
                          max.depth = 4,
                          objective = "reg:tweedie",
                          eval_metric = "tweedie-nloglik@1.1",
                          eta = 0.1,
                          nrounds = as.numeric(bst_cv_e_5[8]),
                          early_stopping_rounds = 20,
                          verbose = 0 )

# Predicting on the test_set
y_test_pred_xgb_e_5 <- predict(bst_model_exp_5, newdata = x_test_e_5)

# Compute R^2
rss_xgb_e_5 <- mean((y_test_e - y_test_pred_xgb_e_5)^2)
#tss_xgb_e_5 <- sum((y_test_e - mean(y_test_e))^2)
tss_xgb_e_5 <- var(y_test_e_5)
r2_xgb_e_5 <- round(1 - (rss_xgb_e_5 / tss_xgb_e_5), 6)
print(paste("R2 for XGBoost:", r2_xgb_e_5))


#### Fitting OLS ####
# Naming columns for x_train and x_test
colnames(x_train_e_5) <- paste0("x", 1:ncol(x_train_e_5))
colnames(x_test_e_5) <- colnames(x_train_e_5)

# Converting matrices to data frames for the lm() function
df_train_e_5 <- data.frame(y_train_e = y_train_e, x_train_e_5)
df_test_e_5 <- data.frame(x_test_e_5) 

# Fitting the OLS model
ols_model_exp_5 <- lm(y_train_e ~ 0 + x1 + x2 + x3 + x4 + x5, data  = df_train_e_5)
sum_ols_exp_5 = summary(ols_model_exp_5)
print(sum_ols_exp_5)

# Predicting with OLS
y_pred_ols_e_5 <- predict(ols_model_exp_5, newdata = df_test_e_5)

# Compute R2 for OLS
rss_ols_e_5 <- mean((y_test_e - y_pred_ols_e_5)^2)
#tss_ols_e_5 <- sum((y_test_e - mean(y_test_e))^2)
tss_ols_e_5 <- var(y_test_e_5)
r2_ols_e_5 <- round(1 - (rss_ols_e_5 / tss_ols_e_5), 6)
print(paste("R2 for OLS:", r2_ols_e_5))

# Compare R2 values
print(paste("Comparison of R2 values - XGBoost: ", r2_xgb_e_5, ", OLS: ", r2_ols_e_5))


########## TASK 5.2 ##########

# Gathering results in a table
results_e_5 <- data.table(max_depth = integer(), rmse = numeric())

# For loop over max_depth values
for(depth in 1:4){
  bst_cv_max = xgb.cv(
    data = dval_e_5,
    objective = "reg:tweedie",
    eval_metric = "tweedie-nloglik@1.1",
    eta = 0.1,
    nfold = 5,
    max_depth = depth,
    nrounds = num_rounds,
    watchlist = watch_list_e,
    early_stopping_rounds = 20,
    verbose = 0 )
  
  # Train the model on the optimal number of trees from CV
  model = xgboost(data = dtrain_e_5, 
                  max.depth = depth,
                  objective = "reg:tweedie",
                  eval_metric = "tweedie-nloglik@1.1",
                  eta = 0.1,
                  nrounds = as.numeric(bst_cv_max[8]),
                  early_stopping_rounds = 20,
                  verbose = 0 )
  
  # Predict test set
  test_pred <- predict(model, dtest_e_5)
  
  # Calculate RMSE for test set
  rmse <- sqrt(mean((y_test_e_5 - test_pred)^2))
  
  # Store the result
  results_e_5 <- rbind(results_e_5, data.table(max_depth = depth, rmse = rmse))
}
print(results_e_5)


########################################################################
########## TASK 6 ##########

#### 6.1 Fitting Ridge Regression ####

# The glmnet uses matrices instead of data frame compared to OLS
x_train_e_5_ridg <- as.matrix(x_train_e_5)
x_test_e_5_ridg <- as.matrix(x_test_e_5)
y_train_e_ridg <- as.vector(y_train_e_5)
y_test_e_ridg <- as.vector(y_test_e_5)

## Using CV to find the optimal lambda
cv_ridge <- cv.glmnet(x_train_e_5_ridg, y_train_e_ridg, alpha = 0, 
                      standardize = TRUE,
                      nfolds = 5)

# We plot the CV process for Ridge Regression
plot(cv_ridge)

# Extracting the optimal lambda
best_lambda_ridge <- round(cv_ridge$lambda.min, 5)
print(paste("Best Lambda for Ridge: ", best_lambda_ridge))

# Fitting our Ridge Model with optimal lambda
ridge_model_exp_5 <- glmnet(x_train_e_5_ridg, y_train_e_ridg, alpha = 0, 
                            lambda = as.numeric(best_lambda_ridge),
                            standardize = TRUE, intercept = FALSE)

# Printing out the coefficients from Ridge Regression
print(coef(ridge_model_exp_5))

# Predicting with Ridge Regression
y_pred_ridge_e_5 <- predict(ridge_model_exp_5, s = as.numeric(best_lambda_ridge), 
                            newx = x_test_e_5_ridg)

# Compute R2 for ridge regression
rss_ridge_e_5 <- mean((y_test_e_ridg - y_pred_ridge_e_5)^2)
#tss_ridge_e_5 <- sum((y_test_e_ridg - mean(y_test_e_ridg))^2)
tss_ridge_e_5 <- var(y_test_e_ridg)
r2_ridge_e_5 <- round(1 - (rss_ridge_e_5 / tss_ridge_e_5), 6)
print(paste("R2 for Ridge:", r2_ridge_e_5))


#### 6.2 Fitting Lasso Regression ####

# The glmnet uses matrices instead of data frame compared to OLS
x_train_e_5_lass <- as.matrix(x_train_e_5)
x_test_e_5_lass <- as.matrix(x_test_e_5)
y_train_e_lass <- as.vector(y_train_e_5)
y_test_e_lass <- as.vector(y_test_e_5)

## Using CV to find the optimal lambda
cv_lass <- cv.glmnet(x_train_e_5_lass, y_train_e_lass, alpha = 1, 
                     standardize = TRUE,
                     nfolds = 5)

# We plot the CV process for Lasso Regression
plot(cv_lass)

# Extracting the optimal lambda
best_lambda_lass <- round(cv_lass$lambda.min, 5)
print(paste("Best Lambda for lasso: ", best_lambda_lass))

# Fitting our Lasso Model with optimal lambda
lass_model_exp_5 <- glmnet(x_train_e_5_lass, y_train_e_lass, alpha = 1, 
                           lambda = as.numeric(best_lambda_lass),
                           standardize = TRUE, intercept = FALSE)

# Printing out the coefficients from Lasso Regression
print(coef(lass_model_exp_5))

# Predicting with Lasso Regression
y_pred_lass_e_5 <- predict(lass_model_exp_5, s = as.numeric(best_lambda_lass), 
                           newx = x_test_e_5_lass)

# Compute R2 for Lasso regression
rss_lass_e_5 <- mean((y_test_e_lass - y_pred_lass_e_5)^2)
#tss_lass_e_5 <- sum((y_test_e_lass - mean(y_test_e_lass))^2)
tss_lass_e_5 <- var(y_test_e_ridg)
r2_lass_e_5 <- round(1 - (rss_lass_e_5 / tss_lass_e_5), 6)
print(paste("R2 for Lassos:", r2_lass_e_5))




################################################################################
#### Extra steps for summarizing ####

results_table_ols <- data.table(
  Model_OLS = c("OLS", "OLS_e", "OLS_e_ch", "OLS_e_5",
            "Ridge_e_5", "Lasso_e_5"),
  R2_Value = c(r2_ols, r2_ols_e, r2_ols_e_ch, r2_ols_e_5,
               r2_ridge_e_5, r2_lass_e_5)
)
print(results_table_ols)

## For XGB ##
results_table_xgb <- data.table(
  Model_XGB = c("XGB", "XGB_e", "XGB_e_ch", "XGB_e_5"),
  R2_Value = c(r2_xgb, r2_xgb_e, r2_xgb_e_ch, r2_xgb_e_5)
)
print(results_table_xgb)





################################################################################

#### Histogram (train) of x-values from original data set ####
par(mfrow=c(2,2)) # Adjust grid layout based on the number of predictors
for(i in 1:ncol(x_train)) {
  hist(x_train[, i], main=paste("Histogram (train) of x",i), xlab="Values", breaks=30)
}


hist(y_train, main="Histogram of y_train", xlab="Values", breaks=30, col="blue")


#### Histogram (train) of x-values from exp ####
par(mfrow=c(2,2)) # Adjust grid layout based on the number of predictors
for(i in 1:ncol(x_train_e)) {
  hist(x_train_e[, i], main=paste("Histogram (train) of x_e",i), xlab="Values", breaks=30)
}

#### Histogram (train) of x-values from exp and changed the 10 first ####
par(mfrow=c(2,2)) # Adjust grid layout based on the number of predictors
for(i in 1:ncol(x_train_e_ch)) {
  hist(x_train_e_ch[, i], main=paste("Histogram (train) of x_e_ch",i), xlab="Values", breaks=30)
}

#### Histogram (train) of x-values from adding 5th and exp first ####
par(mfrow=c(2,2)) # Adjust grid layout based on the number of predictors
for(i in 1:ncol(x_train_e_5)) {
  hist(x_train_e_5[, i], main=paste("Histogram (train) of x_e_5",i), xlab="Values", breaks=30)
}



