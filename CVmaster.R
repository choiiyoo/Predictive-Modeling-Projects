library(caret)
library(glmnet)
library(MASS)
library(class)
library(e1071)
library(gbm)
library(randomForest)

CVmaster <- function(model, training_feature, training_label, K, lf, k_choices = NA,
                     cost_choices = NA, mtry_choices = NA, n.trees.choices = NA){
  
  # validation index;
  folds <- createFolds(unique(training_feature$group), k = K)
  print(folds)
  # cv based on group number; 
  # data_set <- cbind(training_feature, training_label)
  if (model == "KNN"){
    m <- matrix(0, nrow = length(k_choices), ncol = K + 1)
    for (i in seq_along(k_choices)){
      for (k in 1:K){
        group_index <- folds[[k]]
        validation_index <- training_feature$group %in% unique(training_feature$group)[group_index]
        validation_feature <- training_feature[validation_index, -ncol(training_feature)]
        validation_label <- training_label[validation_index]
        
        train_feature <- training_feature[-validation_index, -ncol(training_feature)]
        train_label <- training_label[-validation_index]
        knn_pred <- knn(train = train_feature, 
                        test = validation_feature, 
                        cl = train_label, 
                        k = k_choices[i])
        if (lf == "Classification Accuracy"){
          m[i, k] <- mean(knn_pred == validation_label)
        } else if (lf == "Precision"){
          tab <- table(knn_pred, validation_label)
          m[i, k] <- tab[2,2]/(tab[2,1]+tab[2,2])
        } else if (lf == "Recall"){
          tab <- table(knn_pred, validation_label)
          m[i, k] <- tab[2,2]/(tab[1,2]+tab[2,2])
        } else if (lf == "F1_scoer"){
          tab <- table(knn_pred, validation_label)
          pre <- tab[2,2]/(tab[2,1]+tab[2,2])
          rec <- tab[2,2]/(tab[1,2]+tab[2,2])
          m[i, k] <- 2 * pre * rec / (pre + rec)
        }
      }
      m[i, K+1] <- mean(m[i, 1:K])
    }
    colnames(m) <- c(seq(1, K), "Average")
    return (m)
  } 
    else if (model == "SVM"){
    res <- matrix(0, nrow = length(cost_choices), ncol = K + 1)
    for (i in seq_along(cost_choices)){
      for (k in 1:K){
        print(k)
        group_index <- folds[[k]]
        validation_index <- training_feature$group %in% unique(training_feature$group)[group_index]
        validation_feature <- training_feature[validation_index, -ncol(training_feature)]
        validation_label <- as.factor(training_label[validation_index])
        
        train_feature <- training_feature[-validation_index, -ncol(training_feature)]
        train_label <- as.factor(training_label[-validation_index])
        
        svmfit = svm(x = train_feature, y = train_label, scale = TRUE, 
                     kernel = "radial", cost = cost_choices[i])
        svm_pred <- predict(svmfit, validation_feature, probability = FALSE)
        
        if (lf == "Classification Accuracy"){
          res[i, k] <- mean(svm_pred == validation_label)
        } else if (lf == "Precision"){
          tab <- table(svm_pred, validation_label)
          res[i, k] <- tab[2,2]/(tab[2,1]+tab[2,2])
        } else if (lf == "Recall"){
          tab <- table(svm_pred, validation_label)
          res[i, k] <- tab[2,2]/(tab[1,2]+tab[2,2])
        } else if (lf == "F1_scoer"){
          tab <- table(svm_pred, validation_label)
          pre <- tab[2,2]/(tab[2,1]+tab[2,2])
          rec <- tab[2,2]/(tab[1,2]+tab[2,2])
          res[i, k] <- 2 * pre * rec / (pre + rec)
        }
      }
      res[i, K+1] <- mean(res[i, 1:K])
    }
    colnames(res) <- c(seq(1, K), "Average")
    return (res)
    } else if (model == "RF"){
      res <- matrix(0, nrow = length(mtry_choices), ncol = K + 1)
      for (i in seq_along(mtry_choices)){
        for (k in 1:K){
          group_index <- folds[[k]]
          validation_index <- training_feature$group %in% unique(training_feature$group)[group_index]
          validation_feature <- training_feature[validation_index, -ncol(training_feature)]
          validation_label <- as.factor(training_label[validation_index])
          
          train_feature <- training_feature[-validation_index, -ncol(training_feature)]
          train_label <- as.factor(training_label[-validation_index])
          
          #rf_fit <- randomForest(train_label~., data = cbind(train_feature, train_label), 
                                 #mtry = mtry_choices[i], importance = TRUE)
          rf_fit <- randomForest(x = train_feature, 
                                 y = train_label, 
                                 mtry = mtry_choices[i], 
                                 importance = TRUE)
          rf_pred <- predict(rf_fit, newdata = cbind(validation_feature, validation_label))
          if (lf == "Classification Accuracy"){
            res[i, k] <- mean(rf_pred == validation_label)
          } else if (lf == "Precision"){
            tab <- table(rf_pred, validation_label)
            res[i, k] <- tab[2,2]/(tab[2,1]+tab[2,2])
          } else if (lf == "Recall"){
            tab <- table(rf_pred, validation_label)
            res[i, k] <- tab[2,2]/(tab[1,2]+tab[2,2])
          } else if (lf == "F1_scoer"){
            tab <- table(rf_pred, validation_label)
            pre <- tab[2,2]/(tab[2,1]+tab[2,2])
            rec <- tab[2,2]/(tab[1,2]+tab[2,2])
            res[i, k] <- 2 * pre * rec / (pre + rec)
          }
          
        }
        res[i, K+1] <- mean(res[i, 1:K])
      }
      colnames(res) <- c(seq(1, K), "Average")
      return (res)
    } 
      else if (model == "Boosted Trees"){
        res <- matrix(0, nrow = length(n.trees.choices), ncol = K + 1)
        for (i in seq_along(n.trees.choices)){
          for (k in 1:K){
            group_index <- folds[[k]]
            validation_index <- training_feature$group %in% unique(training_feature$group)[group_index]
            validation_feature <- training_feature[validation_index, -ncol(training_feature)]
            validation_label <- as.integer(training_label[validation_index])
            
            train_feature <- training_feature[-validation_index, -ncol(training_feature)]
            train_label <- as.integer(training_label[-validation_index])
            
            bt_fit <- gbm(train_label~., data = cbind(train_feature, train_label),
                          interaction.depth = 2,
                          distribution = "bernoulli",
                          n.trees = n.trees.choices[i])
            bt_label <- predict(bt_fit, newdata = cbind(validation_feature, validation_label), "response", 
                                n.trees = n.trees.choices[i])
            bt_pred <- as.integer(bt_label > 0.5)
            if (lf == "Classification Accuracy"){
              res[i, k] <- mean(bt_pred == validation_label)
            } else if (lf == "Precision"){
              tab <- table(bt_pred, validation_label)
              res[i, k] <- tab[2,2]/(tab[2,1]+tab[2,2])
            } else if (lf == "Recall"){
              tab <- table(bt_pred, validation_label)
              res[i, k] <- tab[2,2]/(tab[1,2]+tab[2,2])
            } else if (lf == "F1_scoer"){
              tab <- table(bt_pred, validation_label)
              pre <- tab[2,2]/(tab[2,1]+tab[2,2])
              rec <- tab[2,2]/(tab[1,2]+tab[2,2])
              res[i, k] <- 2 * pre * rec / (pre + rec)
            }
            
          }
          res[i, K+1] <- mean(res[i, 1:K])
        }
        colnames(res) <- c(seq(1, K), "Average")
        return (res)
    }
  
    else if (model == "Logistic Regression"){
      res <- c()
      for (k in 1:K){
        group_index <- folds[[k]]
        validation_index <- training_feature$group %in% unique(training_feature$group)[group_index]
        validation_feature <- training_feature[validation_index, -ncol(training_feature)]
        validation_label <- training_label[validation_index]
        
        train_feature <- training_feature[-validation_index, -ncol(training_feature)]
        train_label <- training_label[-validation_index]
        
        fit_lr <- glm(train_label ~., train_feature, family = "binomial")
        predicted_label <- predict(fit_lr, validation_feature)
        predicted_label <- ifelse(predicted_label > 0.5, 1, 0)
        if (ls == "Classification Accuracy"){
          res[k] <- mean(predicted_label == validation_label)
        } else if (lf == "Precision"){
          tab <- table(predicted_label, validation_label)
          res[k] <- tab[2,2]/(tab[2,1]+tab[2,2])
        } else if (lf == "Recall"){
          tab <- table(predicted_laebl, validation_label)
          res[k] <- tab[2,2]/(tab[1,2]+tab[2,2])
        } else if (lf == "F1_scoer"){
          tab <- table(predicted_label, validation_label)
          pre <- tab[2,2]/(tab[2,1]+tab[2,2])
          rec <- tab[2,2]/(tab[1,2]+tab[2,2])
          res[k] <- 2 * pre * rec / (pre + rec)
        }
      }
      return (res)
    }
}