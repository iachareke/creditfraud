#install required packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshap2", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
if(!require(smotefamily)) install.packages("smotefamily", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(cowplot)) install.packages("cowplot", repos = "http://cran.us.r-project.org")
if(!require(ROCR)) install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(neuralnet)) install.packages("neuralnet", repos = "http://cran.us.r-project.org")


#load the data suing relative path of csv file
# the data can download at https://www.kaggle.com/mlg-ulb/creditcardfraud
credit_card <- read.csv("https://www.dropbox.com/s/ry8epx6sip80jee/creditcard.csv?dl=1")

#dislay dataset dimensions
dim(credit_card)

#preview data
head(credit_card)

#display summary of the dataset
summary(credit_card)

#convert transaction class to factor
#credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

#Check for missing values
sum(is.na(credit_card))

#Visualize distribution of legitimate and fraud cases
credit_card %>% group_by(Class) %>% summarize(n = n()) %>% mutate(Class = as.factor(Class)) %>%
  ggplot(aes(Class, n, fill= Class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("#999999", "#E69F00"), breaks=c("0", "1"), labels= c("Legitimate", "Fraud")) +
  scale_y_continuous(labels = comma, trans='log10')

#display distribution in tabular form
table(credit_card$Class)

#visualisze the distribution of transactions over the time feature
credit_card %>% ggplot(aes(Time)) + geom_histogram() + theme_classic()

# visualize the amount trend
credit_card %>% ggplot(aes(Amount)) + geom_histogram() + theme_classic() + 
  scale_y_continuous(labels = comma, trans='log10') 

#investigate correlation of predictors
#-------------------------------------------------
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
cor_cc <- round(cor(credit_card),2)

upper_tri <- get_upper_tri(cor_cc)

melted_corcc <- melt(upper_tri, na.rm = TRUE)

head(melted_corcc)

ggplot(data = melted_corcc, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

#Data preparation
#-------------------------------------------------------------------
# normalize time and amount
credit_card <- credit_card %>% mutate(Time = scale(Time), Amount = scale(Amount))

#convert Class field to factor
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

#-------------------------------------------------------------------
# split into test and training set
set.seed(1)
test_index <- createDataPartition(y = credit_card$Class, times = 1, p = 0.1, list = FALSE)
cc_train <- credit_card[-test_index,]
cc_test <- credit_card[test_index,]

# view dimensions of training set
dim(cc_train)
table(cc_train$Class)

# view dimensions of test set
dim(cc_test)
table(cc_test$Class)

#-----------------------------------------------------------------
#Creat visualization of training set using the positively correlated PCA
#features V4 and V11
a <- cc_train %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point(size = 0.1, stroke = 0, shape = 16, alpha = 0.5) +
  ggtitle('Original Dataset') + 
  theme(legend.position = "none")+
  expand_limits(x = c(-7.5, 17.5), y = c(-5, 15))

#------------------------------------------------------------------------
#Balance training set using Random over sampling
n_legit <- sum(cc_train$Class == 0)
p <- 0.5
n_new = n_legit/p
set.seed(1)
osample_result <- ovun.sample(Class~., data= cc_train, 
                              method= "over",
                              N= n_new,
                              seed=20)
oversampled_credit <- osample_result$data
table(oversampled_credit$Class)

#create visualization of dataset generated through random over sampling
b <- oversampled_credit %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point(position = position_jitter(width = 0.2), size = 0.1, stroke = 0, shape = 16, alpha = 0.5) +
  ggtitle('Random Over Sampling') + 
  theme(legend.position = "none")+
  expand_limits(x = c(-7.5, 17.5), y = c(-5, 15))


#---------------------------------------------
#blance dataset using Random under sampling
n_fraud = sum(cc_train$Class == 1)
n_new_rus <- n_fraud/p
set.seed(1)
usample_result <- ovun.sample(Class~., data= cc_train, 
                             method= "under",
                             N= n_new_rus,
                             seed=20)
undersampled_credit <- usample_result$data
table(undersampled_credit$Class)

# create visualization of data set created using random under sampling

c <- undersampled_credit %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point(size = 0.1, stroke = 0, shape = 16, alpha = 0.5) +
  ggtitle('Random Under Sampling') + 
  theme(legend.position = "none") +
  expand_limits(x = c(-7.5, 17.5), y = c(-5, 15))

#-----------------------------------------------
#Synthetic Minority Over-sampling Technique (SMOTE)

n0 <- sum(cc_train$Class == 0)
n1 <- sum(cc_train$Class == 1)

p=0.65

#calculate number of time to rum smote
ntimes <- ((1-p)/p)*(n0/n1)-1
ntimes
set.seed(1)
smote_result <- SMOTE(X=cc_train[,-c(1,31)],
                      target= cc_train$Class,
                      K=5,
                      dup_size= ntimes)
credit_smote <- smote_result$data

colnames(credit_smote)[30] <- "Class"
credit_smote$Class <- factor(credit_smote$Class, levels = c(0,1))
table(credit_smote$Class)

#create visualization of smote data set
d <- credit_smote %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point(size = 0.1, stroke = 0, shape = 16, alpha = 0.5) +
  ggtitle('SMOTE') + 
  theme(legend.position = "none")+
  expand_limits(x = c(-7.5, 17.5), y = c(-5, 15))

# combine the four visuals into one using plot_grid from cowplot
plot_grid(a,b,c,d)
#---------------------------------------------------
#build prediction models 
#Decision tree on smote data
set.seed(1)
model_CART <- rpart(Class~., credit_smote)
rpart.plot(model_CART)
  
#Predict fraud cases
predict_cart <- predict(model_CART, cc_test[,-1], type = 'class')
c_cart <- confusionMatrix(predict_cart, cc_test$Class)
as.table(c_cart)
#calculate and plot auc
x <- as.numeric(predict_cart)
y <- as.numeric(cc_test$Class)
pred_cart <- ROCR::prediction(x,y)
perf_cart <- performance(pred_cart,"tpr","fpr")
plot(perf_cart,colorize=TRUE)

auc_ROCR_cart <- performance(pred_cart, measure = "auc")
auc_ROCR_cart <- auc_ROCR_cart@y.values[[1]]
auc_ROCR_cart


#--------------------------------------------
#Decision tree on original training set data
set.seed(1)
model_CART_2 <- rpart(Class~., cc_train[,-1])
rpart.plot(model_CART_2)

#Predict fraud cases
predict_cart_2 <- predict(model_CART_2, cc_test[,-1], type = 'class')
c_cart_2 <- confusionMatrix(predict_cart_2, cc_test$Class)
as.table(c_cart_2)

#calculate and plot auc
x <- as.numeric(predict_cart_2)
y <- as.numeric(cc_test$Class)
pred_cart <- ROCR::prediction(x,y)
perf_cart <- performance(pred_cart,"tpr","fpr")
plot(perf_cart,colorize=TRUE)

auc_ROCR_cart_2 <- performance(pred_cart, measure = "auc")
auc_ROCR_cart_2 <- auc_ROCR_cart_2@y.values[[1]]
auc_ROCR_cart_2

#Create table to store results
roc_results <- data_frame(method = "Decision tree", SMOTE = auc_ROCR_cart, original = auc_ROCR_cart_2)
roc_results


#----------------------------------------------------------
#buIld random forest model on SMOTE training Data

#set.seed(1)
#rf_index <- createDataPartition(y = credit_smote$Class, times = 1, p = 0.2, list = FALSE)
#rf_train <- credit_smote[rf_index,]

set.seed(1)
model_RF <- randomForest(Class~., data = credit_smote, importance= TRUE )


#predict fraud cases
predict_RF <- predict(model_RF,cc_test[,-1], type = 'class')
cm_rf <- confusionMatrix(predict_RF, cc_test$Class)
as.table(cm_rf)


#calculate and plot auc
pred_rf <- ROCR::prediction(as.numeric(predict_RF), as.numeric(cc_test$Class))
perf_rf <- performance(pred_rf,"tpr","fpr")
plot(perf_rf,colorize=TRUE)

auc_ROCR_rf <- performance(pred_rf, measure = "auc")
auc_ROCR_rf <- auc_ROCR_rf@y.values[[1]]
auc_ROCR_rf

#----------------------------------------------------------
#buIld random forest model on unbalanced training Data

#set.seed(1)
#rf_index <- createDataPartition(y = cc_train$Class, times = 1, p = 0.2, list = FALSE)
#rf_train <- cc_train[rf_index,]
set.seed(1)
model_RF_2 <- randomForest(Class~., data = cc_train[,-1], importance= TRUE )


#predict fraud cases
predict_RF_2 <- predict(model_RF_2,cc_test[,-1], type = 'class')
cm_rf_2 <- confusionMatrix(predict_RF_2, cc_test$Class)
as.table(cm_rf_2)


#calculate and plot auc
pred_rf <- ROCR::prediction(as.numeric(predict_RF_2), as.numeric(cc_test$Class))
perf_rf <- performance(pred_rf,"tpr","fpr")
plot(perf_rf,colorize=TRUE)

auc_ROCR_rf_2 <- performance(pred_rf, measure = "auc")
auc_ROCR_rf_2 <- auc_ROCR_rf_2@y.values[[1]]
auc_ROCR_rf_2

# store results in table
roc_results <- bind_rows(roc_results, data_frame(method = "Random forest", SMOTE = auc_ROCR_rf, original = auc_ROCR_rf_2))
roc_results #%>% knitr::kable

#---------------------------------------------------------
#build Artificial neural network
#take subset of the training data for training ANN. 
set.seed(1)
nn_index <- createDataPartition(y = credit_smote$Class, times = 1, p = 0.1, list = FALSE)
nn_train <- credit_smote[test_index,]

#Train the ANN model
set.seed(1)
nn <- neuralnet(Class~ V1 + V2 + V3 + V4 + V5
                + V6 + V7 + V8 + V9 + V10 + V11
                + V12 + V13 + V14 + V15 + V16 + V17 
                + V18 + V19 + V20 + V21 + V22 + V23
                + V24 + V25 + V26 + V27 + V28 + Amount, data=nn_train, hidden=c(5,2), linear.output=FALSE)

#plot the model, calculate and plot ROC-AUC
plot(nn)
output <- compute(nn,cc_test[,-1])
prediction <- output$net.result
cm_nn <- confusionMatrix(as.factor(round(prediction[,2])),cc_test$Class)
as.table(cm_nn)

predict_nn <- as.factor(round(prediction[,2]))
pred_nn <- ROCR::prediction(as.numeric(predict_nn), as.numeric(cc_test$Class))
perf_nn <- performance(pred_nn,"tpr","fpr")
plot(perf_nn,colorize=TRUE)

auc_ROCR_nn <- performance(pred_nn, measure = "auc")
auc_ROCR_nn <- auc_ROCR_nn@y.values[[1]]
auc_ROCR_nn



#build Artificial neural network on unbalanced data set
#take a subset of the training data. 
set.seed(1)
nn_index <- createDataPartition(y = cc_train$Class, times = 1, p = 0.1, list = FALSE)
nn_train <- cc_train[nn_index,]

#train the ANN model on the unbalanced data
set.seed(1)
nn_2 <- neuralnet(Class~ V1 + V2 + V3 + V4 + V5
                + V6 + V7 + V8 + V9 + V10 + V11
                + V12 + V13 + V14 + V15 + V16 + V17 
                + V18 + V19 + V20 + V21 + V22 + V23
                + V24 + V25 + V26 + V27 + V28 + Amount, data=nn_train, hidden=c(5,2), linear.output=FALSE)

# generate confusion matrix, AUC and plot the ROC curve

plot(nn_2, main= "Artificial neural network")
output <- compute(nn_2,cc_test[,-1])
prediction <- output$net.result
cm_nn_2 <- confusionMatrix(as.factor(round(prediction[,2])),cc_test$Class)
as.table(cm_nn_2)

predict_nn <- as.factor(round(prediction[,2]))
pred_nn <- ROCR::prediction(as.numeric(predict_nn), as.numeric(cc_test$Class))
perf_nn <- performance(pred_nn,"tpr","fpr")
plot(perf_nn,colorize=TRUE)

auc_ROCR_nn_2 <- performance(pred_nn, measure = "auc")
auc_ROCR_nn_2 <- auc_ROCR_nn_2@y.values[[1]]
auc_ROCR_nn_2

#Final results
roc_results <- bind_rows(roc_results, data_frame(method = "Artificial Neural Network", SMOTE = auc_ROCR_nn, original = auc_ROCR_nn_2))
roc_results #%>% knitr::kable

#Convert all confusion matrices to matrix data types
c_cart <- as.matrix(c_cart)
c_cart_2 <- as.matrix(c_cart_2)
cm_rf <- as.matrix(cm_rf)
cm_rf_2 <- as.matrix(cm_rf_2)
cm_nn <- as.matrix(cm_nn)
cm_nn_2 <- as.matrix(cm_nn_2)

#arrange the confusion matrices into a table
cm_table <- data.frame(Method="decision tree on SMOTE", 
                       TP=c_cart[1,1], 
                       FP=c_cart[1,2],
                       FN=c_cart[2,1], 
                       TN=c_cart[2,2])
cm_table <- bind_rows(cm_table, data.frame(
  Method="decision tree on Original", 
  TP=c_cart_2[1,1],
  FP=c_cart_2[1,2],
  FN=c_cart_2[2,1],
  TN=c_cart_2[2,2]))
cm_table <- bind_rows(cm_table, data.frame(
  Method="Random Forest on SMOTE", 
  TP=cm_rf[1,1],
  FP=cm_rf[1,2],
  FN=cm_rf[2,1],
  TN=cm_rf[2,2]))
cm_table <- bind_rows(cm_table, data.frame(
  Method="Random Forest on Original", 
  TP=cm_rf_2[1,1],
  FP=cm_rf_2[1,2],
  FN=cm_rf_2[2,1],
  TN=cm_rf_2[2,2]))
cm_table <- bind_rows(cm_table, data.frame(
  Method="ANN on SMOTE", 
  TP=cm_nn[1,1],
  FP=cm_nn[1,2],
  FN=cm_nn[2,1],
  TN=cm_nn[2,2]))
cm_table <- bind_rows(cm_table, data.frame(
  Method="ANN on Original", 
  TP=cm_nn_2[1,1],
  FP=cm_nn_2[1,2],
  FN=cm_nn_2[2,1],
  TN=cm_nn_2[2,2]))
#Display usmmary table of confusion matrices.
cm_table 