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


#download the data

credit_card <- read.csv("~/creditfraud/creditcard.csv")

#dislay dataset dimensions
dim(credit_card)

#preview data
head(credit_card)

#display summary of the dataset
summary(credit_card)

#convert transaction class to factor
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

#Check for missing values
sum(is.na(credit_card))

#Visualize distribution of legit and fraud cases
credit_card %>% group_by(Class) %>% summarize(n = n()) %>%
  ggplot(aes(Class, n, fill= Class)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values=c("#999999", "#E69F00"), breaks=c("0", "1"), labels= c("Legitimate", "Fraud")) +
  scale_y_continuous(labels = comma, trans='log10')


table(credit_card$Class)

#time feature
credit_card %>% ggplot(aes(Time)) + geom_histogram() + theme_classic()

# amount trend

credit_card %>% ggplot(aes(Amount)) + geom_histogram() + theme_classic() + 
  scale_y_continuous(labels = comma, trans='log10') 

#investigate correlation of predictors

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



#-------------------------------------------------------------------
# split into test and training set
set.seed(1)
test_index <- createDataPartition(y = credit_card$Class, times = 1, p = 0.1, list = FALSE)
cc_train <- credit_card[-test_index,]
cc_test <- credit_card[test_index,]

dim(cc_train)
table(cc_train$Class)

dim(cc_test)
table(cc_test$Class)

#-----------------------------------------------------------------
cc_train %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point()

#------------------------------------------------------------------------
#Balance training set using Random over sampling
n_legit <- sum(cc_train$Class == 0)
p <- 0.5
n_new = n_legit/p

osample_result <- ovun.sample(Class~., data= cc_train, 
                              method= "over",
                              N= n_new,
                              seed=20)
oversampled_credit <- osample_result$data
table(oversampled_credit$Class)

oversampled_credit %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point(position = position_jitter(width = 0.2)) 

#---------------------------------------------
#Random under sampling
n_fraud = sum(cc_train$Class == 1)
n_new_rus <- n_fraud/p
usample_result <- ovun.sample(Class~., data= cc_train, 
                             method= "under",
                             N= n_new_rus,
                             seed=20)
undersampled_credit <- usample_result$data
table(undersampled_credit$Class)

undersampled_credit %>% ggplot(aes(V4,V11, col = Class)) +
  geom_point() 

#-----------------------------------------------
#Synthetic Minority Over-sampling Technique (SMOTE)

n0 <- sum(cc_train$Class == 0)
n1 <- sum(cc_train$Class == 1)

ntimes <- ((1-p)/p)*(n0/n1)-1

smote_result <- SMOTE(X=cc_train[,-c(1,31)],
                      target= cc_train$Class,
                      K=5,
                      dup_size= ntimes)
credit_smote <- smote_result$data
table(credit_smote$Class)
colnames(credit_smote[30]) <- "Class"

credit_smote %>% ggplot(aes(V4,V11, col = class)) +
  geom_point()

#---------------------------------------------------
#build prediction models

model_CART <- rpart(class~., credit_smote)

