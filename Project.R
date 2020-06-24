#------------------------------------------
## STAT 642-674
## Final Project
#------------------------------------------
#------------------------------------------

## Clear workspace
rm(list=ls())

## Set wd
setwd("D:/Drexel-2019-2020/3rd Quarter/STAT-642-674 Data Mining/Project")

## Load libraries
library(ggplot2)
library(e1071) #naive bayes
library(caret) #pre-processing, machine learning
library(class) #kNN
library(e1071) # SVM
library(rpart) # decision trees
library(rpart.plot) # decision tree plots
library(randomForest) # Random Forest
library(nnet) # neural networks
library(DMwR) #Resampling

##------------------Data Summary------------------------

## Load Data
churn <- read.csv(file='Churn_Modelling.csv')

## View the structure and summary information 
str(churn)
## 10000 obs. of 14 variables
## 'Geography', 'Gender', 'Surename': Factor 
## 'Exited' (Variable of interest): int (0 or 1)
## Other variables are int or num

## We don't need 'RowNumber', 'CustomerId' and 'Surname', thus removing them from 
## the analysis.
df1<-churn[,!(names(churn) %in% c('RowNumber', 'CustomerId', 'Surname'))]

# Descriptive statistics of the modified dataframe
summary(df1)

# Check for missing values
nrow(df1[!complete.cases(df1),])
## The output is '0'. Thus, we don't have any concern regarding missing values.

## As 'Exited' is of int type, I create another dataframe which captures this column
## as factor with 'Stayed' and 'Left' levels, associated with '1' and '0', respectively.
df<-df1
df$Exited[df$Exited==0]<-'Stayed'
df$Exited[df$Exited==1]<-'Left'
df$Exited<-as.factor(df$Exited)

summary(df)
## Left  :2037  $ Stayed:7963
## Churned customers are in minority, around a quarter of stayed ones.
## Thus, this data is imbalanced in terms of outcome variable and we should address it in 
## the following steps.

##------------------Data Exploration & Visualization------------------------

##----------------- Numeric variables

## CreditScore
hist(df$CreditScore, 
     main="Credit Score Histogram", 
     xlab="", 
     col="steelblue")
## Given majority of customers with score between 600 and 700, the distribution
## is fairly normal.

## Age
hist(df$Age, 
     main="Age Histogram", 
     xlab="", 
     col="steelblue")
## Given majority of customers at around 40, the distribution
## is fairly normal with slight skewness to right.

## Balance
hist(df$Balance, 
     main="Balance Histogram", 
     xlab="", 
     col="steelblue")
## Given majority of clients with '0' balance, the distribution cannot be normal.

## Estimated Salary
hist(df$EstimatedSalary, 
     main="Estimated Salary Histogram", 
     xlab="", 
     col="steelblue")
## The distribution is pretty uniform and obviously not normal.
##--------------------------------

## In order to detect any potential outlier, we plot boxplots across the levels 
## of outcome variable (Exited).

## CreditScore
boxplot(df$CreditScore~df$Exited, 
        main="Credit Score Box Plot",
        xlab='',
        ylab='',
        col=cm.colors(2))
## Only a few outliers are shown in the class of churned customers, as shown below:

## In the 'Left' class
boxplot.stats(df$CreditScore[df$Exited=='Left'])$out
## 363 359 350 350 358 351 365 367 350 350 350

## In the entire data
boxplot.stats(df$CreditScore)$out
## 376 376 363 359 350 350 358 351 365 367 350 350 382 373 350

## Age
boxplot(df$Age~df$Exited, 
        main="Age Box Plot",
        xlab='',
        ylab='',
        col=cm.colors(2))
## Some outliers are detected at both classes, as shown below:

## Only few in the 'Left' class
boxplot.stats(df$Age[df$Exited=='Left'])$out
## 73 18 71 74 71 84 71 71 71 71 72 18 71

## 486 outliers in the 'Stayed' class
boxplot.stats(df$Age[df$Exited=='Stayed'])$out
## Min = 57 Mean = 65.63    Max =92 

## 359 outliers in the entire data
boxplot.stats(df$Age)$out
## Min = 63 Mean = 69.27   Max =92

## Balance
boxplot(df$Balance~df$Exited, 
        main="Balance Box Plot",
        xlab='',
        ylab='',
        col=cm.colors(2))
## No outlier is detected.

## Estimated Salary
boxplot(df$EstimatedSalary~df$Exited, 
        main="Estimated Salary Box Plot",
        xlab='',
        ylab='',
        col=cm.colors(2))
## No outlier is detected.

## In general, there is not a serious concern with regard to outliers as
## the ratio of outliers-where they were detected- to the size of data is 
## reasonably low. However, one may decide to anlayze data in the absence of 
## 486 outliers associated with Age, if addressing outliers is necessary.

##--------------------------------
## As Tenure and Number of products include only few integer values, 
## we use bar plot instead of histogram

## Tenure
barplot(table(df$Tenure),
        main="Tenure Bar Plot", col='cornflowerblue',
        xlab='Tenure(Year)')
## The distribution is symmetric with the lowest values at each end, associating with
## deposits tenured in less than a year or 10 years.

## Number of Products
barplot(table(df$NumOfProducts),
        main="Number of Products Bar Plot", col='cornflowerblue',
        xlab='Number of Products')
## The distribution isn't symetric with the largest number of customers subscribed 
## to one or two products at most and less than 500 customers associated with more products.   

##----------------- Categorical variables

## In order to explore the dependency between churning behavior and classes of a categorical variable,
## We perform Chi-Square Test between categorical variables and Exited (as dependent variable (DV)).

## Gender
barplot(table(df$Exited, df$Gender), beside = TRUE,
        legend.text = levels(df$Exited),
        main="Churning Behaviour across Males and Females",
        col=heat.colors(2))

## In general, it is seen that most customers stayed with banks.
## Data is pretty balanced across males and females
## But the ratio of churning seems more for females than males.

chisq.test(table(df$Gender, df$Exited))
## X-squared = 112.92, df = 1, p-value < 2.2e-16
## Thus, females represent significantly lower churning rate than males.

## Countries
barplot(table(df$Exited, df$Geography), beside = TRUE,
        legend.text = levels(df$Exited),
        main="Churning Behaviour across Countries",
        col=heat.colors(2))

## Across all three countries most customers stayed with banks.
## Data is pretty balanced between banks from Germany and Spain but records from 
## the France-based banks are in majority, around twice the records from banks  
## located in Germany or Spain while their associated churning records are similar to 
## those of Germany and Spain. 

chisq.test(table(df$Geography, df$Exited))
## X-squared = 301.26, df = 2, p-value < 2.2e-16
## Thus, customers of French banks associate with significantly lower churning rate 
## rather than customers of banks located in Germany or Spain.

## HasCrCard and IsActiveMember
## First, convert 0,1 values to their associated defenitions for the sake of better visualization.
df2<-df
df2$HasCrCard[df2$HasCrCard==0]<-'Without Credit Card'
df2$HasCrCard[df2$HasCrCard==1]<-'With Credit Card'
df2$HasCrCard<-as.factor(df2$HasCrCard)

df2$IsActiveMember[df2$IsActiveMember==0]<-'Not Active Member'
df2$IsActiveMember[df2$IsActiveMember==1]<-'Active Member'
df2$IsActiveMember<-as.factor(df2$IsActiveMember)

## Having a Credit Card
barplot(table(df2$Exited, df2$HasCrCard), beside = TRUE,
        legend.text = levels(df2$Exited),
        main="Churning Behaviour vs. Having a Credit Card",
        col=heat.colors(2))
## Likewise, at either group most customers stayed with bank 
## and the ratio of churning seems similar. 

chisq.test(table(df$HasCrCard, df$Exited))
## X-squared = 0.47134, df = 1, p-value = 0.4924
## Thus, having just a credit card through the bank doesn't necessarily associate with a higher retention rate.

## Being an Active Member
barplot(table(df2$Exited, df2$IsActiveMember), beside = TRUE,
        legend.text = levels(df2$Exited),
        main="Churning Behaviour vs. Being an Active Member",
        col=heat.colors(2))
## Likewise, at either group most customers stayed with bank 
## But the ratio of churning is more for non-active customers.

chisq.test(table(df$IsActiveMember, df$Exited))
## X-squared = 242.99, df = 1, p-value < 2.2e-16
## Thus, active customers associate with higher retention rate rather than non-active ones. 

## Although we dealt with NumOfProducts as a numeric variable,
## We can still study the distribution of outcome variable across its only 4 values.
barplot(table(df2$Exited, df2$NumOfProducts), beside = TRUE,
        legend.text = levels(df2$Exited),
        main="Churning Behaviour across Number of Products",
        xlab='Number of Products',
        col=heat.colors(2))
## As seen, most customers who purchased one or two products stayed with the bank.
## On the other hand, most of those who purchased three or four products left the bank.

chisq.test(table(as.factor(df$NumOfProducts), df$Exited))
## X-squared = 1503.6, df = 3, p-value < 2.2e-16
## Thus, we can only deduce that there is dependency between chuning behavior and the number
## of products a customer is affiliated with (Notice that the churn ratio doesn't represent
## a constantly decreasing or increasing trend with respect to the number of products.)

## ----------------- Correlation between Numeric variables

## Selecting numeric variables
df3<-df1[,c('Exited', 'CreditScore', 'Age', 'Tenure', 'Balance', 
           'NumOfProducts', 'EstimatedSalary', "IsActiveMember", "HasCrCard")]

corr<-cor(df3)
View(corr)
summary(corr[upper.tri(corr)])
##      Min.      Mean      Max. 
##   -0.304180 -0.001867 0.285323
sort(corr[1,], decreasing = TRUE) # Pulling correlations of all IVs with DV

## As shown, none of independent variables (IVs) are strongly correlated with each other.
## Balance and Number of Products uncover the lowest and negative correlation (-0.304) 


## None of IVs are also strongly correlated with the DV, 
## given Age and IsActiveMember with the highest (0.256) and the lowest (-0.156) values, respectively. 
##------------------Analysis------------------------

## Our data lacks missing values, irrelevant and redundant variables based on what we've explored so far.
## Thus, the only pre-processing steps which may be required prior to the analysis are 
## creating dummies for categorical variables and rescaling variables or transforming them to 
## address the violation from normality assumption to some extent.

## ----------------- Naive Bayes

## Load libraries
## library(e1071) #naive bayes
## library(caret) #pre-processing, machine learning

## Since we have numerical inputs, the data is assumed to be Gaussian and thus, data needs to be transformed.

prepObj <- preProcess(df, method="BoxCox")
dfnb <- predict(prepObj, df)

## Lets check the outcome of transformation for Balance and Estimated Salary which had some issues 
## regarding the normality assumption.

## Balance
hist(dfnb$Balance, 
     main="Transformed Balance Histogram", 
     xlab="", 
     col="steelblue")

## Estimated Salary
hist(dfnb$EstimatedSalary, 
     main="Transformed Estimated Salary Histogram", 
     xlab="", 
     col="steelblue")

## As shown the effect of transformation is visible for the estimated salary by making it 
## more inclined to normal rather than uniform. However, Balance still suffers from the presence 
## of the largest number of customers with zero balance.

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(dfnb$Exited, p=0.80, list=FALSE)
train_nb = dfnb[sub, ] 
test_nb = dfnb[-sub, ]

nb_mod <- naiveBayes(formula=Exited~.,
                     data=train_nb)

## Training Performance
nb.train <- predict(object=nb_mod, 
                    newdata=train_nb[,-11], 
                    type="class")
train_nb_perf<-confusionMatrix(data=nb.train, 
                               reference=train_nb$Exited, 
                               positive="Left",
                               mode="prec_recall")
train_nb_perf

##              Reference
## Prediction   Left Stayed
## Left         420    270
## Stayed       1210   6101

## Accuracy : 0.815
## Kappa : 0.2741 
## Precision : 0.60870         
## Recall : 0.25767         
## F1 : 0.36207 

## Testing Performance
nb.test <- predict(object=nb_mod, 
                    newdata=test_nb[,-11], 
                    type="class")
test_nb_perf<-confusionMatrix(data=nb.test, 
                              reference=test_nb$Exited, 
                              positive="Left",
                              mode="prec_recall")
test_nb_perf

##              Reference
## Prediction   Left Stayed
## Left         114    58
## Stayed       293   1534

## Accuracy : 0.8244
## Kappa : 0.3104 
## Precision : 0.66279        
## Recall : 0.28010         
## F1 : 0.39378 

## Kappa is neither poor nor good but fair (0.2<kappa<0.4).
## F1 is also closer to 0 rather than 1.
## Thus, The overall performance is not good 
## (the model seems very slightly underfitted!)

## ----------------- k-Nearest Neighbors

## Load libraries
## library(class) #kNN

## Since we have categorical variables (Gender and Geography), their associated dummies are required to be created.

## By setting fullRank=TRUE, "Female" and "France" are excluded from Gender and Geography levels, respectively.
dum <- dummyVars(~Gender+Geography, data=df,
                 sep="_", fullRank = TRUE)

dfknn1 <- predict(dum, df)

## keeping all columns in dfknn1 except our original categorical variables
dfknn2 <- data.frame(df[,!names(df) %in% c("Gender", "Geography")], dfknn1)
str(dfknn2)

## kNN has been shown to perform well with min-max normalization. So we rescale the numerical variables.
prepObj <- preProcess(x=dfknn2, method="range")
dfknn <- predict(prepObj, dfknn2)

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(dfknn$Exited, p=0.80, list=FALSE)
train_knn = dfknn[sub, ] 
test_knn = dfknn[-sub, ]

## Hyperparameter Tuning
## We want to tune the number of nearest neighbors (k) 
## using a grid search with accuracy as the performance measure
## and a repeated 10-Fold cross validation for three times

## We set up our grid to be only odd numbers from 3 to 33,
## then applying a repeated 10-Fold cross validation for three times 

grids <- expand.grid(k=seq(from=3,to=33,by=2))

ctrl_grid <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 3,
                          search = "grid")

set.seed(831)
knnFit <- train(form = Exited ~ ., 
                data = train_knn, 
                method = "knn", 
                trControl = ctrl_grid, 
                tuneGrid = grids)

knnFit
## The final value used for the model was k = 9.
## Accuracy   Kappa
## 0.8161061  0.2663960
plot(knnFit)

## Training Performance
inpreds_knn <- predict(knnFit, newdata=train_knn)
train_knn_perf<-confusionMatrix(data=inpreds_knn, 
                                reference=train_knn$Exited, 
                                positive="Left",
                                mode="prec_recall")
train_knn_perf

##              Reference
## Prediction   Left Stayed
## Left         534    172
## Stayed       1096   6199

## Accuracy : 0.8415
## Kappa : 0.381 
## Precision : 0.75637        
## Recall : 0.32761         
## F1 : 0.45719

## Testing Performance
outpreds_knn <- predict(knnFit, newdata=test_knn)
test_knn_perf<-confusionMatrix(data=outpreds_knn, 
                               reference=test_knn$Exited, 
                               positive="Left",
                               mode="prec_recall")
test_knn_perf

##              Reference
## Prediction   Left Stayed
## Left         92    50
## Stayed       315   1542

## Accuracy : 0.8174
## Kappa : 0.2569 
## Precision : 0.64789        
## Recall : 0.22604         
## F1 : 0.33515

## Kappa is neither poor nor good but fair (0.2<kappa<0.4).
## F1 is also closer to 0 rather than 1.
## Thus, The overall performance is not good 

## ----------------- SVM
## Load libraries
## library(e1071) # SVM

## Since we have categorical variables (Gender and Geography), their associated dummies are required to be created.
## By setting fullRank=TRUE, "Female" and "France" are excluded from Gender and Geography levels, respectively.
dum <- dummyVars(~Gender+Geography, data=df,
                 sep="_", fullRank = TRUE)

dfsvm1 <- predict(dum, df)

## keeping all columns in dfksvm1 except our original categorical variables
dfsvm <- data.frame(df[,!names(df) %in% c("Gender", "Geography")], dfsvm1)
str(dfsvm)

## Rescaling is required but as z-score rescaling will be completed when creating the 
## SVM model,  we do not need to rescale the data prior to modeling.

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(dfsvm$Exited, p=0.80, list=FALSE)
train_svm = dfsvm[sub, ] 
test_svm = dfsvm[-sub, ]

## Radial Kernel
set.seed(831)
svm_modR <- svm(Exited~., 
                data=train_svm, 
                method="C-classification", 
                kernel="radial", 
                scale=TRUE)

# Training Performance
svm.trainR <- predict(svm_modR, 
                      train_svm[,-9], 
                      type="class")
train_svm_perf<-confusionMatrix(svm.trainR, 
                                train_svm$Exited, 
                                positive="Left", 
                                mode="prec_recall")
train_svm_perf

##              Reference
## Prediction   Left Stayed
## Left         679    125
## Stayed       951   6246

## Accuracy : 0.8655
## Kappa : 0.4892
## Precision : 0.84453      
## Recall : 0.41656        
## F1 : 0.55793

# Testing Performance
svm.testR <- predict(svm_modR, 
                     test_svm[,-9], 
                     type="class")
test_svm_perf<-confusionMatrix(svm.testR, 
                               test_svm$Exited, 
                               positive="Left", 
                               mode="prec_recall")
test_svm_perf
##              Reference
## Prediction   Left Stayed
## Left         168    29
## Stayed       239   1563

## Accuracy : 0.8659
## Kappa : 0.4883
## Precision : 0.85279      
## Recall : 0.41278        
## F1 : 0.55629

## Kappa is not still good but moderate (0.4<kappa<0.6).
## F1 is also closer to 1 rather than 0.
## Thus, The overall performance is moderate. 
## Even without hyperparameter tuning, SVM performs better than Naive Bayes and K-NN.

## ----------------- Decision Trees
## Load libraries
## library(rpart) # decision trees
## library(rpart.plot) # decision tree plots

## we can use the dataset as-is in our modeling, without any transformations and
## creating dummies of categorical variables.

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(df$Exited, p=0.80, list=FALSE)
train_dt = df[sub, ] 
test_dt = df[-sub, ]

## Hyperparameter Tuning
## We want to tune the cost complexity parameter (cp) 
## using a grid search with accuracy as the performance measure
## and a repeated 10-Fold cross validation for three times

grids <- expand.grid(cp=seq(from=0,to=.25,by=.01))

ctrl_grid <- trainControl(method="repeatedcv",
                          number = 10,
                          repeats = 3,
                          search="grid")

set.seed(831)
DTFit <- train(form=Exited ~ ., 
               data = train_dt, 
               method = "rpart",
               trControl = ctrl_grid, 
               tuneGrid=grids)

DTFit
## The final value used for the model was cp = 0.01
## Accuracy   Kappa    
## 0.8563938  0.4567984
plot(DTFit)

## Variable importance information 
varImp(DTFit)

## rpart variable importance
##
## NumOfProducts    100.000
## Age               85.858
## GeographyGermany  40.732
## IsActiveMember    33.616
## Balance           21.253
## GenderMale         6.007
## EstimatedSalary    1.906
## Tenure             1.545
## CreditScore        0.000
## HasCrCard          0.000
## GeographySpain     0.000

# Get the summary of best fitted tree
DTFit$finalModel
rpart.plot(DTFit$finalModel)

## Training Performance
inpreds_dt <- predict(DTFit, newdata=train_dt)
train_dt_perf<-confusionMatrix(data=inpreds_dt, 
                               reference=train_dt$Exited, 
                               positive="Left",
                               mode="prec_recall")
train_dt_perf

##              Reference
## Prediction   Left Stayed
## Left         656    162
## Stayed       974   6209

## Accuracy : 0.858
## Kappa : 0.4628 
## Precision : 0.80196        
## Recall : 0.40245         
## F1 : 0.53595

## Testing Performance
outpreds_dt <- predict(DTFit, newdata=test_dt)
test_dt_perf<-confusionMatrix(data=outpreds_dt, 
                              reference=test_dt$Exited, 
                              positive="Left",
                              mode="prec_recall")
test_dt_perf

##              Reference
## Prediction   Left Stayed
## Left         172    34
## Stayed       235   1558

## Accuracy : 0.8654
## Kappa : 0.4916 
## Precision : 0.83495       
## Recall : 0.42260         
## F1 : 0.56117

## Kappa is not still good but moderate (0.4<kappa<0.6).
## F1 is also closer to 1 rather than 0.
## Thus, The overall performance is moderate. 
## The model seems very slightly underfitted.
## Decision trees performs better than Naive Bayes and K-NN and pretty as well as svm.

## ----------------- Random Forest

## Load libraries
## library(randomForest) # Random Forest

## we can use the dataset as-is in our modeling, without any transformations and
## creating dummies of categorical variables.

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(df$Exited, p=0.80, list=FALSE)
train_rf = df[sub, ] 
test_rf = df[-sub, ]

## Hyperparameter Tuning
## We want to tune the number of variables to randomly sample as potential variables to split on (mtry).
## using a grid search with accuracy as the performance measure
## and a repeated 5-Fold cross validation for three times

grids = expand.grid(mtry = seq(from = 1, to = 10, by = 1))

grid_ctrl <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          search="grid")
set.seed(831)
fit.rf <- train(Exited~., 
                data=train_rf, 
                method="rf", 
                trControl=grid_ctrl,
                tuneGrid=grids)

## Converged in 15 minutes
fit.rf
## The final value used for the model was mtry = 4.
## Accuracy   Kappa
## 0.8637677  0.51488926
plot(fit.rf)

## Variable importance information 
varImp(fit.rf)

## rf variable importance
##
## Age              100.000
## Balance           58.268
## EstimatedSalary   57.314
## CreditScore       55.235
## NumOfProducts     49.417
## Tenure            28.345
## IsActiveMember    12.607
## GeographyGermany   6.363
## GenderMale         2.433
## HasCrCard          1.903
## GeographySpain     0.000

## Training Performance
rf.train <- predict(fit.rf, 
                     train_rf[,-11])

train_rf_perf<-confusionMatrix(rf.train, 
                               train_rf$Exited,
                               positive="Left",
                               mode="prec_recall")
train_rf_perf
##              Reference
## Prediction   Left Stayed
## Left         1630    0
## Stayed       0    6371

## Accuracy : 1
## Kappa : 1
## Precision : 1.0000     
## Recall : 1.0000        
## F1 : 1.0000

## Testing Performance
rf.test <- predict(fit.rf, 
                    test_rf[,-11])

test_rf_perf<-confusionMatrix(rf.test, 
                               test_rf$Exited,
                               positive="Left",
                               mode="prec_recall")
test_rf_perf
##              Reference
## Prediction   Left Stayed
## Left         193    57
## Stayed       214    1535

## Accuracy : 0.8644
## Kappa : 0.5119
## Precision : 0.77200     
## Recall : 0.47420        
## F1 : 0.58752

## Kappa is not still good but moderate (0.5<kappa<0.6).
## F1 is also closer to 1 rather than 0.
## Thus, The overall performance is moderate.
## Overfitting is substantial by the model.
## Random forest performs better than all previous models in terms of all measures.

## Compared to decision trees, the rank of variables in terms of their importance is 
## significantly different. Specifically, variables of tenure, credit score and estimated salary
## which appear to be less important by decision trees, turned out to be more impactful
## by random forest. On the other hand, being located in Germany is recognized
## less important by random forest while it is suggested as an important feature relying on
## decision trees. 

## ------------------- ANN
## Load libraries
## library(nnet) # neural networks

## Since we have categorical variables (Gender and Geography), their associated dummies are required to be created.
## By setting fullRank=TRUE, "Female" and "France" are excluded from Gender and Geography levels, respectively.
dum <- dummyVars(~Gender+Geography, data=df,
                 sep="_", fullRank = TRUE)

dfann1 <- predict(dum, df)

## keeping all columns in dfksvm1 except our original categorical variables
dfann2 <- data.frame(df[,!names(df) %in% c("Gender", "Geography")], dfann1)
str(dfann2)

## Rescaling is required as ANN does best when data ranges between 0-1. 
## Thus, we will use the min-max normalization to rescale our data.

dfann3 <- preProcess(dfann2, method="range")
dfann <- predict(dfann3, dfann2)
summary(dfann)

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(dfann$Exited, p=0.80, list=FALSE)
train_ann = dfann[sub, ] 
test_ann = dfann[-sub, ]

## Hyperparameter Tuning
## We want to tune the number of nodes in the only hidden layer of the model plus weight decay.
## using a grid search with accuracy as the performance measure
## and a repeated 5-Fold cross validation for three times

grids_ann = expand.grid(size = seq(from = 1, to = 5, by = 1),
                    decay = seq(from = 0.1, to = 0.5, by = 0.1))

ctrl <- trainControl(method="repeatedcv",
                     number = 5,
                     repeats=3,
                     search="grid")

set.seed(831)
annMod <- train(Exited~ ., data = train_ann, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids_ann,
                verbose=FALSE)

## Converged in 10 minutes
annMod
## The final values used for the model were size = 5 and decay = 0.1.
## Accuracy   Kappa 
## 0.8610183  0.5040465
plot(annMod)


## Training Performance
inpreds_ann <- predict(annMod, newdata=train_ann)
train_ann_perf<-confusionMatrix(inpreds_ann, 
                                train_ann$Exited, 
                                positive="Left",
                                mode="prec_recall")
train_ann_perf
##              Reference
## Prediction   Left Stayed
## Left         784    245
## Stayed       846    6126

## Accuracy : 0.8636
## Kappa : 0.5129
## Precision : 0.76190     
## Recall : 0.48098        
## F1 : 0.58970

## Testing Performance
outpreds_ann <- predict(annMod, newdata=test_ann)
test_ann_perf<-confusionMatrix(outpreds_ann, 
                                test_ann$Exited, 
                                positive="Left",
                                mode="prec_recall")
test_ann_perf
##              Reference
## Prediction   Left Stayed
## Left         784    245
## Stayed       846    6126

## Accuracy : 0.8669
## Kappa : 0.5141
## Precision : 0.79747     
## Recall : 0.46437       
## F1 : 0.58696

## Kappa is not still good but moderate (0.5<kappa<0.6).
## F1 is also closer to 1 rather than 0.
## Thus, The overall performance is moderate.
## ANN performs very close to random forest, with a very slight improvement.

## ------------------- Performance Comparison

## Training set
train_perf<-cbind(train_nb=train_nb_perf$byClass, 
                  train_knn=train_knn_perf$byClass,
                  train_svm=train_svm_perf$byClass,
                  train_dt=train_dt_perf$byClass,
                  train_rf=train_rf_perf$byClass,
                  train_ann=train_ann_perf$byClass)
train_perf

## Testing set
test_perf<-cbind(test_nb=test_nb_perf$byClass, 
                test_knn=test_knn_perf$byClass, 
                test_svm=test_svm_perf$byClass,
                test_dt=test_dt_perf$byClass,
                test_rf=test_rf_perf$byClass,
                test_ann=test_ann_perf$byClass)
test_perf

## Random forest and ANN are two competing models for this problem which we can study 
## in detail to improve their performance.

## -------------------------- Feature Selection

## Variable importance by Random Forest
set.seed(831)
churn.rf <- randomForest(Exited~.,
                       data=train_rf,
                       importance=TRUE, 
                       proximity=TRUE, 
                       ntree=500)
churn.rf$importance
varImpPlot(churn.rf)

## As seen, the order of important variables depends on whether accuracy or Gini 
## is selected for comparing between mean decrease by the exclusion of these variables.
## This observation is in accordance with the difference we noticed earlier
## by comapring the outputs from decision trees and random forest models.

## We can only say that having credit card is the least important variable 
## by either criterion and age along with balance is among the top four important 
## features. However, discarding HasCard variable and keeping only Age and Balance
## as they are the only two consistent variables by mean decrease criteria 
## isn't a meaningful decision for this problem.

## Thereby, we use recursive feature elimination (RFE) from wrapper methods
## to have a more sound and persuasive procedure for feature selection.

set.seed(831)
control_rfe <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      number = 10,
                      repeats = 3,
                      verbose = FALSE)
churn_rfe <- rfe(x = train_rf[,-11], 
               y = train_rf$Exited,
               rfeControl = control_rfe)

## Converged in 10 minutes
churn_rfe
## Variables  Accuracy  Kappa AccuracySD KappaSD Selected
## 4            0.8503 0.4441   0.007627 0.03465         
## 8            0.8624 0.4933   0.008628 0.03867         
## 10           0.8628 0.5081   0.008969 0.03638        *

## The top 5 variables (out of 10):
## Age, NumOfProducts, IsActiveMember, Balance, Geography

## Thus, RFE suggests the inclusion of all variables which will result
## in what we have obtained so far. Thus, in order to test the model under
## a different condition, we select only those top 5 features which are 
## the same top 5 variables detected by the mean decrease in accuracy.

train_fs <- train_rf[, colnames(train_rf) %in% c('Exited',churn_rfe$optVariables[1:5])]
test_fs <- test_rf[,colnames(test_rf) %in% c('Exited',churn_rfe$optVariables[1:5])]

## -------------------------- Random forest over selected features

grids = expand.grid(mtry = seq(from = 1, to = 10, by = 1))

grid_ctrl <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          search="grid")
set.seed(831)
fit.rf_fs <- train(Exited~., 
                data=train_fs, 
                method="rf", 
                trControl=grid_ctrl,
                tuneGrid=grids)

## Converged in 15 minutes
fit.rf_fs
## The final value used for the model was mtry = 3.
## Accuracy   Kappa
## 0.8621013  0.5116485
plot(fit.rf_fs)

## Training Performance
rf.train_fs <- predict(fit.rf_fs, 
                    train_fs[,-6])

train_fs_rf_perf<-confusionMatrix(rf.train_fs, 
                               train_fs$Exited,
                               positive="Left",
                               mode="prec_recall")
## Testing Performance
rf.test_fs <- predict(fit.rf_fs, 
                       test_fs[,-6])

test_fs_rf_perf<-confusionMatrix(rf.test_fs, 
                                  test_fs$Exited,
                                  positive="Left",
                                  mode="prec_recall")

## -------------------------- ANN over selected features

train_fsnn <- train_ann[, colnames(train_ann) %in% 
                          c('Exited',"Age","NumOfProducts","IsActiveMember","Balance",
                            "Geography_Germany","Geography_Spain")]

test_fsnn <- test_ann[,colnames(test_ann) %in% 
                          c('Exited',"Age","NumOfProducts","IsActiveMember","Balance",
                                "Geography_Germany","Geography_Spain")]

grids_ann = expand.grid(size = seq(from = 1, to = 5, by = 1),
                        decay = seq(from = 0.1, to = 0.5, by = 0.1))

ctrl <- trainControl(method="repeatedcv",
                     number = 5,
                     repeats=3,
                     search="grid")

set.seed(831)
annMod <- train(Exited~ ., data = train_fsnn, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids_ann,
                verbose=FALSE)

## Converged in 10 minutes
annMod
## The final values used for the model were size = 5 and decay = 0.1.
## Accuracy   Kappa 
## 0.8611845  0.5026098
plot(annMod)


## Training Performance
inpreds_fsnn <- predict(annMod, newdata=train_fsnn)
train_fs_nn_perf<-confusionMatrix(inpreds_fsnn, 
                                train_fsnn$Exited, 
                                positive="Left",
                                mode="prec_recall")

## Testing Performance
outpreds_fsnn <- predict(annMod, newdata=test_fsnn)
test_fs_nn_perf<-confusionMatrix(outpreds_fsnn, 
                                 test_fsnn$Exited, 
                                 positive="Left",
                                 mode="prec_recall")

## ---------------- Performance comparison following feature selection

## Training Set
train_fs_perf<-cbind(train_rf=train_rf_perf$byClass, 
                     train_fs_rf=train_fs_rf_perf$byClass,
                     train_ann=train_ann_perf$byClass,
                     train_fs_nn=train_fs_nn_perf$byClass)
train_fs_perf
##                      train_rf  train_fs_rf  train_ann train_fs_nn
## Precision            1.0000000   0.8675373 0.76190476   0.7673546
## Recall               1.0000000   0.5705521 0.48098160   0.5018405
## F1                   1.0000000   0.6883790 0.58969537   0.6068249
## Balanced Accuracy    1.0000000   0.7741318 0.72126305   0.7314571

## Testing Set
test_fs_perf<-cbind(test_rf=test_rf_perf$byClass, 
                    test_fs_rf=test_fs_rf_perf$byClass,
                    test_ann=test_ann_perf$byClass,
                    test_fs_nn=test_fs_nn_perf$byClass)
test_fs_perf
##                        test_rf  test_fs_rf  test_ann   test_fs_nn
## Precision            0.77200000 0.75903614 0.79746835  0.78000000
## Recall               0.47420147 0.46437346 0.46437346  0.47911548
## F1                   0.58751903 0.57621951 0.58695652  0.59360731
## Balanced Accuracy    0.71919873 0.71334251 0.71711136 0.72228387

## It is seen that excluding 5 less important variables affects the 
## classification result at a really slight degree (positively for ANN but 
## negatively for Random Forest). 
## Thus, we can have a simpler and less-complicated model while obtaining
## the same predictions.

## -------------------------- Class Imbalance

## As we noticed, churned customers are in minority.
## The advantage of undersampling over the "Stayed" class is that we are 
## not creating any artificial data points and because original data is sufficiently large
## we are not concerned about missing too much information.
## However, because the gap between two classes is not too large, 
## oversampling of the "Left" class doesn't also lead to serious concerns 
## regarding the insertion of artificially-created data points.

## We will test both methods, using SMOTE as a type of oversampling.
## In this section we perform resampling over the newly-created training and
## testing sets which include only top 5 attributes.

## Load libraries
## library(DMwR) #Resampling

## Undersampling
set.seed(831)
train_ds <- downSample(x=train_fs[,-6], 
                       y=train_fs$Exited, 
                       yname="Exited")
## New class distribution
table(train_ds$Exited)
## Left Stayed 
## 4890   6520 

## Oversampling using Synthetic Minority Oversampling Technique
set.seed(831)
train_sm <- SMOTE(Exited~., data=train_fs)
## New class distribution
table(train_sm$Exited)

## -------------------------- Random forest over the balanced Data

grids = expand.grid(mtry = seq(from = 1, to = 10, by = 1))

grid_ctrl <- trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 3,
                          search="grid")
## Downsampled
set.seed(831)
fit.rf_ds <- train(Exited~., 
                   data=train_ds, 
                   method="rf", 
                   trControl=grid_ctrl,
                   tuneGrid=grids)

fit.rf_ds
## The final value used for the model was mtry = 2.
## Accuracy   Kappa
## 0.7725971  0.5451943
plot(fit.rf_ds)

## Training Performance
rf.train_ds <- predict(fit.rf_fs, 
                       train_ds[,-6])

train_ds_rf_perf<-confusionMatrix(rf.train_ds, 
                                  train_ds$Exited,
                                  positive="Left",
                                  mode="prec_recall")
## Testing Performance
rf.test_ds <- predict(fit.rf_ds, 
                      test_fs[,-6])

test_ds_rf_perf<-confusionMatrix(rf.test_ds, 
                                 test_fs$Exited,
                                 positive="Left",
                                 mode="prec_recall")

## Upsampled (SMOTE)
set.seed(831)
fit.rf_sm <- train(Exited~., 
                   data=train_sm, 
                   method="rf", 
                   trControl=grid_ctrl,
                   tuneGrid=grids)

fit.rf_sm
## The final value used for the model was mtry = 5.
## Accuracy   Kappa
## 0.8678352  0.7263061
plot(fit.rf_sm)

## Training Performance
rf.train_sm <- predict(fit.rf_sm, 
                       train_sm[,-6])

train_sm_rf_perf<-confusionMatrix(rf.train_sm, 
                                  train_sm$Exited,
                                  positive="Left",
                                  mode="prec_recall")
## Testing Performance
rf.test_sm <- predict(fit.rf_sm, 
                      test_fs[,-6])

test_sm_rf_perf<-confusionMatrix(rf.test_sm, 
                                 test_fs$Exited,
                                 positive="Left",
                                 mode="prec_recall")

## ---------------- RF Performance comparison following resampling

## Training Set
train_rs_rf_perf<-cbind(train_fs_rf=train_fs_rf_perf$byClass,
                     train_ds_rf=train_ds_rf_perf$byClass,
                     train_sm_rf=train_sm_rf_perf$byClass)
train_rs_rf_perf
##                      train_fs_rf train_ds_rf train_sm_rf
## Precision              0.8675373   0.9648033   0.9880878
## Recall                 0.5705521   0.5717791   0.9668712
## F1                     0.6883790   0.7180277   0.9773643
## Balanced Accuracy      0.7741318   0.7754601   0.9790644

## Testing Set
test_rs_rf_perf<-cbind(test_fs_rf=test_fs_rf_perf$byClass,
                    test_ds_rf=test_ds_rf_perf$byClass,
                    test_sm_rf=test_sm_rf_perf$byClass)
test_rs_rf_perf
##                      test_fs_rf test_ds_rf test_sm_rf
## Precision            0.75903614  0.5031646  0.5277207
## Recall               0.46437346  0.7813268  0.6314496
## F1                   0.57621951  0.6121270  0.5749441
## Balanced Accuracy    0.71334251  0.7920453  0.7434886

## The overall improvement through balancing data is substantial
## with respect to all measures except for recall. 
## Other than precision, undersampling improves performance measures 
## to a greater extent than SMOTE.

## -------------------------- ANN over the balanced Data

grids_ann = expand.grid(size = seq(from = 1, to = 5, by = 1),
                        decay = seq(from = 0.1, to = 0.5, by = 0.1))

ctrl <- trainControl(method="repeatedcv",
                     number = 5,
                     repeats=3,
                     search="grid")

## Downsampled
set.seed(831)
train_dsnn <- downSample(x=train_fsnn[,-5], 
                       y=train_fsnn$Exited, 
                       yname="Exited")

set.seed(831)
annMod <- train(Exited~ ., data = train_dsnn, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids_ann,
                verbose=FALSE)

annMod
## The final values used for the model were size = 5 and decay = 0.1.
## Accuracy   Kappa 
## 0.7788344  0.5576687
plot(annMod)

## Training Performance
inpreds_dsnn <- predict(annMod, newdata=train_dsnn)
train_ds_nn_perf<-confusionMatrix(inpreds_dsnn, 
                                  train_dsnn$Exited, 
                                  positive="Left",
                                  mode="prec_recall")

## Testing Performance
outpreds_dsnn <- predict(annMod, newdata=test_fsnn)
test_ds_nn_perf<-confusionMatrix(outpreds_dsnn, 
                                 test_fsnn$Exited, 
                                 positive="Left",
                                 mode="prec_recall")

## Oversampling using Synthetic Minority Oversampling Technique
set.seed(831)
train_smnn <- SMOTE(Exited~., data=train_fsnn)

set.seed(831)
annMod <- train(Exited~ ., data = train_smnn, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids_ann,
                verbose=FALSE)

annMod
## The final values used for the model were size = 5 and decay = 0.2.
## Accuracy   Kappa 
## 0.7917908  0.5693388
plot(annMod)

## Training Performance
inpreds_smnn <- predict(annMod, newdata=train_smnn)
train_sm_nn_perf<-confusionMatrix(inpreds_smnn, 
                                  train_smnn$Exited, 
                                  positive="Left",
                                  mode="prec_recall")

## Testing Performance
outpreds_smnn <- predict(annMod, newdata=test_fsnn)
test_sm_nn_perf<-confusionMatrix(outpreds_smnn, 
                                 test_fsnn$Exited, 
                                 positive="Left",
                                 mode="prec_recall")

## ---------------- ANN Performance comparison following resampling

## Training Set
train_rs_nn_perf<-cbind(train_fs_nn=train_fs_nn_perf$byClass,
                        train_ds_nn=train_ds_nn_perf$byClass,
                        train_sm_nn=train_sm_nn_perf$byClass)
train_rs_nn_perf
##                      train_fs_nn train_ds_nn train_sm_nn
## Precision              0.7673546   0.8088140   0.7802370
## Recall                 0.5018405   0.7656442   0.7137014
## F1                     0.6068249   0.7866373   0.7454876
## Balanced Accuracy      0.7314571   0.7923313   0.7814673

## Testing Set
test_rs_nn_perf<-cbind(test_fs_nn=test_fs_nn_perf$byClass,
                       test_ds_nn=test_ds_nn_perf$byClass,
                       test_sm_nn=test_sm_nn_perf$byClass)
test_rs_nn_perf
##                      test_fs_nn test_ds_nn test_sm_nn
## Precision            0.78000000  0.5056911  0.5612648
## Recall               0.47911548  0.7641278  0.6977887
## F1                   0.59360731  0.6086106  0.6221249
## Balanced Accuracy    0.72228387  0.7865865  0.7791707

## The overall improvement through balancing data is not substantial
## with respect to all measures except for recall.
## Undersampling and SMOTE, either one, improves the performance measures 
## to a different extent. In terms of Precision and Recall, SMOTE 
## suggests a bit better results but for F1 and Balanced Accuracy, 
## undersampling performs better.

## -------------------------- Performance in terms of ROC
## --------- Random Forest
ctrl_rocgrid <- trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 3,
                             search = "grid",
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             savePredictions = "final")
## Undersampled
set.seed(831)
fit.rf_ds_roc <- train(Exited~., 
                   data=train_ds, 
                   method = "rf", 
                   trControl = ctrl_rocgrid, 
                   tuneGrid = grids,
                   metric = "ROC")
fit.rf_ds_roc
## The final value used for the model was mtry = 2.
## ROC        Sens       Spec     
## 0.8547976  0.7550102  0.7901840
plot(fit.rf_ds_roc)

## Upsampled
set.seed(831)
fit.rf_sm_roc <- train(Exited~., 
                       data=train_sm, 
                       method = "rf", 
                       trControl = ctrl_rocgrid, 
                       tuneGrid = grids,
                       metric = "ROC")
fit.rf_sm_roc
## The final value used for the model was mtry = 5.
## ROC        Sens       Spec     
## 0.9356802  0.7895706  0.9265337
plot(fit.rf_sm_roc)

## --------- ANN

## Underampled
set.seed(831)
annMod_ds_roc <- train(Exited~ ., 
                data = train_dsnn, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl_rocgrid, 
                tuneGrid=grids_ann,
                metric = "ROC",
                verbose=FALSE)
annMod_ds_roc
## The final values used for the model were size = 5 and decay = 0.1.
## ROC        Sens       Spec 
## 0.8611154  0.7474438  0.8102249
plot(annMod_ds_roc)

## Upsampled
set.seed(831)
annMod_sm_roc <- train(Exited~ ., 
                       data = train_smnn, 
                       method = "nnet", 
                       maxit=500,
                       trControl = ctrl_rocgrid, 
                       tuneGrid=grids_ann,
                       metric = "ROC",
                       verbose=FALSE)
annMod_sm_roc
## The final values used for the model were size = 5 and decay = 0.2.
## ROC        Sens       Spec 
## 0.8697712  0.7054533  0.8565440
plot(annMod_ds_roc)

## Overall, SMOTE returns a larger ROC value than undersampling
## and Random forest achieves the largest ROC value through it.

## ----------------- RF & ANN Performance comparison following resampling

train_rs_perf<-cbind(train_rs_rf_perf [,-1], train_rs_nn_perf[,-1])
train_rs_perf

##                      train_ds_rf train_sm_rf  train_ds_nn train_sm_nn
## Precision             0.9648033  0.9880878    0.8088140   0.7802370
## Recall                0.5717791  0.9668712    0.7656442   0.7137014
## F1                    0.7180277  0.9773643    0.7866373   0.7454876
## Balanced Accuracy     0.7754601  0.9790644    0.7923313   0.7814673

test_rs_perf<-cbind(test_rs_rf_perf [,-1], test_rs_nn_perf[,-1])
test_rs_perf

##                      test_ds_rf test_sm_rf test_ds_nn test_sm_nn
## Precision             0.5031646  0.5277207  0.5056911  0.5612648
## Recall                0.7813268  0.6314496  0.7641278  0.6977887
## F1                    0.6121270  0.5749441  0.6086106  0.6221249
## Balanced Accuracy     0.7920453  0.7434886  0.7865865  0.7791707

## Random forest seems more prone to suggest an overfitted model, particularly in SMOTE mode. 
## ANN model results in overfitting to a lower extent, particularly in SMOTE mode. 

## Given these results, ANN model oversampled by SMOTE method seems to be 
## a slightly better predictive model than random forest for this problem.

## -------------------------- Testing models in the absence of outliers

## In this section we re-create models this time for
## selected features and oversampled data by SMOTE Method

## From Age and CreditScore which were recognized with outliers, only
## Age is included in the model and includes pretty more outliers 
## to be concerned about.

## 486 outliers were detected in the 'Stayed' class of Age variable
outliers<-boxplot.stats(df$Age[df$Exited=='Stayed'])$out

## Excluding rows including outliers from the analysis
df4 <- df[-which(df$Age %in% outliers),]

## -------------------------- Random Forest

set.seed(831)
sub <- createDataPartition(df4$Exited, p=0.80, list=FALSE)
train_rf_clean = df4[sub, ] 
test_rf_clean = df4[-sub, ]

## RF on selected features
train_fs1 <- train_rf_clean[, colnames(train_rf_clean) %in% c('Exited',churn_rfe$optVariables[1:5])]
test_fs1 <- test_rf_clean[,colnames(test_rf_clean) %in% c('Exited',churn_rfe$optVariables[1:5])]

## Oversampling using Synthetic Minority Oversampling Technique
set.seed(831)
train_sm1 <- SMOTE(Exited~., data=train_fs1)


set.seed(831)
fit.rf_clean <- train(Exited~., 
                   data=train_sm1, 
                   method="rf", 
                   trControl=grid_ctrl,
                   tuneGrid=grids)

fit.rf_clean
## The final value used for the model was mtry = 10.
## Accuracy   Kappa
## 0.8697196  0.7309350
plot(fit.rf_clean)

## Training Performance
rf.train_clean <- predict(fit.rf_clean, 
                       train_sm1[,-6])

train_clean_rf_perf<-confusionMatrix(rf.train_clean, 
                                  train_sm1$Exited,
                                  positive="Left",
                                  mode="prec_recall")
## Testing Performance
rf.test_clean <- predict(fit.rf_clean, 
                      test_fs1[,-6])

test_clean_rf_perf<-confusionMatrix(rf.test_clean, 
                                 test_fs1$Exited,
                                 positive="Left",
                                 mode="prec_recall")

## -------------------------- ANN
## Preprocessing
dum1 <- dummyVars(~Gender+Geography, data=df4,
                 sep="_", fullRank = TRUE)

dfann4 <- predict(dum1, df4)

## keeping all columns in df4 except our original categorical variables
dfann5 <- data.frame(df4[,!names(df4) %in% c("Gender", "Geography")], dfann4)

## Rescaling is required as ANN does best when data ranges between 0-1. 
## Thus, we will use the min-max normalization to rescale our data.
dfann6 <- preProcess(dfann5, method="range")
dfann_clean <- predict(dfann6, dfann5)
summary(dfann_clean)

## Training and Testing
## Splitting the data into training and testing sets using an 80/20 split rule

set.seed(831)
sub <- createDataPartition(dfann_clean$Exited, p=0.80, list=FALSE)
train_clean = dfann_clean[sub, ] 
test_clean = dfann_clean[-sub, ]

## ANN on selected Features
train_fsnn1 <- train_clean[, colnames(train_clean) %in% 
                                c('Exited',"Age","NumOfProducts","IsActiveMember","Balance",
                                  "Geography_Germany","Geography_Spain")]

test_fsnn1 <- test_clean[,colnames(test_clean) %in% 
                              c('Exited',"Age","NumOfProducts","IsActiveMember","Balance",
                                "Geography_Germany","Geography_Spain")]

## Oversampling using Synthetic Minority Oversampling Technique
set.seed(831)
train_smnn1 <- SMOTE(Exited~., data=train_fsnn1)

set.seed(831)
annMod_clean <- train(Exited~ ., data = train_smnn1, 
                method = "nnet", 
                maxit=500,
                trControl = ctrl, 
                tuneGrid=grids_ann,
                verbose=FALSE)

annMod_clean
## The final values used for the model were size = 5 and decay = 0.1.
## Accuracy   Kappa 
## 0.7954870  0.5782677
plot(annMod)


## Training Performance
inpreds_clean_nn <- predict(annMod_clean, newdata=train_smnn1)
train_clean_nn_perf<-confusionMatrix(inpreds_clean_nn, 
                                  train_smnn1$Exited, 
                                  positive="Left",
                                  mode="prec_recall")

## Testing Performance
outpreds_clean_nn <- predict(annMod_clean, newdata=test_fsnn1)
test_clean_nn_perf<-confusionMatrix(outpreds_clean_nn, 
                                 test_fsnn1$Exited, 
                                 positive="Left",
                                 mode="prec_recall")

## ---------------- Performance comparison following data cleaning

## Training Set
train_clean_perf<-cbind(train_rf=train_sm_rf_perf$byClass,
                           train_clean_rf=train_clean_rf_perf$byClass,
                           train_nn=train_sm_nn_perf$byClass,
                           train_clean_nn=train_clean_nn_perf$byClass)
train_clean_perf
##                       train_rf train_clean_rf  train_nn train_clean_nn
## Precision            0.9880878      0.9844852 0.7802370      0.7874251
## Recall               0.9668712      0.9728223 0.7137014      0.7331010
## F1                   0.9773643      0.9786190 0.7454876      0.7592927
## Balanced Accuracy    0.9790644      0.9806620 0.7814673      0.7923345

## Testing Set
test_clean_perf<-cbind(test_rf=test_sm_rf_perf$byClass,
                          test_clean_rf=test_clean_rf_perf$byClass,
                          test_nn=test_sm_nn_perf$byClass,
                          test_clean_nn=test_clean_nn_perf$byClass)
test_clean_perf
##                      test_rf   test_clean_rf   test_nn test_clean_nn
## Precision            0.5277207     0.4492441 0.5612648     0.5117188
## Recall               0.6314496     0.5810056 0.6977887     0.7318436
## F1                   0.5749441     0.5066991 0.6221249     0.6022989
## Balanced Accuracy    0.7434886     0.7052185 0.7791707     0.7823097

## Overall, it is seen that excluding outliers doesn't improve the performance.
## Interestingly the results are slightly worse for random forest after 
## diminishing the size of majority class following this removal. 
## Thus, again ANN seems to be more robust to noises of this type. 

## --------------------------
save.image("Project_Charandabi.RData")


