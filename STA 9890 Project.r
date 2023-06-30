rm(list = ls())    #delete objects
cat("\014")        #clear console
setwd("/Users/ella/Desktop/Data Mining/Final Project")
library(tidyverse); library(modelr); ## packages for data manipulation and computing rmse easily.
library(glmnet)
library(glmnetUtils)
library(MASS)
library(tree)
library(randomForest)

rsq <- function(y, y.hat) {
  1 - mean((y - y.hat) ^ 2) / mean((y - mean(y))^2)
}

#Part 3
## read data
var.names <- read_csv("varnames.csv", col_names = FALSE) %>% pull(1)
d <- read_csv("traindata.csv", col_names = var.names)
names(d) <- tolower(names(d))
## remove samples with no data on response variable
d <- d %>% filter(!is.na(caravan))
#d <- d %>% select(!is.na(d))


## Part 3
#100 samples for Rsquare training and test data
M              =     2
Rsq.test.ls    =     rep(0,M)  # ls = lasso
Rsq.train.ls   =     rep(0,M)
Rsq.test.ridge    =     rep(0,M)
Rsq.train.ridge   =     rep(0,M)
Rsq.test.rf= rep(0,M)
Rsq.train.rf= rep(0,M)

set.seed(123)
n=nrow(d)
n.train        =     floor(0.8*n)
n.test         =     n-n.train
#fitting data onto matrix
X= model.matrix(caravan~.,data=d)
y=d$caravan

for (m in c(1:M)) {
  #indexing
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit ridge and calculate and record the train and test R squares 
  cat(sprintf("Running Ridge Regression Iteration %d\n", m))
  a=0 # ridge
  cv.fit.ridge           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
  y.hat.train      =     predict(fit, newx = X.train, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
  y.hat.test       =     predict(fit, newx = X.test, type = "response") # y.hat.test=X.test %*% fit$beta  + fit$a0
  Rsq.train.ridge[m]  =  rsq(y.train, y.hat.train) 
  Rsq.test.ridge[m]   =  rsq(y.test, y.hat.test) # 1-mean((y.test - y.hat.test)^2)/mean((y.test - mean(y.test))^2)
  #Rsquare values
  cat(sprintf("m=%3.f| Rsq.test.ridge=%.2f|  Rsq.train.ridge=%.2f\n", m,  
             Rsq.test.ridge[m], Rsq.train.ridge[m]))
  
  # fit lasso and calculate and record the train and test R squares 
  cat(sprintf("Running Lasso Regression Iteration %d\n", m))
  a=1 # lasso
  cv.fit.ls           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
  y.hat.train      =     predict(fit, newx = X.train, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
  y.hat.test       =     predict(fit, newx = X.test, type = "response") # y.hat.test=X.test %*% fit$beta  + fit$a0
  Rsq.train.ls[m]  =     rsq(y.train, y.hat.train)
  Rsq.test.ls[m]   =     rsq(y.test, y.hat.test)

  #Rsquare values
  cat(sprintf("m=%3.f| Rsq.test.ls=%.2f|  Rsq.train.ridge=%.2f\n", m,  
              Rsq.test.ls[m], Rsq.train.ls[m]))
  
  
  #fitting random forest
  cat(sprintf("Running Random Forest Iteration %d\n", m))
  p=dim(d)[2]-1
  rf.d  =  randomForest(caravan~., data=d, subset = train, mtry=floor(sqrt(p)), importance=TRUE)
  y.hat.test   = predict(rf.d, newdata = d[-train,])
  y.hat.train   = predict(rf.d, newdata = d[train,])
  y.test       = y[test]
  Rsq.test.rf[m]=    rsq(y.test, y.hat.test)
  Rsq.train.rf[m]=     rsq(y.train, y.hat.train) 
  
  cat(sprintf("m=%3.f| Rsq.test.rf=%.2f| Rsq.train.rf=%.2f\n", m,  
              Rsq.test.rf[m], Rsq.train.rf[m]))
}

rsq.df <- as_tibble(cbind(Rsq.test.ls, Rsq.train.ls, Rsq.test.ridge, Rsq.train.ridge, 
                Rsq.test.rf, Rsq.train.rf))
write_csv(rsq.df, "rsq values.csv")

a <- read_csv("rsq values.csv")
Rsq.test.ls <- a$Rsq.test.ls
Rsq.train.ls <- a$Rsq.train.ls
Rsq.test.ridge <- a$Rsq.test.ridge
Rsq.train.ridge <- a$Rsq.train.ridge
Rsq.test.rf <- a$Rsq.test.rf
Rsq.train.rf <- a$Rsq.train.rf
aa <- tibble(Rsq.ls = c(Rsq.train.ls, Rsq.test.ls),
             Rsq.ridge = c(Rsq.train.ridge, Rsq.test.ridge),
             Rsq.rf = c(Rsq.train.rf, Rsq.test.rf),
             Type=c(rep("train", M), rep("test", M)))

b <- tibble(Rsq.test=c(Rsq.test.ls, Rsq.test.ridge, Rsq.test.rf), 
            Rsq.train = c(Rsq.train.ls, Rsq.train.ridge, Rsq.train.rf),
            Model=c(rep("lasso", 100), rep("ridge", 100), rep("rf", 100)))

par(mfrow=c(1, 1))
boxplot(Rsq.test ~ Model, data=b, xlab="Test", ylab="R squared")
boxplot(Rsq.train ~ Model, data=b, xlab="Train", ylab="R squared")

bb1 <- tibble(Rsq=c(Rsq.test.ls, Rsq.test.ridge, Rsq.test.rf), Type="test")
bb2 <- tibble(Rsq=c(Rsq.train.ls, Rsq.train.ridge, Rsq.train.rf), Type="train")
bb <- bb1 %>% bind_rows(bb2)

#plots and boxplots for Rsquare training and testing 
par(mfrow = c(1,1))
boxplot(b$Rsq.test, xlab = "Test", ylab = "R squared")
boxplot(b$Rsq.train, xlab = "Train", ylab = "R squared")
boxplot(bb$Rsq ~ bb$Type, ylab="R squared")

#importance and variance importance plot for random forest
importance(rf.d)
varImpPlot(rf.d)



## Part 4
#The main goal of the project is to assess which predictors best predict if a customer gets a mobile home policy.

#Response Variable: Caravan, or if a family gets or does not get a mobile home policy.

#Predictor Variables: Socioeconomic variables such as average size of household and product ownership variables such as contribution boat policies. 

#n=5,822 and p=85 variables

#Number of categorical predictors: 5

#Number of numerical predictors: 80

df<-data.frame(fits=c('Ridge Train','Ridge Test', 'RF Train', 'RF Test', 
                      'Lasso Train', 'Lasso Test'),rsquares=c(Rsq.train.ridge[1],Rsq.test.ridge[1],Rsq.train.rf[1],
                                                             Rsq.test.rf[1],Rsq.train.ls[1],Rsq.test.ls[1]))


boxplot(df$rsquares~df$fits, main='Boxplots of Rsquared train and Rsquared test', xlab='Fits', ylab='R-squared values')

start.lassosample<-Sys.time()
a=1 # lasso
cv.fit.ls.sample           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
fit.ls.sample           =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
plot(cv.fit.ls.sample)
end.lassosample<-Sys.time()
time.lassosample <- end.lassosample - start.lassosample
round(time.lassosample,digits=2)

start.ridgesample<-Sys.time()
a=0 # ridge
cv.fit.ridge.sample =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
fit.ridge.sample    =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
plot(cv.fit.ridge.sample)
end.ridgesample<-Sys.time()
time.ridgesample <- end.ridgesample - start.ridgesample
round(time.ridgesample,digits=2)


## Part 5
#fit entire data for lasso, ridge, and random forest
Rsq.full.ridge    =   rep(0,M)
Rsq.full.ls   =     rep(0,M)
Rsq.full.rf= rep(0,M)
for (m in c(1:M)) {
  
  set.seed(123)
  n=nrow(d)
  
  #fitting data
  X= model.matrix(caravan~.,data=d)
  y=d$caravan
  
  #indexing
  shuffled_indexes =     sample(n)
  full            =     shuffled_indexes[1:n]
  X.full         =     X[full, ]
  y.full          =     y[full]
  
  # fit ridge and calculate and record the R square 
  a=0 # ridge
  cv.fit.ridge           =     cv.glmnet(X.full, y.full, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.full, y.full,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
  y.full.hat      =       predict(fit, newx = X.full, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
  Rsq.full.ridge[m]   =     1-mean((y.full - y.full.hat)^2)/mean((y.full - mean(y.full))^2)
  
  # fit lasso and calculate and record the R square 
  a=1 # lasso
  cv.fit.ls           =     cv.glmnet(X.full, y.full, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.full, y.full,intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
  y.full.hat      =     predict(fit, newx = X.full, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
  Rsq.full.ls[m]   =     1-mean((y.full - y.full.hat)^2)/mean((y.full - mean(y.full))^2)
  
  cat(sprintf("m=%3.f| Rsq.full.ridge=%.2f,  Rsq.full.ls=%.2f\n", m,  
              Rsq.full.ridge[m], Rsq.full.ls[m]))
  #plots and boxplots for ridge and lasso regression
  plot(cv.fit.ridge)
  plot(cv.fit.ls)
  boxplot(Rsq.full.ridge)
  boxplot(Rsq.full.ls)
  
  #fit random forest
  p=dim(d)[2]-1
  rf.full  =  randomForest(caravan~., data=d, mtry= floor(sqrt(p)), importance=TRUE)
  y.hat.full   = predict(rf.d, newdata = d)
  Rsq.full.rf[m]=   1-mean((y.full - y.hat.full)^2)/mean((y.full - mean(y.full))^2)
  
  cat(sprintf("m=%3.f| Rsq.full.rf=%.2f\n", m,  
              Rsq.full.rf[m]))
  
}
#importance and variance importance plot for random forest
importance(rf.full)
varImpPlot(rf.full)


#90% CI for R2 and Time to Fit Models w/All Data

ridge.90<-quantile(Rsq.full.ridge,c(.025,.925))
ls.90<-quantile(Rsq.full.ls,c(.025,.925))
rf.90<-quantile(Rsq.full.rf,c(.025,.925))

#Time to Fit a Single Lasso, Ridge, and Random Forest
set.seed(123)
n=nrow(d)


X= model.matrix(caravan~.,data=d)
y=d$caravan

shuffled_indexes =     sample(n)
full            =     shuffled_indexes[1:n]
X.full         =     X[full, ]
y.full          =     y[full]

# fit ridge and calculate R-squared and record time it takes to fit ridge
start.ridge <- Sys.time()
a=0 # ridge
cv.fit.ridge2           =     cv.glmnet(X.full, y.full, intercept = FALSE, alpha = a, nfolds = 10)
fit2              =     glmnet(X.full, y.full,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
y.full.hat2      =       predict(fit, newx = X.full, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
Rsq.full.ridge2   =     1-mean((y.full - y.full.hat)^2)/mean((y.full - mean(y.full))^2)
end.ridge<-Sys.time()
time.ridge <- end.ridge - start.ridge

# fit lasso and calculate R-squared and record time it takes to fit lasso 
start.lasso<- Sys.time()
a=1 # lasso
cv.fit.ls2           =     cv.glmnet(X.full, y.full, intercept = FALSE, alpha = a, nfolds = 10)
fit2              =     glmnet(X.full, y.full,intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
y.full.hat2      =     predict(fit, newx = X.full, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
Rsq.full.ls2   =     1-mean((y.full - y.full.hat)^2)/mean((y.full - mean(y.full))^2)
end.lasso <- Sys.time()
time.lasso <- end.lasso - start.lasso


cat(sprintf("Rsq.full.ridge=%.2f,  Rsq.full.ls=%.2f\n",
            Rsq.full.ridge2, Rsq.full.ls2))

plot(cv.fit.ridge2)
plot(cv.fit.ls2)
boxplot(Rsq.full.ridge2)
boxplot(Rsq.full.ls2)

#fit random forest and calculate R-squared and record time it takes to fit random forest
start.rf <- Sys.time()
p=dim(d)[2]-1
rf.full2  =  randomForest(caravan~., data=d, mtry= floor(sqrt(p)), importance=TRUE)
y.hat.full   = predict(rf.d, newdata = d)
Rsq.full.rf2=   1-mean((y.full - y.hat.full)^2)/mean((y.full - mean(y.full))^2)
end.rf<-Sys.time()
time.rf <- end.rf - start.rf
cat(sprintf("Rsq.full.rf=%.2f\n",
            Rsq.full.rf2))

#Table of CI and Time it takes to fit lasso, ridge, and random forest
tab <- matrix(c(ridge.90,round(time.ridge,digits=2),ls.90,round(time.lasso,digits=2),
                rf.90,round(time.rf,digits=2)), ncol=3, byrow=TRUE)
colnames(tab) <- c('90% CI R-squared interval','90% CI R-squared Interval','Time To Fit')
rownames(tab) <- c('Ridge','Lasso','Random Forest')
tab <- as.table(tab)

tab


