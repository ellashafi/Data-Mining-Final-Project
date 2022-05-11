library(tidyverse)
library(modelr)
library(glmnet)
library(glmnetUtils)
library(MASS)
library(tree)
library(randomForest)
library(knitr)

rsq <- function(y, y.hat) {
  1 - mean((y - y.hat) ^ 2) / mean((y - mean(y))^2)
}

## read data
#setwd("/Users/ella/Desktop/Data Mining/Final Project")
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
  cat(sprintf("m=%3.f| Rsq.test.ls=%.2f|  Rsq.train.ls=%.2f\n", m,  
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

# rsq.df <- as_tibble(cbind(Rsq.test.ls, Rsq.train.ls, Rsq.test.ridge, Rsq.train.ridge, 
#                           Rsq.test.rf, Rsq.train.rf))
# write_csv(rsq.df, "rsq values.csv")

## Part 4b
b <- tibble(Rsq.test=c(Rsq.test.ls, Rsq.test.ridge, Rsq.test.rf), 
            Rsq.train = c(Rsq.train.ls, Rsq.train.ridge, Rsq.train.rf),
            Model=c(rep("Lasso", M), rep("Ridge", M), rep("RF", M)))

par(mfrow=c(1, 2))
boxplot(Rsq.train ~ Model, data=b, xlab="Train", ylab="R squared")
boxplot(Rsq.test ~ Model, data=b, xlab="Test", ylab="")

## Part 4c
start.lassosample<-Sys.time()
a=1 # lasso
cv.fit.ls.sample           =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
fit.ls.sample           =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
end.lassosample<-Sys.time()
time.lassosample <- end.lassosample - start.lassosample

start.ridgesample<-Sys.time()
a=0 # ridge
cv.fit.ridge.sample =     cv.glmnet(X.train, y.train, intercept = FALSE, alpha = a, nfolds = 10)
fit.ridge.sample    =     glmnet(X.train, y.train,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
end.ridgesample<-Sys.time()
time.ridgesample <- end.ridgesample - start.ridgesample

par(mfrow=c(1, 2))
plot(cv.fit.ls.sample, sub='Lasso')
plot(cv.fit.ridge.sample, sub='Ridge')

time.tab = tibble(method=c('Lasso', 'Ridge'), time=c(round(time.lassosample,digits=2),round(time.ridgesample,digits=2)))
kable(time.tab, col.names = c("Method", "Run Time"))

## Part 5
#fit entire data for lasso, ridge, and random forest
# fit ridge and calculate and record the R square 
cat(sprintf("Running Full Ridge Regression"))
start.ridge <- Sys.time()
a=0 # ridge
cv.fit.ridge           =     cv.glmnet(X, y, intercept = FALSE, alpha = a, nfolds = 10)
fit              =     glmnet(X, y,intercept = FALSE, alpha = a, lambda = cv.fit.ridge$lambda.min)
y.hat.full      =       predict(fit, newx = X, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
Rsq.ridge.full  =     rsq(y, y.hat.full)
end.ridge<-Sys.time()
time.ridge <- end.ridge - start.ridge

# fit lasso and calculate and record the R square 
cat(sprintf("Running Lasso full Regression"))
start.lasso<- Sys.time()
a=1 # lasso
cv.fit.ls           =     cv.glmnet(X, y, intercept = FALSE, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, intercept = FALSE, alpha = a, lambda = cv.fit.ls$lambda.min)
y.hat.full      =     predict(fit, newx = X, type = "response") # y.hat.train=X.train %*% fit$beta + fit$a0
Rsq.ls.full  =     rsq(y, y.hat.full)
end.lasso <- Sys.time()
time.lasso <- end.lasso - start.lasso

#fit random forest
cat(sprintf("Running full Random Forest"))
start.rf <- Sys.time()
p=dim(d)[2]-1
rf.full  =  randomForest(caravan~., data=d, mtry= floor(sqrt(p)), importance=TRUE)
y.hat.full   = predict(rf.d, newdata = d)
Rsq.rf.full=    rsq(y, y.hat.full)
end.rf<-Sys.time()
time.rf <- end.rf - start.rf


#plots and boxplots for ridge and lasso regression
plot(cv.fit.ridge)
plot(cv.fit.ls)
boxplot(Rsq.ridge.full)
boxplot(Rsq.ls.full)

#importance and variance importance plot for random forest
importance(rf.full)
varImpPlot(rf.full)


#90% CI for R2 and Time to Fit Models w/All Data

ci.90.ridge <- quantile(Rsq.test.ridge, c(0.05, 0.95))
ci.90.ls <- quantile(Rsq.test.ls, c(0.05, 0.95))
ci.90.rf <- quantile(Rsq.test.rf, c(0.05, 0.95))

print(c(time.ridge, time.lasso,time.rf))
cat(sprintf("Ridge time = %0.2f \nLasso time = %0.2f \nRandom Forest time = %0.2f \n", time.ridge, time.lasso, time.rf))

#Final table
result.table <- tibble(method = c("Ridge", "Lasso", "Random Forest"), 
                       CI.05 = c(ci.90.ridge[1], ci.90.ls[1], ci.90.rf[1]),
                       CI.95 = c(ci.90.ridge[2], ci.90.ls[2], ci.90.rf[2]),
                       time.full = c(round(time.ridge,2), round(time.lasso,2), round(time.rf,2)))

kable(result.table, col.names = c("Method", "CI %90 LB", "CI %90 UB", "Full Run Time"))
save(list = ls(), file = "allresults.RData")
                       