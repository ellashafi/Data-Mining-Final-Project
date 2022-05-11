rm(list = ls())    #delete objects
cat("\014")        #clear console
setwd("/Users/ella/Desktop/Data Mining/Final Project/")
library(tidyverse); library(modelr); ## packages for data manipulation and computing rmse easily.
library(glmnet)
library(glmnetUtils)

## read data
var.names <- read_csv("varnames.csv", col_names = FALSE) %>% pull(1)
d <- read_csv("traindata.csv", col_names = var.names)
names(d) <- tolower(names(d))
## remove samples with no data on response variable
d <- d %>% filter(!is.na(mostype))
#d <- d %>% select(!is.na(d))
