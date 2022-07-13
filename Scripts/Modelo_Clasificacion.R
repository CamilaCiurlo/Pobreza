##################################################
####### ALUMNO: PAOLO VALCARCEL PINEDA ###########
######## ALUMNA: CAMILA CIURCLO ARAGON ###########
###### CURSO: BIG DATA Y MACHINE LEARNING ########
############ PROBLEM SET: POVERTY ################
##################################################


setwd("D:/NUEVO D/UNIANDES/BIG DATA/Problem Set 2")
TH<-readRDS("test_hogares.Rds")
TP<-readRDS("test_personas.Rds")
TRH<-readRDS("train_hogares.Rds")
TRP<-readRDS("train_personas.Rds")

library(pacman)
p_load(rio, tidyverse, skimr, caret)
library(ggplot2)
require("stargazer")
library(ggplot2)
library(patchwork)
library(caTools)
library(foreign)
library(RColorBrewer)
library(stargazer)
library(dplyr)
library("nycflights13")
library("gamlr")
library(car)

TRF <- as.data.frame(left_join(x=TRP , y=TRH, by=c("id")))
head(TRF$Ingtot)
TRH<-NULL
TRP<-NULL

TF <- as.data.frame(left_join(x=TP , y=TH, by=c("id")))
TH<-NULL
TP<-NULL

#Generación de variables
TRF$Edad <- TRF$P6040
TRF$Edad2 <- TRF$P6040^2
TRF$Sexo <- recode(TRF$P6020, "1=1; 2=0")
TRF$Jefe <- ifelse(TRF$P6050==1,1,0)
TRF$Vivienda <- ifelse(TRF$P5090==1 | TRF$P5090==2,1,0) 
#Años de educación
TRF$Esc <- ifelse(TRF$P6210==1 | TRF$P6210==2,0,NA)
TRF$Esc <- ifelse(TRF$P6210==3,TRF$P6210s1,TRF$Esc)
TRF$Esc <- ifelse(TRF$P6210==4,TRF$P6210s1,TRF$Esc)
TRF$Esc <- ifelse(TRF$P6210==4 & TRF$P6210s1==0,5,TRF$Esc)
TRF$Esc <- ifelse(TRF$P6210==5,TRF$P6210s1,TRF$Esc)
TRF$Esc <- ifelse(TRF$P6210==6 & TRF$P6210s1==0,11,TRF$Esc)
TRF$Esc <- ifelse(TRF$P6210==6,TRF$P6210s1+11,TRF$Esc)
TRF$Esc[is.na(TRF$Esc)] = 0

#Habitaciones
TRF$P500 <- ifelse(TRF$P500==98,0,TRF$P500)
TRF$Habit <- TRF$P500


#Comprobaciones
table(TRF$Vivienda)
table(TRF$Pobre)
is.na(TRF$Esc) %>% table
is.na(TRF$Nper) %>% table
is.na(TRF$Sexo) %>% table
is.na(TRF$Jefe) %>% table
is.na(TRF$Vivienda) %>% table
is.na(TRF$Edad) %>% table
is.na(TRF$Esc) %>% table
is.na(TRF$Habit) %>% table

#Categorizando
TRF$Pobre <- factor(TRF$Pobre, labels = c("No", "Si"))
TRF$Sexo <- factor(TRF$Sexo, labels = c("No", "Si"))
TRF$Jefe <- factor(TRF$Jefe, labels = c("No", "Si"))
TRF$Vivienda <- factor(TRF$Vivienda, labels = c("No", "Si"))

TRF <- subset(TRF, select = c(id,Pobre,Nper,Sexo,Jefe,Vivienda,Edad,Edad2,Esc,Habit))

########################
##### LOGIT LASSO ######
########################

require(caret)
## First, split the training set off
set.seed(10101)
split1 <- createDataPartition(TRF$Pobre, p = .7)[[1]]
length(split1)

other <- TRF[-split1,]
training <- TRF[ split1,]

set.seed(10101)
split2 <- createDataPartition(other$Pobre, p = 1/3)[[1]]
evaluation <- other[ split2,]
testing <- other[-split2,]

dim(training)
dim(testing)
dim(evaluation)

#Maximización de la capacidad predictiva del modelo
ctrl_def <- trainControl(method = "cv",
                         number = 5,
                         summaryFunction = defaultSummary,
                         classProbs = TRUE,
                         verbose=FALSE,
                         savePredictions = T)
#logit
make.names(c("No", "Si"), unique = TRUE)
TRF$Pobre[make.names(TRF$Pobre) != TRF$Pobre] 
TRF$Sexo[make.names(TRF$Sexo) != TRF$Sexo] 
TRF$Jefe[make.names(TRF$Jefe) != TRF$Jefe] 
TRF$Vivienda[make.names(TRF$Vivienda) != TRF$Vivienda]

set.seed(10101)
logit_caret_def <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glm", 
  trControl = ctrl_def,
  family = "binomial",
  preProcess = c("center", "scale")
)


logit_caret_def


ctrl_two <- trainControl(method = "cv",
                         number = 5,
                         summaryFunction = twoClassSummary,
                         classProbs = TRUE,
                         verbose=FALSE,
                         savePredictions = T)
set.seed(10101)
logit_caret_two <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glm", #for logit
  trControl = ctrl_two,
  family = "binomial",
  preProcess = c("center", "scale")
)

logit_caret_two 


fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))
ctrl<- trainControl(method = "cv",
                    number = 5,
                    summaryFunction = fiveStats,
                    classProbs = TRUE,
                    verbose=FALSE,
                    savePredictions = T)
#logit
set.seed(10101)
logit_caret <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glm", #for logit
  trControl = ctrl,
  family = "binomial",
  preProcess = c("center", "scale")
)

logit_caret

#Lasso
lambda_grid <- 10^seq(-4, 0.01, length = 10) 
lambda_grid

set.seed(10101)
logit_lasso_acc <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "Accuracy",
  tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_lasso_acc


set.seed(10101)
logit_lasso_roc <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_lasso_roc

set.seed(10101)
logit_caret_sens <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "Sens",
  tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_caret_sens 

#Ridge
set.seed(10101)
logit_ridge_acc <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "Accuracy",
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_ridge_acc

set.seed(10101)
logit_ridge_roc <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_ridge_roc


set.seed(10101)
logit_ridge_sens <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc + Habit, 
  data = training,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "Sens",
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_ridge_sens

#Evaluando en Lasso
evalResults <- data.frame(Pobre = evaluation$Pobre)
evalResults$Roc <- predict(logit_lasso_roc,
                           newdata = evaluation,
                           type = "prob")[,1]

library(pROC)
rfROC <- roc(evalResults$Pobre, evalResults$Roc, levels=rev(levels(evalResults$Pobre)))
rfROC
rfThresh <- coords(rfROC, x = "best", best.method = "closest.topleft")

#Punto de corte
rfThresh
evalResults<-evalResults %>% mutate(hat_def_05=ifelse(evalResults$Roc>0.5,"Si","No"),
                                    hat_def_rfThresh=ifelse(evalResults$Roc>rfThresh$threshold,"Si","No"))

with(evalResults,table(Pobre,hat_def_05))
with(evalResults,table(Pobre,hat_def_rfThresh))


#Evaluando en Ridge
evalResults2 <- data.frame(Pobre = evaluation$Pobre)
evalResults2$Roc <- predict(logit_ridge_roc,
                           newdata = evaluation,
                           type = "prob")[,1]

library(pROC)
rfROC2 <- roc(evalResults2$Pobre, evalResults2$Roc, levels=rev(levels(evalResults2$Pobre)))
rfROC2
rfThresh2 <- coords(rfROC2, x = "best", best.method = "closest.topleft")

#Punto de corte
rfThresh2
evalResults2<-evalResults2 %>% mutate(hat_def_05=ifelse(evalResults2$Roc>0.5,"Si","No"),
                                    hat_def_rfThresh2=ifelse(evalResults2$Roc>rfThresh2$threshold,"Si","No"))

with(evalResults2,table(Pobre,hat_def_05))
with(evalResults2,table(Pobre,hat_def_rfThresh2))


##########################
####### REMUESTREO #######
##########################

#### Lasso #####
#Down-sampling
set.seed(10101)
upSampledTrain <- upSample(x = training,
                           y = training$Pobre,
                           yname = "Pobre")

dim(training)
dim(upSampledTrain)
table(upSampledTrain$Pobre)

set.seed(10101)
logit_lasso_upsample <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc,
  data = upSampledTrain,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_lasso_upsample


#Down-sampling
set.seed(10101)
downSampledTrain <- downSample(x = training,
                               y = training$Pobre,
                               yname = "Default")
dim(training)
dim(downSampledTrain)
table(downSampledTrain$Pobre)

set.seed(10101)
logit_lasso_downsample <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc,
  data = downSampledTrain,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 0,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_lasso_downsample


#### Ridge #####
#Down-sampling
set.seed(10101)
upSampledTrain <- upSample(x = training,
                           y = training$Pobre,
                           yname = "Pobre")

dim(training)
dim(upSampledTrain)
table(upSampledTrain$Pobre)

set.seed(10101)
logit_ridge_upsample <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc,
  data = upSampledTrain,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_ridge_upsample

#Down-sampling
set.seed(10101)
downSampledTrain <- downSample(x = training,
                               y = training$Pobre,
                               yname = "Default")
dim(training)
dim(downSampledTrain)
table(downSampledTrain$Pobre)

set.seed(10101)
logit_ridge_downsample <- train(
  Pobre~ Nper + Sexo + Jefe + Vivienda + Edad + Edad2 + Esc,
  data = downSampledTrain,
  method = "glmnet",
  trControl = ctrl,
  family = "binomial",
  metric = "ROC",
  tuneGrid = expand.grid(alpha = 1,lambda=lambda_grid),
  preProcess = c("center", "scale")
)

logit_ridge_downsample


#Testeando
testResults <- data.frame(Pobre = testing$Pobre)

testResults$logit<- predict(logit_caret,
                            newdata = testing,
                            type = "prob")[,1]

testResults$lasso<- predict(logit_lasso_roc,
                            newdata = testing,
                            type = "prob")[,1]

testResults$lasso_thresh<- predict(logit_lasso_roc,
                                   newdata = testing,
                                   type = "prob")[,1]

testResults$lasso_upsample<- predict(logit_lasso_upsample,
                                     newdata = testing,
                                     type = "prob")[,1]

testResults$logit_lasso_downsample<- predict(logit_lasso_downsample,
                                               newdata = testing,
                                               type = "prob")[,1]

testResults$ridge<- predict(logit_ridge_roc,
                            newdata = testing,
                            type = "prob")[,1]

testResults$ridge_thresh<- predict(logit_ridge_roc,
                                   newdata = testing,
                                   type = "prob")[,1]

testResults$ridge_upsample<- predict(logit_ridge_upsample,
                                     newdata = testing,
                                     type = "prob")[,1]

testResults$logit_ridge_downsample<- predict(logit_ridge_downsample,
                                             newdata = testing,
                                             type = "prob")[,1]

testResults<-testResults %>%
  mutate(logit=ifelse(logit>0.5,"Si","No"),
         lasso=ifelse(lasso>0.5,"Si","No"),
         ridge=ifelse(ridge>0.5,"Si","No"),
         lasso_thresh=ifelse(lasso_thresh>rfThresh$threshold,"Si","No"),
         lasso_upsample=ifelse(lasso_upsample>0.5,"Si","No"),
         logit_lasso_downsample=ifelse(logit_lasso_downsample>0.5,"Si","No"),
         ridge_thresh=ifelse(ridge_thresh>rfThresh2$threshold,"Si","No"),
         ridge_upsample=ifelse(ridge_upsample>0.5,"Si","No"),
         logit_ridge_downsample=ifelse(logit_ridge_downsample>0.5,"Si","No"),
  )

#Nos quedamos con Logit Lasso
with(testResults,table(Pobre,logit))
with(testResults,table(Pobre,lasso))
with(testResults,table(Pobre,lasso_thresh))
with(testResults,table(Pobre,lasso_upsample))
with(testResults,table(Pobre,logit_lasso_downsample))
with(testResults,table(Pobre,ridge))
with(testResults,table(Pobre,ridge_thresh))
with(testResults,table(Pobre,ridge_upsample))
with(testResults,table(Pobre,logit_ridge_downsample))

#Enviando la predicción al test

#Generación de variables
TF$Edad <- TF$P6040
TF$Edad2 <- TF$P6040^2
TF$Sexo <- recode(TF$P6020, "1=1; 2=0")
TF$Jefe <- ifelse(TF$P6050==1,1,0)
TF$Vivienda <- ifelse(TF$P5090==1 | TF$P5090==2,1,0) 
#Años de educación
TF$Esc <- ifelse(TF$P6210==1 | TF$P6210==2,0,NA)
TF$Esc <- ifelse(TF$P6210==3,TF$P6210s1,TF$Esc)
TF$Esc <- ifelse(TF$P6210==4,TF$P6210s1,TF$Esc)
TF$Esc <- ifelse(TF$P6210==4 & TF$P6210s1==0,5,TF$Esc)
TF$Esc <- ifelse(TF$P6210==5,TF$P6210s1,TF$Esc)
TF$Esc <- ifelse(TF$P6210==6 & TF$P6210s1==0,11,TF$Esc)
TF$Esc <- ifelse(TF$P6210==6,TF$P6210s1+11,TF$Esc)
TF$Esc[is.na(TF$Esc)] = 0

#Habitaciones
TF$P500 <- ifelse(TF$P500==98,0,TF$P500)
TF$Habit <- TF$P500

#Categorizando
TF$Sexo <- factor(TF$Sexo, labels = c("No", "Si"))
TF$Jefe <- factor(TF$Jefe, labels = c("No", "Si"))
TF$Vivienda <- factor(TF$Vivienda, labels = c("No", "Si"))

TF <- subset(TF, select = c(id,Nper,Sexo,Jefe,Vivienda,Edad,Edad2,Esc,Habit))

#Predicción
TF$Pobreza <- predict(logit_ridge_roc, newdata =TF, type="prob")[,1]

#Regla
TF$Pob_Pred <- ifelse(TF$Pobreza>=rfThresh2$threshold,1,0)


BD_Pob <- subset(TF, Jefe=="Si")
BD_Pob <- subset(BD_Pob, select = c(id,Pob_Pred))

saveRDS(BD_Pob, file = "D:/NUEVO D/UNIANDES/BIG DATA/Problem Set 2/Clasificacion.RDS")
