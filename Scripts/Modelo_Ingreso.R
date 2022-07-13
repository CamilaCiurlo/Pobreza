##################################################
####### ALUMNA: CAMILA CIURCLO ARAGON#############
#######ALUMNO: PAOLO VALCARCEL PINEDA ############
##################################################
###### CURSO: BIG DATA Y MACHINE LEARNING ########
############ PROBLEM SET: POBREZA ################
##################################################

rm(list = ls())

#######----------Cargar bases de datos----------####### 

setwd("C:/Users/Camila Ciurlo/Desktop/MeCA/Big data/Problem set 2")
TH<-readRDS("test_hogares.Rds")
TP<-readRDS("test_personas.Rds")
TRH<-readRDS("train_hogares.Rds")
TRP<-readRDS("train_personas.Rds")

#######----------Cargar librerías----------####### 

library(pacman)
p_load(rio, tidyverse, skimr)
library(ggplot2)
library(ggplot2)
library(patchwork)
library(caret)
library(caTools)
library(RColorBrewer)
library(stargazer)
library(dplyr)
library("nycflights13")
library(lubridate)
library(tseries)
library(car)
library(foreign)
library(timsac)
library(lmtest)
library(mFilter)
library(nlme)
library(lmtest)
library(broom)
library(kableExtra)
library(knitr)
library(MASS)
library(parallel)
library(mlogit)
library(tidyr)
library(forecast)
library(stats)
library(quantmod)
library(foreach)
library(ISLR)
library(caret)
#install.packages("Matrix")
library(glmnet)
pacman::p_load(gridExtra, scales, ggcorrplot, e1071)


#######----------Merge bases de datos----------####### 

TRF <- as.data.frame(left_join(x=TRP , y=TRH, by=c("id")))#Train personas y train hogares 
TF <- as.data.frame(left_join(x=TP , y=TH, by=c("id")))#Test personas y Test hogares 


#######----------Subset bases de dato----------####### 

#Se trabaj con las variables comunes en Train y Test que puedan explicar pobreza 

Train<-TRF [ , c("id","Dominio.x","P6020","Ingpcug", "Ingtotugarr", "Lp",  
                 "Npersug","P6210","P6040","P6050","P6210s1", "P5090","P5000")] #Subset Train
Test <-TF [ , c("id","Dominio.x","P6020", "Lp",
                 "Npersug","P6210","P6040","P6050","P6210s1","P5090","P5000")]#Subset Test 
#Removemos objetos 
rm(TF, TH, TP, TRF, TRH, TRP)

#######----------Creación de variables----------####### 

#Escolaridad

Train$Esc <- ifelse(Train$P6210==1 | Train$P6210==2,0,NA)
Train$Esc <- ifelse(Train$P6210==3,Train$P6210s1,Train$Esc)
Train$Esc <- ifelse(Train$P6210==4,Train$P6210s1,Train$Esc)
Train$Esc <- ifelse(Train$P6210==4 & Train$P6210s1==0,5,Train$Esc)
Train$Esc <- ifelse(Train$P6210==5,Train$P6210s1,Train$Esc)
Train$Esc <- ifelse(Train$P6210==6 & Train$P6210s1==0,11,Train$Esc)
Train$Esc <- ifelse(Train$P6210==6,Train$P6210s1+11,Train$Esc)
Train$Esc[is.na(Train$Esc)] = 0

Test$Esc <- ifelse(Test$P6210==1 | Test$P6210==2,0,NA)
Test$Esc <- ifelse(Test$P6210==3,Test$P6210s1,Test$Esc)
Test$Esc <- ifelse(Test$P6210==4,Test$P6210s1,Test$Esc)
Test$Esc <- ifelse(Test$P6210==4 & Test$P6210s1==0,5,Test$Esc)
Test$Esc <- ifelse(Test$P6210==5,Test$P6210s1,Test$Esc)
Test$Esc <- ifelse(Test$P6210==6 & Test$P6210s1==0,11,Test$Esc)
Test$Esc <- ifelse(Test$P6210==6,Test$P6210s1+11,Test$Esc)
Test$Esc[is.na(Test$Esc)] = 0

#Ingreso per cápita del hogar e ingreso total del hogar en Test 

Test$Ingpcug <-0
Test$Ingtotugarr <- 0

#Edad, edad2, jefe, Sexo, vivienda 

Train$Edad <- Train$P6040
Train$Edad2 <- Train$P6040^2
Train$Sexo <- ifelse(Train$P6020==1,1,0)
Train$Jefe <- ifelse(Train$P6050==1,1,0)
Train$Vivienda <- ifelse(Train$P5090==1 | Train$P5090==2,1,0)
Train$Habit <- Train$P5000

Test$Edad <- Test$P6040
Test$Edad2 <- Test$P6040^2
Test$Sexo <- ifelse(Test$P6020==1,1,0)
Test$Jefe <- ifelse(Test$P6050==1,1,0)
Test$Vivienda <- ifelse(Test$P5090==1 | Test$P5090==2,1,0)
Test$Habit <- Test$P5000


#Se eliminan columnas: limpieza de ambiente  

Train$P6040 <- NULL 
Train$P6020 <- NULL
Train$P6050 <- NULL 
Train$P5090 <- NULL
Train$P6210 <- NULL
Train$P6210s1 <-NULL
Train$P5140 <- NULL
Train$P5000 <-NULL

Test$P6040 <- NULL 
Test$P6020 <- NULL
Test$P6050 <- NULL 
Test$P5090 <- NULL
Test$P6210 <- NULL
Test$P6210s1 <- NULL
Test$P5140 <- NULL
Test$P5000 <-NULL

#######----------Algunas estadísticas descriptivas----------####### 

#Promedio del ingreso por sexo

ingsexo = aggregate(Ingpcug ~ Sexo, data = Train, FUN = mean)
ingsexo $Sexo <- factor (ingsexo$Sexo, levels=c("0", "1"), labels = c("Mujer", "Hombre"))
colnames(ingsexo) <- c("Sexo","Promedio del ingreso")
export(ingsexo, "ingsexo.xlsx")


#Promedio del ingreso por tenencia de vivienda 

Vivienda1 = aggregate(Ingpcug ~ Vivienda, data = Train, FUN = mean)
Vivienda1 $Vivienda <- factor (Vivienda1$Vivienda, levels=c("0", "1"), labels = c("Sin vivienda", "Vivienda propia"))
colnames(Vivienda1) <- c("Vivienda","Promedio del ingreso")
export(Vivienda1, "Vivienda.xlsx")

#Senda de ingreso por escolaridad 

incomestudies = aggregate(ingtot ~ Educ, data = db, FUN = mean)
incomestudies $Educ <- factor (incomestudies$Educ, levels=c("1", "2", "3", "4", "5", "6"), 
                               labels = c("Ninguno", "Pre-escolar", "Primaria", "Secundaria", "Media", "Universitario/Superior"))
colnames(incomestudies) <- c("Grado de escolaridad","Promedio del ingreso")



#######----------Modelo 1: regresión lineal múltiple----------####### 

set.seed(10101) 
id_Train <- sample(1:nrow(Train), size=0.7*nrow(Train), replace = F)
x.Train <- Train[id_Train, ]
x.Test <- Train[-id_Train, ]

names(x.Train)
Modelo_1 = lm(Ingpcug ~ Esc + Sexo + Edad + Edad2 + Jefe + Npersug + Vivienda + Habit, data = x.Train)
summary(Modelo_1)

Modelo_1$coefficients%>%
  enframe (name="variable", value="Coeficiente")

#Predicción dentro y fuera de muestra 

y_hat_in<- predict(Modelo_1, x.Train)
y_hat_out<-predict(Modelo_1, x.Test)
y_real_in<-x.Train$Ingpcug
y_real_out<-x.Test$Ingpcug

#MSE

MSE_in_1 <- mean ((y_hat_in - y_real_in)^2)
MSE_out_1 <- mean ((y_hat_out - y_real_out)^2)

MSLE_in_1 <-log(MSE_in_1)
MSLE_out_1 <-log(MSE_out_1)

MSE_out_1/MSE_in_1

#######----------Modelo 2: regresión lineal múltiple con menos variables explicativas----------####### 

Modelo_2 = lm(Ingpcug ~Sexo + Edad + Edad2 + Esc, data = x.Train)
summary(Modelo_2)

Modelo_2$coefficients%>%
  enframe (name="variable", value="Coeficiente")

#Predicción dentro y fuera de muestra 

y_hat_in2<- predict(Modelo_2, x.Train)
y_hat_out2<-predict(Modelo_2, x.Test)
y_real_in2<-x.Train$Ingpcug
y_real_out2<-x.Test$Ingpcug

#MSE 

MSE_in_2 <- mean ((y_hat_in2 - y_real_in2)^2)
MSE_out_2 <- mean ((y_hat_out2 - y_real_out2)^2)

MSLE_in_2 <-log(MSE_in_2)
MSLE_out_2 <-log(MSE_out_2)

MSE_out_2/MSE_in_2


#######----------Modelo Ridge----------####### 

X_Train1<- select(x.Train, -Ingpcug, -Ingtotugarr, -id, -Dominio.x, -Lp)
X_Train <- model.matrix(~., data = X_Train1)
Y_Train<- x.Train$Ingpcug

X_Test1 <- select(x.Test, -Ingpcug, -Ingtotugarr, -id, -Dominio.x, -Lp)
X_Test <- model.matrix(~., data = X_Test1)
Y_Test<- x.Test$Ingpcug


ridge.model<-glmnet(x=X_Train, y=Y_Train, alpha=0,nlambda = 100,standardize = T)
summary(ridge.model)

#Lamdas y coficientes 
plot(ridge.model, xvar = "lambda",label=TRUE)
plot (ridge.model, xvar = "dev", label=TRUE)  

Coefridge <- coef(ridge.model)
print(Coefridge)
ridge.model$lambda

#Encontrar el mejor lamda: validación cruzada: K-fold

cv_error<-cv.glmnet(x=as.matrix(X_Train), y=Y_Train, alpha=0, nfolds=10,
                    type.measure = "mse",
                    standarize=TRUE,
                    nmlamda=100)
plot(cv_error)
mejor.lambda_min <- cv_error$lambda.min
mejor.lambda_min
log(mejor.lambda_min)


#Predecir valores de y con el mejor lamda 

y_predict_ridge <- predict(ridge.model, s=mejor.lambda_min, newx = X_Test)
length(y_predict_ridge)

#SSE y SST 
SST_ridge<- sum((Y_Test - mean(Y_Test))^2)
SSE_ridge<- sum((y_predict_ridge-Y_Test)^2)

rsquare_ridge <- 1- (SSE_ridge/SST_ridge)

#MSE y comparaciones 

MSE_ridge <-(sum((y_predict_ridge-Y_Test)^2)/length(y_predict_ridge))

MSLE_ridge <-log(MSE_ridge)

MSE_out_1
MSE_out_2

#######----------Modelo Lasso----------####### 

lasso.model<-glmnet(x=X_Train, y=Y_Train, alpha=1,nlambda = 100,standardize = T)
summary(lasso.model)

#Lamdas y coeficientes 
plot(lasso.model, xvar = "lambda",label=TRUE)
plot (lasso.model, xvar = "dev", label=TRUE) 

Coeflasso <- coef(lasso.model)
print(Coeflasso)
lasso.model$lambda

#Encontrar el mejor lamda 

cv_error_lasso <-cv.glmnet(x=as.matrix(X_Train), y=Y_Train, alpha=1,nfolds=10,
                    type.measure = "mse",
                    standarize=TRUE,
                    nmlamda=100)
plot(cv_error_lasso)
mejor.lambda_min_lasso <- cv_error_lasso$lambda.min
mejor.lambda_min_lasso
log(mejor.lambda_min_lasso)


#Predecir valores de y con el mejor lamda

y_predict_lasso <- predict(lasso.model, s=mejor.lambda_min_lasso, newx = X_Test)
length(y_predict_lasso)

#SSE y SST 
SST_lasso<- sum((Y_Test - mean(Y_Test))^2)
SSE_lasso<- sum((y_predict_lasso -Y_Test)^2)

rsquare_lasso <- 1 - (SSE_lasso/SST_lasso)

#MSE y comparaciones 

MSE_lasso <-(sum((y_predict_lasso-Y_Test)^2)/length(y_predict_lasso))

MSLE_lasso <-log(MSE_lasso)

MSE_lasso
MSE_ridge
MSE_out_1
MSE_out_2


#######----------Elastic Net----------####### 

Enet.model<-glmnet(x=X_Train, y=Y_Train, alpha=0.7,nlambda = 100,standardize = T)
summary(Enet.model)


#Lamdas y coeficientes 
plot(Enet.model, xvar = "lambda",label=TRUE)
plot (Enet.model, xvar = "dev", label=TRUE)  

CoefEnet <- coef(Enet.model)
print(CoefEnet)
Enet.model$lambda

#Encontrar el mejor lamda 

cv_error_Enet <-cv.glmnet(x=as.matrix(X_Train), y=Y_Train, alpha=0.7,nfolds=10,
                           type.measure = "mse",
                           standarize=TRUE,
                           nmlamda=100)
plot(cv_error_Enet)
mejor.lambda_min_Enet <- cv_error_Enet$lambda.min
mejor.lambda_min_Enet
log(mejor.lambda_min_Enet)


#Predecir valores de y con el mejor lamda 

y_predict_Enet <- predict(Enet.model, s=mejor.lambda_min_Enet, newx = X_Test)
length(y_predict_Enet)

#SSE y SST 
SST_Enet<- sum((Y_Test - mean(Y_Test))^2)
SSE_Enet<- sum((y_predict_Enet -Y_Test)^2)

rsquare_Enet <- 1- (SSE_Enet/SST_Enet)

#MSE y comparaciones 

MSE_Enet <-(sum((y_predict_Enet-Y_Test)^2)/length(y_predict_Enet))

MSLE_Enet <-log(MSE_Enet)

MSE_Enet
MSE_lasso
MSE_ridge
MSE_out_1
MSE_out_2

#######----------Predicción del ingreso en la base Test con el mejor modelo: Lasso----------#######  


#Se genera el vector de variables independientes en la base Train para correr reg Lasso

x <- model.matrix(Ingpcug~ Esc + Sexo + Edad + Edad2 + Jefe + Npersug + Vivienda + Habit, Train)
x <- model.matrix(Ingpcug~ Esc + Sexo + Edad + Edad2 + Jefe + Npersug + Vivienda + Habit, Train)[,-1]#se quita el intercepto

#Se genera el vector de la variable dependiente 

y=Train$Ingpcug

#Modelo lasso

Modelo_final<-glmnet(x, y, alpha=1,nlambda = 100,standardize = T)
summary(Modelo_final)

#Lamdas y coeficientes 
plot(Modelo_final, xvar = "lambda",label=TRUE)
plot (Modelo_final, xvar = "dev", label=TRUE)  

Coef_Final <- coef(Modelo_final)
print(Coef_Final)
Modelo_final$lambda

#Encontrar el mejor lamda 

cv_error_Final <-cv.glmnet(x, y, alpha=1,nfolds=10,
                          type.measure = "mse",
                          standarize=TRUE,
                          nmlamda=100)
plot(cv_error_Final)
mejor.lambda_Final <- cv_error_Final$lambda.min
mejor.lambda_Final
log(mejor.lambda_Final)

#Hacemos la predicción sobre el test

x.test1<-model.matrix(Ingpcug~ Esc + Sexo + Edad + Edad2 + Jefe + Npersug + Vivienda + Habit, Test)
x.test1<-model.matrix(Ingpcug~ Esc + Sexo + Edad + Edad2 + Jefe + Npersug + Vivienda + Habit, Test)[,-1]#se quita el intercept

coef(Modelo_final)[,which(Modelo_final$lambda==mejor.lambda_Final)]
pred <- predict(Modelo_final, s=mejor.lambda_Final,newx = x.test1)
pred

Test$pred<- pred 

#######----------Cñasificación de pobreza---------#######

IngHogar <- Test %>%
  group_by(id) %>%
  summarize(Ingreso = sum(pred), personas = max(Npersug), Lp=max(Lp))
IngHogar

IngHogar$pobre<- ifelse(IngHogar$Ingreso/IngHogar$personas<IngHogar$Lp,1,0)

table(IngHogar$pobre)
export(IngHogar, "Pobreza.Rds")
export(IngHogar, "Pobre_modelo_lasso.xlsx")

#############################################################
#############################################################