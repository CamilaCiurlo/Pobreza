

setwd("D:/NUEVO D/UNIANDES/BIG DATA/Problem Set 2")
parte_1<-readRDS("Clasificacion.Rds")
parte_2<-readRDS("Pobreza.Rds")

Base_pre <- as.data.frame(left_join(x=parte_1 , y=parte_2, by=c("id")))

Base_pre$Pobre_ingreso <- Base_pre$pobre
Base_pre$Pobre_class <- Base_pre$Pob_Pred

Ing_Class <- subset(Base_pre, select = c(id,Pobre_ingreso,Pobre_class))


write.csv(Ing_Class, "Predictions_Valcarcel_Ciurlo_c8_r8.csv")