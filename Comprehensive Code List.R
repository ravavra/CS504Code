#Correlation Matrix and plotting Sample Code

correlatedData <-cor(datafraem[column], dataframe[column], method="pearson/spearman")  #this creates a correlations to plot in the next command 

Corrplot(correlatedData, method="number", tl.cex=.5, number.cex=.4 #plot the corrleated data with smaller numbers and axis labels ot fit on the screen 
         
         Scatterplot Sample Code - 
           
           options(scipen=50)    # this tells R that we are not using scientific notation, but integers  
         
         clswkrAndSalaryOUTLIERS<-ds_Outliers_ORGNAMES[c("A_CLSWKR", "PEARNVAL")] #uses those 2 pertinent columns from the dataframe 
         
         clswNames<-c("None", "Private", "FED GOV", "State GOV", "Local GOV", "Self-INC", "Self-NoINC", "No Pay", "NVR WRKD") ## uses the Class wof wokers from the dataset to be used ina future step as the X Axis 

plot(classWorkerandSalaryNOOUTLIERS, xlab="", ylab="Salary") # plots the scatterplot 

axis(1,at=0:8, labels=clswNames, las=2) # adds the Classes of workers as the X axis 

##Lasso Regression Code - R 

x <- read.csv('Final_Data_No_Outliers.csv') 

y <- x[,41] 

x <- x[,-41] 

nr <- nrow(x) 

trainingSize <- nr * .7 # 70% training size 

  

#sets a random seed 

set.seed(100) 

train <- sample(1:nr,trainingSize) 

  

#partition and label data 

x.train <- x[train,] 

y.train <- y[train] 

head(x.train) 

head(y.train) 

  

x.test <- x[-train,] 

y.test <- y[-train] 

head(x.test) 

  

#Show variable plot 

lasso.mod <- glmnet(x[train,],y[train], alpha=1) 

plot(lasso.mod,las=1) 

  

#show tuning parameter plot (lambda) 

cv.out <- cv.glmnet(data.matrix(x[train,]), y[train] ,alpha=1) 

plot(cv.out) 

title("Lasso Tuning and Variable Selection", line = 2.5) 

  

#lambda value for most powerful regression 

lambda.min <- cv.out$lambda.min 

#predict values using best lambda 

lasso.predBest <- predict(lasso.mod, s=lambda.min, newx=as.matrix(x.test)) 

  

#Display RMSE and r2 

RMSE(lasso.predBest,y.test) 

summary(lm(lasso.predBest~y.test)) 

 

##Random Forest Code - R 

 

data<-read.csv('Final_Data_With_Outliers.csv') 

 

#partition data 

set.seed(100) 

trainrows<-sample(nrow(data),nrow(data)*.8, replace = FALSE) 

traindata<-data[trainrows,] 

testdata<-data[-trainrows,] 

  

#run rf 

rf<-randomForest(x=traindata[,-41],y=traindata[,41],ntree=100,mtry=5,do.trace=1) 

varImpPlot(rf, main="Random Forest Top 10 Variables", n.var=10) 

  

#predict 

rfpredictions<-predict(rf, newdata = testdata) 

  

#calculate statistics 

RMSE(rfpredictions,testdata[,41]) 

summary(lm(testdata[,41]~rfpredictions)) 

 

 

##XGBoost - R 

 

#select training rows 80% training and 20% testing 

set.seed(100) 

trainrows<-sample(nrow(data),nrow(data)*.8) 

  

#pull out response variable 

labels<-data$PEARNVAL 

newdata.no.pearnval<-subset(data,select=-PEARNVAL) 

  

#build training data 

train<-as.matrix(newdata.no.pearnval[trainrows,]) 

train.labels<-as.matrix(labels[trainrows]) 

  

#build testing data 

test<-as.matrix(newdata.no.pearnval[-trainrows,]) 

test.labels<-as.matrix(labels[-trainrows]) 

  

#make data xgb objects 

xgtrain<-xgb.DMatrix(data=train, label=train.labels) 

xgtest<-xgb.DMatrix(data=test, label =test.labels) 

  

#regression on training data 

model<-xgboost(data=xgtrain,nround=20,verbose=2) 

  

#predict with test data - results etc.  

defaultpredictions<-predict(model,xgtest) 

defaultRMSE<-RMSE(x=defaultpredictions, ref=test.labels) 

defaultxglm<-lm(defaultpredictions~test.labels) 

defaultr2<-summary(defaultxglm)$r.squared 

  

#NEW TUNING TESTS 

  

#controls the tuning param search,  

#Here it is searching for nrounds and the learning rate 

tune_grid <- expand.grid(
  nrounds = c(1000,1250,1500,1750,2000),
  eta = c(0.015,0.02,0.025),
  max_depth = 8,
  gamma = 1.8356162155113733,
  colsample_bytree = 0.9748898924248475,
  min_child_weight = 0.0026129992902634088,
  subsample = .6010619837085674
)


  

#specifies some conditions for validation 

tune_control <- caret::trainControl( 

  method = "cv", # cross-validation 

  number = 3, # with n folds  

  #index = createFolds(tr_treated$Id_clean), # fix the folds 

  verboseIter = TRUE, # no training log 

  allowParallel = TRUE # FALSE for reproducible results  

) 

  

#this code executes the tune search - may take a long time depending on search 

xgb_tune <- caret::train( 

  x = train, 

  y = as.double(train.labels), 

  trControl = tune_control, 

  tuneGrid = tune_grid, 

  method = "xgbTree", 

  verbose = TRUE, 

  objective = "reg:squarederror" 

) 

  

#pull out best tuning params 

params<-as.list(xgb_tune$bestTune) 

  

#execute model with new params 

testmodel<-xgboost(data=xgtrain,verbose=2, params = params) 

  

#plot search w/helper function 

tuneplot <- function(x, probs = .9) { 

  ggplot(x) + 

    coord_cartesian(ylim = c(quantile(x$results$RMSE, probs = probs), min(x$results$RMSE))) + 

    theme_bw() 

} 

  

#plotting call 

tuneplot(xgb_tune) 

  

#predict with new tuning params 

predictions<-predict(testmodel,xgtest) 

newRMSE<-RMSE(x=predictions, ref=test.labels) 

  

xglm<-lm(predictions~test.labels) 

newr2<-summary(xglm)$r.squared 

  

data.table(newRMSE,defaultRMSE) 

data.table(newr2,defaultr2) 

#Graphing Predcitions vs. Actual
library(ggplot2)

plotme<-data.frame(predictions,test.labels)

error=abs(predictions-test.labels)


ggplot(plotme)+aes(x=test.labels,y=predictions, colour = error)+
  geom_point(alpha=.2, shape=16)+
  scale_colour_gradientn(colours=c('black','red'), name ='Error',
                         limits=c(0,20000),oob = scales::squish)+ 
  labs(x='Actual', y='Prediction', title='Actual vs. Predicted Income Error')+
  theme(plot.title = element_text(hjust = 0.5))

error2=(predictions-test.labels)

ggplot(as.data.frame(error2)) + aes(x=error2) + geom_histogram(binwidth=8000) + 
  xlim(c(-40000,40000)) + labs(title='Modeling Error Distribution')
