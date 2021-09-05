# PART 1
############################### SETTING UP THE ENVIRNMENT #######################################
library(ggplot2)
library(mltools)
library(data.table)
library(corrplot)
library(caTools)
library(caret)
library(car)
library(leaps)
library(dplyr)
library(glmnet)
library(ggfortify)
library(proxy)
library(pls)
############################### DATASET INSPECTION ###############################################
df <- data.frame(read.csv("/Users/andreafrattini/Desktop/University/Statistical Learning/EXAM/cal-housing.csv"))
summary(df)
df <- na.omit(df)
apply(df,2,function(x) sum(is.na(x))) #double check

################################## SCALING THE DATASET #################################################
# numerical variables
pre_proc <- preProcess(df[,c(1:9)], method=c("center", "scale"))
df1 <- predict(pre_proc, df[,c(1:9)])
# categorical:
df1$ocean_proximity <- df$ocean_proximity
df1 <- one_hot(as.data.table(df1))
summary(df1)
df1$`ocean_proximity_ISLAND` <- NULL # since we have only 5 observations

################################# CLEANING UP THE DATASET ##############################################

Boxplot(df1) # As expected

# total_rooms
Q <- quantile(df1$total_rooms, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(df1$total_rooms)
up <-  Q[2]+1.5*iqr # Upper Range  
low<- Q[1]-1.5*iqr # Lower Range
df1 <- df1[which(df1$total_rooms > (Q[1] - 1.5*iqr) & df1$total_rooms < (Q[2]+1.5*iqr)),]

# total_bedrooms
Q <- quantile(df1$total_bedrooms, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(df1$total_bedrooms)
up <-  Q[2]+1.5*iqr  
low<- Q[1]-1.5*iqr
df1 <- df1[which(df1$total_bedrooms > (Q[1] - 1.5*iqr) & df1$total_bedrooms < (Q[2]+1.5*iqr)),]

# population
Q <- quantile(df1$population, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(df1$population)
up <-  Q[2]+1.5*iqr   
low<- Q[1]-1.5*iqr 
df1 <- df1[which(df1$population > (Q[1] - 1.5*iqr) & df1$population < (Q[2]+1.5*iqr)),]

# households
Q <- quantile(df1$households, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(df1$households)
up <-  Q[2]+1.5*iqr   
low<- Q[1]-1.5*iqr 
df1 <- df1[which(df1$households > (Q[1] - 1.5*iqr) & df1$households < (Q[2]+1.5*iqr)),]

# median_income
Q <- quantile(df1$median_income, probs=c(.25, .75), na.rm = FALSE)
iqr <- IQR(df1$median_income)
up <-  Q[2]+1.5*iqr   
low<- Q[1]-1.5*iqr
df1 <- df1[which(df1$median_income > (Q[1] - 1.5*iqr) & df1$median_income < (Q[2]+1.5*iqr)),]

Boxplot(df1)

###################################### COLLINEARITY ###############################################à
cor <- cor(df1)
corrplot(cor, method = "square", tl.cex = 0.6, title = "Correlation of Variables", tl.col = "black", mar=c(0,0,1,0))

################################ FORWARD STEPWISE WITH CROSS-VALIDATION ##################################################
forward <- regsubsets(median_house_value~., data=df1, method="forward", nvmax = 12)
summary(forward)
plot(forward,scale="Cp") # I expect to have 12 variables in the model

predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  mat[,names(coefi)]%*%coefi
}

# k-fold stepwise 
cv.errors <- matrix(NA,10,12)
for(k in 1:10){
  set.seed(13*k*k+24)
  train <- sample_frac(df1, 0.7)
  sid <- as.numeric(rownames(train)) # because rownames() returns character
  test <- df1[-sid,]
  best.fit=regsubsets(median_house_value~., data = train, method="forward", nvmax = 12)
  for(i in 1:12){
    pred <- predict(best.fit, test, id=i)
    cv.errors[k,i]= mean((test$median_house_value-pred)^2)
  }
}
rmse.cv=sqrt(apply(cv.errors,2,mean))
plot(rmse.cv,pch=19,type="b")

min_val <- function(mse){
  val <- min(mse)
  index <- which.min(mse) 
  return(list( "Value" = val, "Index" = index))
}
mse_forward <- min_val(rmse.cv)
mse_forward
################################ RIDGE REGRESSION WITH CROSS VALIDATION ##################################################
x <- model.matrix(median_house_value~. , data = df1)
y <- df1$median_house_value
fit_ridge <- glmnet(x, y, alpha=0) 
plot(fit_ridge, xvar="lambda", label=TRUE)
cv_ridge <- cv.glmnet(x, y ,alpha=0) 
plot(cv_ridge) 

min_val <- function(cv_ridge){
  lambda_min_mse <- cv_ridge$lambda.min
  lambda_min_var <- cv_ridge$lambda.1se
  i <- which(cv_ridge$lambda == cv_ridge$lambda.1se)
  mse.min <- cv_ridge$cvm[i]
  return(list( "Value of log(lambda) providing minimum MSE" = log(lambda_min_mse),
               "Value of log(lambda) providing minimum variance" = log(lambda_min_var),
               "MSE associated to the lambda which provides the lowest variance" = mse.min))
}
mse_ridge12 <- min_val(cv_ridge)
mse_ridge12
coef(cv_ridge)

################################### LASSO MODEL ########################################################
lasso <- glmnet(x, y)
plot(lasso, xvar="lambda", label=T)
cv_lasso <- cv.glmnet(x, y)
plot(cv_lasso)
min_val <- function(cv_lasso){
  lambda_min_mse <- cv_lasso$lambda.min
  lambda_min_var <- cv_lasso$lambda.1se
  i <- which(cv_lasso$lambda == cv_lasso$lambda.1se)
  mse.min <- cv_lasso$cvm[i]
  return(list( "Value of log(lambda) providing minimum MSE" = log(lambda_min_mse),
               "Value of log(lambda) providing minimum variance" = log(lambda_min_var),
               "MSE associated to the lambda which provides the lowest variance" = mse.min))
}
mse_lasso <- min_val(cv_lasso)
mse_lasso
coef(cv_lasso)

################################ PARTIAL LEAST SQUARES REGRESSION (PLSR) ##########################################
set.seed(123)
pls.fit1 <- plsr(median_house_value~., data=train, scale=TRUE, validation="CV")
summary(pls.fit1)
validationplot(pls.fit1, val.type="MSEP", legendpos = "topright", main="PLSR Test", xlab = "Number of components")
pls.pred <- predict(pls.fit1, test, ncomp=6)
mse_pls <- mean((pls.pred - y[nrow(test)])^2)
mse_pls
################################ COMPARISON BETWEEN THE MODELS ###########################################
df2 <- data.frame(stringsAsFactors = FALSE,
                  MSE = c(mse_forward$Value,
                          mse_ridge12$`MSE associated to the lambda which provides the lowest variance`,
                          mse_lasso$`MSE associated to the lambda which provides the lowest variance`,
                          mse_pls),
                  Models = c("MSE_forward","MSE_ridge12", "MSE_lasso", "MSE_pls"))
ggplot(df2, aes(Models,MSE))+geom_point(aes(size = 0.5), show.legend = F)

# PART 2
############################## PRINCIPAL COMPONENT ANALYSIS ###############################################

df3 <- df1
df3$median_house_value <- NULL 
str(df3) # the binary variables are int, but PCA works well only with num variables
df3$`ocean_proximity_<1H OCEAN` <- as.numeric(df3$`ocean_proximity_<1H OCEAN`)
df3$ocean_proximity_INLAND <- as.numeric(df3$ocean_proximity_INLAND)
df3$`ocean_proximity_NEAR BAY` <- as.numeric(df3$`ocean_proximity_NEAR BAY`)
df3$`ocean_proximity_NEAR OCEAN` <- as.numeric(df3$`ocean_proximity_NEAR OCEAN`)
str(df3)

prin_comp <- prcomp(df3)
prin_comp$center 
prin_comp$rotation[,1:4]

biplot(prin_comp, scale = 0, main = "Principal Component Analysis")

std_dev <- prin_comp$sdev 
# standard deviations of each principal component
pr_var <- std_dev^2 
# variance of each principal component
pr_var[1:10] 

prop_varex <- pr_var/sum(pr_var)
prop_varex[1:10]

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#################################### K-MEANS CLUSTERING ###################################################

wssplot <- function(df, nc=15, seed=123){
  wss <- (nrow(df)-1)*sum(apply(df, 2, var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(df, centers = i)$withinss)}
  plot(1:nc, wss, type = "b", xlab = "Number of clusters", ylab = "Within groups sum of squares")
}
wssplot(df1)

KM <- kmeans(df1,2)

autoplot(KM,df1,frame=TRUE)
# cluster centers
KM$centers
