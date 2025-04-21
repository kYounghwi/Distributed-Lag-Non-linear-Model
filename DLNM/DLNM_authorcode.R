################################################################################
# EXAMPLE OF PREDICTIONS USING DLNMS
################################################################################

# load package
library(dlnm)

# create the crossbasis object for temperature
varknots <- equalknots(chicagoNMMAPS$temp,fun="bs",df=5,degree=2)
lagknots <- logknots(30, 3)
cb <- crossbasis(chicagoNMMAPS$temp, lag=30, argvar=list(fun="bs",
  knots=varknots), arglag=list(knots=lagknots))

# run the model with other predictors
library(splines)
model <- glm(death ~  cb + ns(time, 7*14) + o3 + dow, family=quasipoisson(),
  chicagoNMMAPS)

# extract the attributes of the cross-basis (to replicate the transformation)
attrcb <- attributes(cb)

################################################################################
# prediction of deaths in the first year

# select data for the first year
datafy <- subset(chicagoNMMAPS, year==1987)

# create new cross-basis for predictions
# nb: prediction for the first year using the observed variables
# nb: functions must be defined exactly the same, better to use saved attributes
cb <- do.call(crossbasis,list(x=datafy$temp,lag=attrcb$lag,argvar=attrcb$argvar,
  arglag=attrcb$arglag))

# predict and plot
pred <- predict(model,datafy,type="response")
plot(datafy$date,pred)

################################################################################
# prediction for a given temperature history (cumulated)

# create new cross-basis for predictions
# nb: history here as temperature in range 20-30C along the lag period
# other variables as the first day of the series
# nb: functions must be defined exactly the same, better to use saved attributes
hist <- t(runif(30+1, 20, 30))
cb <- do.call(crossbasis,list(x=hist,lag=attrcb$lag,argvar=attrcb$argvar,
  arglag=attrcb$arglag))
newdata <- chicagoNMMAPS[1,]

# predict
pred <- predict(model,newdata,type="response")
pred

################################################################################
# prediction for a given temperature (cumulated) 

# create new cross-basis for predictions
# nb: history here as fixed temperature along the lag period
# other variables fixed
# nb: functions must be defined exactly the same, better to use saved attributes
hist <- matrix(-20:30, nrow=length(-20:30), ncol=30+1)
cb <- do.call(crossbasis,list(x=hist,lag=attrcb$lag,argvar=attrcb$argvar,
  arglag=attrcb$arglag))
newdata <- data.frame(time=rep(1,nrow(hist)),o3=rep(20,nrow(hist)),
  dow=rep("Friday",nrow(hist)))

# predict and plot
pred <- predict(model,newdata,type="response")
plot(-20:30,pred,ylab="Deaths",xlab="temperature")

################################################################################
# other example

library(dlnm)

# DEFINE THE CROSS-BASIS, RUN THE MODEL, PREDICT, AND PLOT
cb <- crossbasis(chicagoNMMAPS$temp, lag=3, argvar=list(fun="ns", df=3),
  arglag=list(knots=1))
mod <- glm(death ~  cb + dow, family=quasipoisson(), chicagoNMMAPS)
cp <- crosspred(cb, mod, cen=21)
plot(cp, "overall")

# DEFINE AN EXPOSURE HISTORY OF FOUR DAYS IN A ROW OF 30C
hist <- rep(30, 4)

# RECOMPUTE THE CROSS-BASIS TERMS USING THE SAME PARAMETRISATION
cbpred <- crossbasis(t(hist), lag=3, argvar=attr(cb, "argvar"), 
  arglag=attr(cb, "arglag"))

# PREDICT THE DEATHS IN A WEDNESDAY WITH SUCH EXPOSURE
predict(mod, newdata=list(cb=cbpred, dow="Wednesday"), type="response")


#
