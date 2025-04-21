library(dlnm)

head(nested)
drug

############################################################

Qdrug <- as.matrix(drug[,rep(7:4, each=7)])
colnames(Qdrug) <- paste("lag", 0:27, sep="")
Qdrug[1:3,1:14]


############################################################

Qnest <- t(apply(nested, 1, function(sub) exphist(rep(c(0,0,0,sub[5:14]),
                                                        each=5), sub["age"], lag=c(3,40))))
colnames(Qnest) <- paste("lag", 3:40, sep="")
Qnest[1:3,1:11]


############################################################

cbdrug <- crossbasis(Qdrug, lag=27, argvar=list("lin"),
                     arglag=list(fun="ns",knots=c(9,18)))

summary(cbdrug)

mdrug <- lm(out~cbdrug+sex, drug)
pdrug <- crosspred(cbdrug, mdrug, at=0:20*5)

with(pdrug,cbind(allfit,alllow,allhigh)["50",])

pdrug$matfit["20","lag3"]

plot(pdrug, zlab="Effect", xlab="Dose", ylab="Lag (days)")  # 노출-지연 효과
plot(pdrug, var=60, ylab="Effect at dose 60", xlab="Lag (days)", ylim=c(-1,5))  # 특정 노출에서의 효과
plot(pdrug, lag=10, ylab="Effect at lag 10", xlab="Dose", ylim=c(-1,5))         # 특정 시점에서의 효과


############################################################

cbnest <- crossbasis(Qnest, lag=c(3,40), argvar=list("bs",degree=2,df=3),
                     arglag=list(fun="ns",knots=c(10,30),intercept=F))

library(survival)
mnest <- clogit(case~cbnest+strata(riskset), nested)
pnest <- crosspred(cbnest, mnest, cen=0, at=0:20*5)

plot(pnest, zlab="OR", xlab="Exposure", ylab="Lag (years)")
plot(pnest, var=50, ylab="OR for exposure 50", xlab="Lag (years)", xlim=c(0,40))
plot(pnest, lag=5, ylab="OR at lag 5", xlab="Exposure", ylim=c(0.95,1.15))

expnested <- rep(c(10,0,13), c(5,5,10))
hist <- exphist(expnested, time=length(expnested), lag=c(3,40))

pnesthist <- crosspred(cbnest, mnest, cen=0, at=hist)
with(pnesthist, c(allRRfit,allRRlow,allRRhigh))












