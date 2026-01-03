setwd("~/Dropbox/software/GAMER/examples/")

rm(list=ls())

library(SeqArray)

seqVCF2GDS("example.vcf.gz", "example.gds")
genofile <- seqOpen("example.gds")

gt <- seqGetData(genofile, "$dosage")
seqClose(genofile)
dim(gt)
X <- 2 - gt

set.seed(1)
C <- matrix(runif(1135*3,-1,1),nrow=1135,ncol=3)

set.seed(1)
alpha <- round(rnorm(3,mean=0,sd=1),2)

write.table(C,"C.txt",append = F,quote = F,sep = "\t",row.names = F,col.names = F)

circle_prod <- function(X,beta){
  index <- which(beta!=0)
  r <- dim(X)[1]
  y <- c()
  for (i in 1:r){
    x_beta <- X[i,index] * beta[index] + 1
    y <- c(y,prod(x_beta))
  }
  return(y)
}


N = 8
h2 = 0.7
sigma_1 = 0.8

i=1

## simulate random effects
beta <- rep(0,ncol(X))
set.seed(i)
beta_non0_index <- sort(sample(ncol(X),N))
set.seed(i)
beta[beta_non0_index] <- abs(rnorm(N,mean=0,sd=sigma_1))

Xb_multi <- circle_prod(as.matrix(X),beta)
sigma_e_2 <- (1-h2)*var(Xb_multi) / h2
set.seed(i)
e <- rnorm(nrow(X),mean = 0,sd = sqrt(sigma_e_2))
y_multi <- as.matrix(C)%*%alpha + Xb_multi + e

name = paste("example_multi",N,h2,i,sep = "_")

write.table(y_multi,file = paste0(name,".txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
write.table(beta_non0_index,file = paste0(name,"_true_non0_index.txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
write.table(beta,file = paste0(name,"_true_beta.txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)

test_beta <- read.delim("new_test_multiplicative_beta.txt",header=T)
true_beta <- read.delim("example_multi_8_0.7_1_true_beta.txt",header=F)
true_index <- read.delim("example_multi_8_0.7_1_true_non0_index.txt",header=F)
trace <- read.delim("example_multi_8_0.7_3_additive_model_trace.txt")

plot(test_beta$beta_mean[true_index$V1],true_beta$V1[true_index$V1])
abline(a=0,b=1)



## additive

N =50
h2 = 0.7
sigma_1 = 1

i=10

## random effects
beta <- rep(0,ncol(X))
set.seed(i)
beta_non0_index <- sort(sample(ncol(X),N))
set.seed(i)
beta[beta_non0_index] <- rnorm(N,mean=0,sd=sigma_1)

Xb_additive <- as.matrix(X) %*% beta
sigma_e_2 <- (1-h2)*var(Xb_additive) / h2
set.seed(i)
e <- rnorm(nrow(X),mean = 0,sd = sqrt(sigma_e_2))
y_additive <- as.matrix(C)%*%alpha + Xb_additive + e

name = paste("example_additive",N,h2,i,sep = "_")

write.table(y_additive,file = paste0(name,".txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
write.table(beta_non0_index,file = paste0(name,"_true_non0_index.txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)
write.table(beta,file = paste0(name,"_true_beta.txt"),append = F,quote = F,sep = "\t",row.names = F,col.names = F)


test_beta <- read.delim("new_test_additive_beta.txt",header=T)
test_alpha <- read.delim("new_test_additive_alpha.txt",header=F)
true_beta <- read.delim("example_additive_50_0.7_10_true_beta.txt",header=F)
true_index <- read.delim("example_additive_50_0.7_10_true_non0_index.txt",header=F)

plot(test_beta$beta_mean[true_index$V1],true_beta$V1[true_index$V1])
abline(a=0,b=1)

test_beta <- read.delim("example_additive_50_0.7_10_multiplicative_beta.txt",header=T)
test_alpha <- read.delim("example_additive_50_0.7_10_multiplicative_alpha.txt",header=F)
true_beta <- read.delim("example_additive_50_0.7_10_true_beta.txt",header=F)
true_index <- read.delim("example_additive_50_0.7_10_true_non0_index.txt",header=F)

plot(test_beta$beta_mean[true_index$V1],true_beta$V1[true_index$V1])
abline(a=0,b=1)

