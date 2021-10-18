install.packages(c("ggplot2","dplyr", "zoo", "tidyverse", "xtsextra", "PortfolioAnalytics"))
install.packages("PortfolioAnalytics")
install.packages(c("ROI","ROI.plugin.glpk","ROI.plugin.quadprog", "ROI.plugin.alabama"))
install.packages("IntroCompFinR", repos="http://R-Forge.R-project.org")
install.packages("PerformanceAnalytics")
install.packages("tidyverse")
install.packages("colorspace")
install.packages("rlang")
install.packages("stringr")
library(rlang)
library(dplyr)
library(tidyverse)
library(xts)
library(xtsExtra)
library(data.table)
library(IntroCompFinR)
library(PortfolioAnalytics)
library(ROI.plugin.glpk)
library(ROI.plugin.quadprog)
library(ROI.plugin.alabama)
require(quantmod)
library(PerformanceAnalytics)
require(ggplot2)
library(stringr)


options(digits = 5)
# setting working directory to where script is saved
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# reading data and checking data
data <- read.csv("ps1_data.csv")
class(data)
str(data)
class(data$X)

data$X <- as.Date(data$X, format = "%Y-%m-%d") #format first column as date
names(data)[names(data)=="X"]<- "Date" #renaming column 1 to Date
class(data$Date)

# Exercise 1a

#discrete return

#xts1 <- xts(x=1:nrow(data), order.by = data$Date)
xtsall <- xts(data[,2:ncol(data)], order.by = data$Date)
discrete_return <- na.omit(Return.calculate(xtsall, method = "discrete"))

#log return
log_return <- na.omit(Return.calculate(xtsall, method = "log"))

#mean monthly return
discretemean <- lapply(discrete_return, FUN=mean)
logmean <- lapply(log_return, FUN=mean)
head(logmean)
#standard deviations of monhtly return
discreteSD <- lapply(discrete_return, FUN=sd)
log_SD <- lapply(log_return, FUN=sd)

#annualized mean return for each month
#given we have not worked in percentage before it is fairly straight forward
annualized_discrete_return <- discrete_return*12 #showing the monthly mean return in annualized format for each month
annualized_log_return <- log_return*12 #showing the monthly mean log return in annualized format

# Exercise 1b
difference_dislog <- discrete_return-log_return

maxdiff <- as.data.frame(matrix(NA, nrow = 1, ncol = 10))# creating data frame to save results
colnames(maxdiff) <- colnames(difference_dislog)
for(i in 1:10){ #creating for loop
   maxdiff[,i] <- max(difference_dislog[,i])
}

minmaxdiff <- which.min(maxdiff) #indicating the column where the maximal difference is the largest
maxmaxdiff <- which.max(maxdiff) #indicating the column where the maximal difference is the smallest
print(minmaxdiff)
print(maxmaxdiff)
print(maxdiff)

DB <- cbind(discrete_return$DEUTSCHE_BANK,log_return$DEUTSCHE_BANK)
colnames(DB) <- c("Disc_Return", "Log_Return")
head(DB)

EON <- cbind(discrete_return$E_ON,log_return$E_ON)
colnames(EON) <- c("Disc_Return", "Log_Return")
head(EON)



DBplot <- ggplot( data=DB, 
        mapping = aes(x=Log_Return,y=Disc_Return)) +
  ggtitle("DB Plot") +
  geom_line(col="red") +
  xlim(-0.5,0.5) +
  ylim(-0.5,0.5) 
  
  
EONplot <- ggplot( data=EON, 
        mapping = aes(x=Log_Return,y=Disc_Return)) +
  ggtitle("EON Plot") +
  geom_line(col="red") +
  xlim(-0.5,0.5) +
  ylim(-0.5,0.5) +
  theme(legend.position = "bottom")  


print(EONplot) 
dev.new()
print(DBplot)

#Comment: There is barely any difference noticable in these graphs, given using log returns have a normalizing effect
# What can be clearly seen in both plots is that the discrete return is always larger than the log returns and therefore a slightly upward curvature. 

#Exercise 1c
#Usually the discrete return is used for calculating the return of a portfolio (i.e. multiple assets) and when choosing the different weights of assets in a portfolio. 
# Log returns are used when returns are aggregated across time and when comparing investment horizons for the same asset.

# Exercise 1d

equalWeightMonthlyReturns <- 
  Return.portfolio(discrete_return,
                   weights = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1),
                   value=100,
                   verbose=TRUE)


equalWeightMonthlyReturns <- as.data.frame(equalWeightMonthlyReturns)
equalWeightMonthlyReturns <- equalWeightMonthlyReturns[1:nrow(equalWeightMonthlyReturns), 42:ncol(equalWeightMonthlyReturns)]
res <- sum(equalWeightMonthlyReturns[nrow(equalWeightMonthlyReturns),1:10])
res


#Exercise 2a

covs <- cov(discrete_return)

which(covs ==min(covs), arr.ind = TRUE)
print(covs)

