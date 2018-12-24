library(tidyverse)
library(plyr)

skellam.likelihood <- function(row, params) {
    mu <- params[1]
    h <- params[2]
    
    #params[3] <- -sum(params[4:22])
    #params[23] <- -sum(params[24:42])
    
    z <- row$GDiff
    
    lambda1 <- exp(mu + h + params[2 + row$HomeTeam] + params[22 + row$AwayTeam])
    lambda2 <- exp(mu + params[2 + row$AwayTeam] + params[22 + row$HomeTeam])
    
    bessel <- besselI(2 * sqrt(lambda1 * lambda2), abs(z))
    return(log(exp(-(lambda1 + lambda2)) * (lambda1/lambda2)^(z/2) * bessel))
}

neg.log.likelihood <- function(params) {
    sum <- 0
    for(i in 1:nrow(current)) {
        sum <- sum + skellam.likelihood(current[i,], params=params)
    }
    return(sum)
}

skellam <- function(z,lambda1,lambda2) {
    bessel <- besselI(2 * sqrt(lambda1 * lambda2), abs(z))
    return(exp(-(lambda1 + lambda2)) * (lambda1/lambda2)^(z/2) * bessel)
}

match_results <- read_csv("epl_results.csv")
match_results <- match_results %>% mutate(GDiff = HG-AG)

current <- match_results %>% filter(Season == "2015-16") %>% select(HomeTeam, AwayTeam, HG, AG, GDiff)

home_team_mapped <- as.numeric(mapvalues(current$HomeTeam, from = unique(current$HomeTeam), to = seq(1:20)))
key_vals <- home_team_mapped[1:20]
names(key_vals) <- current$HomeTeam[1:20]

for(i in 1:nrow(current)) {
    current$AwayTeam[i] <- which(names(key_vals) == current$AwayTeam[i])
}
current$HomeTeam <- home_team_mapped
current$AwayTeam <- as.numeric(current$AwayTeam)

#result <- optim(par=rnorm(42,0,0.5), fn=NLL, data=current, method = "L-BFGS-B", control=list(trace=0))
#y <- result$par[23:42]; names(y) <- names(key_vals)

library(maxLik)
A <- matrix(0, 2, 42)
B <- matrix(0, 2, 1)
A[1,3:22] <- 1
A[2,23:42] <- 1

res <- maxNM(neg.log.likelihood, start = rnorm(42,0.1,0.1), constraints=list(eqA = A, eqB = B))
y <- res$estimate[3:22]; names(y) <- names(key_vals)

sum <- 0
for(j in 1:10) {
    sum <- sum + skellam(j, lambda1, lambda2)
}
