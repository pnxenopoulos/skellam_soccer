library(tidyverse)
library(plyr)

skellam.likelihood <- function(row, params) {
    mu <- params[1]
    h <- params[2]
    
    params[3] <- -sum(params[4:22])
    params[23] <- -sum(params[24:42])
    
    z <- row$GDiff
    
    lambda1 <- exp(mu + h + params[2 + row$HomeTeam] + params[22 + row$AwayTeam])
    lambda2 <- exp(mu + params[2 + row$AwayTeam] + params[22 + row$HomeTeam])
    
    bessel <- besselI(2 * sqrt(lambda1 * lambda2), abs(z))
    return(log(exp(-(lambda1 + lambda2)) * (lambda1/lambda2)^(z/2) * bessel))
}

NLL <- function(data, params) {
    sum <- 0
    for(i in 1:nrow(data)) {
        sum <- sum + skellam.likelihood(data[i,], params=params)
    }
    return(-sum)
}

match_results <- read_csv("epl_results.csv")
match_results <- match_results %>% mutate(GDiff = HG-AG)

current <- match_results %>% filter(Season == "2015-16") %>% select(HomeTeam, AwayTeam, HG, AG, GDiff)

home_params <- paste(unique(current$HomeTeam), "_H", sep = "")
away_params <- paste(unique(current$AwayTeam), "_A", sep = "")
params <- rnorm(42,0,1)

home_team_mapped <- as.numeric(mapvalues(current$HomeTeam, from = unique(current$HomeTeam), to = seq(1:20)))
key_vals <- home_team_mapped[1:20]
names(key_vals) <- current$HomeTeam[1:20]

for(i in 1:nrow(current)) {
    current$AwayTeam[i] <- which(names(key_vals) == current$AwayTeam[i])
}
current$HomeTeam <- home_team_mapped
current$AwayTeam <- as.numeric(current$AwayTeam)

result <- optim(par=rnorm(42,0,0.3), fn=NLL, data=current, method = "BFGS",
                hessian = TRUE , control = list(trace=0))
