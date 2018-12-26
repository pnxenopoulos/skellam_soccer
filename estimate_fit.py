import numpy as np
import pandas as pd

import scipy.optimize
from scipy.stats import skellam

SEASON = '2015-16'

data = pd.read_csv("epl_results.csv")
data['GDiff'] = data['HG'] - data['AG']
current = data[data['Season'] == SEASON]
current = current[['HomeTeam', 'AwayTeam', 'HG', 'AG', 'GDiff']]

teamDict = dict([(y,x+1) for x,y in enumerate(sorted(set(current['HomeTeam'])))])

current['HomeTeam'] = current.HomeTeam.replace(teamDict)
current['AwayTeam'] = current.AwayTeam.replace(teamDict)

data_array = np.array(current[['HomeTeam', 'AwayTeam', 'GDiff']])

def likelihoodFn(params, data):
	mu = params[0]
	h = params[1]
	sum_lik = 0
	for r in range(0,data.shape[0]):
		row = data[r,]
		home_id = row[0]
		away_id = row[1]
		z = row[2]
		lambda_one = np.exp(mu + h + params[1 + home_id] + params[21 + away_id])
		lambda_two = np.exp(mu + params[1 + away_id] + params[21 + home_id])
		#bessel = scipy.special.iv(2 * np.sqrt(lambda_one*lambda_one),np.abs(z))
		#print('we see at', r, 'a val of ', np.log(np.exp(-(lambda_one+lambda_two)) * (lambda_one/lambda_two)**(z/2) * bessel))
		sum_lik -= np.log(skellam.pmf(z,lambda_one, lambda_two))
	return sum_lik

from scipy.optimize import minimize, LinearConstraint
cons = np.stack((np.concatenate((np.array([0,0]),np.concatenate((np.repeat(1,20), np.repeat(0,20)), axis = 0))),
	np.concatenate((np.array([0,0]),np.concatenate((np.repeat(0,20), np.repeat(1,20)), axis = 0)))))
linear_constraint = LinearConstraint(cons, [0, 0], [0, 0])

start = np.random.uniform(0,1,42)
res = minimize(likelihoodFn, start, args=(data_array), method='trust-constr', constraints=[linear_constraint], options={'verbose': 1})


