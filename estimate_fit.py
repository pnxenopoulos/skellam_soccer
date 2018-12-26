import numpy as np
import pandas as pd

from scipy.optimize import minimize, LinearConstraint
from scipy.stats import skellam

SEASON = '2015-16'  # Define season for the script

# Read in data, create goal diff column, keep relevant columns
data = pd.read_csv("epl_results.csv")
data['GDiff'] = data['HG'] - data['AG']
current = data[data['Season'] == SEASON]

# Create dictionary to make team names into integers
teamDict = dict([(y,x+1) for x,y in enumerate(sorted(set(current['HomeTeam'])))])
current['HomeTeam'] = current.HomeTeam.replace(teamDict)
current['AwayTeam'] = current.AwayTeam.replace(teamDict)

# Keep relevant columns
data_clean = np.array(current[['HomeTeam', 'AwayTeam', 'GDiff']])

def likelihoodFn(params, data):
	''' Function to specify the likelihood given a set of parameters and data
	@param params: Array of parameters to use
	@param data: Array of data to use
	'''
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
		sum_lik -= np.log(skellam.pmf(z,lambda_one, lambda_two))
	return sum_lik

# Set linear constraints
cons = np.stack((np.concatenate((np.array([0,0]),np.concatenate((np.repeat(1,20), np.repeat(0,20)), axis = 0))),
	np.concatenate((np.array([0,0]),np.concatenate((np.repeat(0,20), np.repeat(1,20)), axis = 0)))))
linear_constraint = LinearConstraint(cons, [0, 0], [0, 0])

# Set the starting params for optimization
start = np.random.uniform(0,1,42)

# Minimize the likelihood
res = minimize(likelihoodFn, start, args=(data_clean), method='trust-constr', constraints=[linear_constraint], options={'verbose': 1})

def calculateProbs(z, home_team, away_team, team_dict, params):
	''' Function to calculate the outcome probabilities between two teams
	@param z: Goal difference
	@param home_team: Home team string
	@param away_team: Away team string
	@param team_dict: Dictionary mapping team names to integers
	@param params: Array of parameters, result of the minimization problem
	'''
	home_id = team_dict[home_team]
	away_id = team_dict[away_team]
	mu = params[0]
	h = params[1]
	lambda_one = np.exp(mu + h + params[1 + home_id] + params[21 + away_id])
	lambda_two = np.exp(mu + params[1 + away_id] + params[21 + home_id])
	return skellam.pmf(z,lambda_one, lambda_two)
