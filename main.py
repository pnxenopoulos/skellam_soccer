import numpy as np
import pandas as pd

from scipy.optimize import minimize, LinearConstraint
from scipy.stats import skellam

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

def minimize_likelihood(data):
	''' Function to minimize the likelihood function and return estimated parameters
	@param data: Data to use to minimize the likelihood
	'''
	# Set constraints
	cons = np.stack((np.concatenate((np.array([0,0]),np.concatenate((np.repeat(1,20), np.repeat(0,20)), axis = 0))),
	np.concatenate((np.array([0,0]),np.concatenate((np.repeat(0,20), np.repeat(1,20)), axis = 0)))))
	linear_constraint = LinearConstraint(cons, [0, 0], [0, 0])
	# Start starting params
	start = np.random.uniform(0,1,42)
	# Minimize the likelihood
	res = minimize(likelihoodFn, start, args=(data), method='trust-constr', constraints=[linear_constraint], options={'verbose': 1})
	return res.x

def calculateProb(home_id, away_id, params, num_teams):
	''' Function to calculate the outcome probabilities between two teams
	@param z: Goal difference
	@param home_id: Home team id
	@param away_id: Away team id
	@param params: Array of parameters, result of the minimization problem
	'''
	mu = params[0]
	h = params[1]
	lambda_one = np.exp(mu + h + params[1 + home_id] + params[num_teams + 1 + away_id])
	lambda_two = np.exp(mu + params[1 + away_id] + params[num_teams + 1 + home_id])
	home_loss = 0
	draw = 0
	home_win = 0
	for z in range(1,20):
		home_loss += skellam.pmf(-1*z, lambda_one, lambda_two)
		home_win += skellam.pmf(z, lambda_one, lambda_two)
	draw = skellam.pmf(0, lambda_one, lambda_two)
	return np.array((home_win, draw, home_loss))

all_seasons = pd.read_csv("test.csv")
leagues = all_seasons.name.unique()

for league in leagues:
	seasons = all_seasons[all_seasons['name'] == league].season.unique()
	for season in seasons:
		filtered_by_season = all_seasons[all_seasons['name'] == league]
		single_season = filtered_by_season[filtered_by_season['season'] == season]
		teamDict = dict([(y,x+1) for x,y in enumerate(sorted(set(single_season['team_long_name'])))])
		teams = len(teamDict)
		single_season['team_long_name'] = single_season.team_long_name.replace(teamDict)
		single_season['team_long_name_away'] = single_season.team_long_name_away.replace(teamDict)
		max_stage = single_season.stage.max()
		mid_stage = int(max_stage/2)
		df_ = pd.DataFrame(columns = ['team_long_name', 'team_long_name_away', 'gdiff', 'home_prob', 'draw_prob', 'away_prob', 'home_prob_est', 'draw_prob_est', 'away_prob_est'])
		for stage in range(mid_stage, max_stage-1):
			print("We are on", stage)
			train_data = single_season[single_season['stage'] < stage]
			train_data = np.array(train_data[['team_long_name', 'team_long_name_away', 'gdiff']])
			test_data = single_season[single_season['stage'] == stage]
			test_data = test_data[['team_long_name', 'team_long_name_away', 'gdiff', 'home_prob', 'draw_prob', 'away_prob', 'home_prob_est', 'draw_prob_est', 'away_prob_est']]
			est_params = minimize_likelihood(train_data)
			for index, row in test_data.iterrows():
				ht = int(row[0])
				at = int(row[1])
				estimated_odds = calculateProb(ht,at,est_params,teams)
				filter_team = (test_data['team_long_name'] == ht)
				test_data.home_prob_est[filter_team] = estimated_odds[0]
				test_data.draw_prob_est[filter_team] = estimated_odds[1]
				test_data.away_prob_est[filter_team] = estimated_odds[2]
			df_ = df_.append(test_data)
		file_name = str(league) + str(season)
		file_name = file_name.replace(" ","").replace("/","")
		df_.to_csv(str(league) + str(season) + ".csv",index=False)

