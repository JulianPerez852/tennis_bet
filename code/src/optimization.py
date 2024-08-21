import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Optimization:
    
    def __init__(self, df_risk, total_money, max_loss_percentage,min_percentage):
        self.df_risk                = df_risk
        self.ev                     = df_risk['Best_Bet_EV'].values
        self.prob_win               = df_risk['Prob_Win'].values  
        self.sharpe_ratio           = df_risk['Best_Bet_Sharpe'].values
        self.total_money            = total_money
        self.max_loss_percentage    = max_loss_percentage
        self.min_percentage         = min_percentage
                
    # Define the utility function to maximize
    def utility_function(self, bets):
        # The utility can be defined as a weighted combination of EV, probability of win, and Sharpe ratio
        utility = np.sum(bets * (self.ev * self.prob_win + 0.5 * self.sharpe_ratio))
        return -utility  # We minimize the negative utility for maximization
    
    def conditions(self):
        total_money             = self.total_money
        min_percentage          = self.min_percentage
        max_loss_percentage     = self.max_loss_percentage

        # Constraints: the sum of bets should equal total_money
        constraints = [{'type': 'eq', 'fun': lambda bets: np.sum(bets) - total_money}]

        # Every prob of win greater than 0.6 is a mandatory bet
        # If a prob isn't greater than 0.5 I won't bet a lot of money, but if it has a good SR or EV, I still can bet.
        for i, prob in enumerate(self.prob_win):
            if prob > 0.6:
                constraints.append({'type': 'ineq', 'fun': lambda bets, i=i: bets[i] - min_percentage * total_money})
            elif prob < 0.5:
                constraints.append({'type': 'ineq', 'fun': lambda bets, i=i: max_loss_percentage * total_money - bets[i]})
                
        # Bounds: each bet must be non-negative
        bounds = [(0, total_money) for _ in range(len(self.df_risk))]

        # Initial guess: split the money equally among all matches
        initial_bets = np.full(len(self.df_risk), total_money / len(self.df_risk))
                
        return constraints, bounds, initial_bets
    
    def optimize(self):
        constraints, bounds, initial_bets = self.conditions()

        # Solve the optimization problem
        result = minimize(self.utility_function, initial_bets, method='SLSQP', bounds=bounds, constraints = constraints)

        # Extract the optimal bets
        optimal_bets = [f"{bet:.2f}" for bet in result.x]
        self.df_risk['Money_to_Bet'] = pd.to_numeric(optimal_bets)
        self.df_risk = self.df_risk[(self.df_risk['Money_to_Bet'] > 0)]
        
        return self.df_risk