import pandas as pd
import numpy as np


def problem():
    problem_BH = { 
    'abm_name': 'BH',
    'names': ['dividend_fluctuation_range', 
                         'delta', 
                         'trend_1',
                         'trend_2', 
                         'beta', 
                         'nu',
                         'bias_1',
                         'bias_2',
                         'weight_past_profits',
                         'rational_expectation_cost',
                         'risk_free_return',
                         'dividend_stream',
                         'init_pdev_fund',
                         'sigma2',
                         'share_type_1'],
               'bounds': [[0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0], 
                          [0.0,1.0]]}

    problem_BH['num_vars'] = len(problem_BH['names'])
    return problem_BH



def model(params, T=200):
    
    #np.random.seed(_RNG_SEED)

    dividend_fluctuation_range, delta, trend_1, trend_2, beta, nu, bias_1, bias_2, weight_past_profits, rational_expectation_cost, risk_free_return, dividend_stream, init_pdev_fund, sigma2, share_type_1 = params
    risk_free_return += 1.0

    """ Dividend Thresholds """
    rng_uniform = np.random.uniform(low=-dividend_fluctuation_range, high=dividend_fluctuation_range, size=T)

    """ Default Response Value """
    response = np.array([0.0])

    """Set Fixed Parameters"""
    init_wtype_1 = 0
    init_wtype_2 = 0
    fund_price = dividend_stream / (risk_free_return - 1.0)

    """ Check that the price is positive """
    if share_type_1 * (trend_1 * init_pdev_fund + bias_1) + (1.0 - share_type_1) * (
            trend_2 * init_pdev_fund + bias_2) > 0:
        """ Preallocate Containers """
        X = np.zeros(T)
        P = np.zeros(T)
        N1 = np.zeros(T)

        """ Run simulation """
        for time in range(T):
            # Produce Forecast
            forecast = share_type_1 * (trend_1 * init_pdev_fund + bias_1)
            share_type_2 = 1.0 - share_type_1

            # Equilibrium before actual realized price
            forecast = forecast + share_type_2 * (trend_2 * init_pdev_fund + bias_2)

            # Realized equilibrium_price
            equilibrium_price_realized = forecast / risk_free_return

            # Accumulated type 1 profits
            init_wtype_1 = weight_past_profits * init_wtype_1 + (
                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                   trend_1 * dividend_stream + bias_1 - risk_free_return * init_pdev_fund) / (
                                   nu * sigma2) - rational_expectation_cost

            # Accumulated type 2 profits
            init_wtype_2 = weight_past_profits * init_wtype_2 + (
                    equilibrium_price_realized - risk_free_return * init_pdev_fund) * (
                                   trend_2 * dividend_stream + bias_2 - risk_free_return * init_pdev_fund) / (
                                   nu * sigma2)

            # Update fractions
            forecast = np.exp(beta * init_wtype_1)
            forecast = forecast + np.exp(beta * init_wtype_2)

            share_type_1 = delta * share_type_1 + (1.0 - delta) * (np.exp(beta * init_wtype_1) / forecast)
            share_type_2 = 1 - share_type_1

            # Set initial conditions for next period
            dividend_stream = init_pdev_fund
            init_pdev_fund = equilibrium_price_realized + rng_uniform[time]

            """ Record Results """
            # Prices
            X[time] = init_pdev_fund
            P[time] = X[time] + fund_price
            N1[time] = share_type_1

        lrets = pd.DataFrame(X[~np.isnan(X)])
        #lrets = np.log(lrets / lrets.shift(1)).dropna().values.T[0]
        return np.std(X[~np.isnan(X)])
    else:
        return None
    



