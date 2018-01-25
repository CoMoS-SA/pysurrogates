import numpy as np
import pandas as pd
import os.path
import numba
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.stats.distributions import entropy

import openturns as ot
from joblib import Parallel, delayed
from tqdm import tqdm

#""" (agent based) models"""
#from ABM_functions import *

""" Sampling / Design of experiments """
from SALib.sample import saltelli, sobol_sequence, latin
from pyDOE import *
import sobol_seq
import pynolh

""" Ignore Warnings """
import warnings
warnings.filterwarnings("ignore")

""" surrogate models """
# Xtreeme Gradient Boosted Decision Trees
from xgboost import XGBRegressor, XGBClassifier

# Gaussian Process Regression (Kriging)
# modified version of kriging to make a fair comparison with regard
# to the number of hyperparameter evaluations
from sklearn.gaussian_process import GaussianProcessRegressor

from joblib import Parallel, delayed

""" cross-validation
Cross validation is used in each of the rounds to approximate the selected 
surrogate model over the data samples that are available. 
The evaluated parameter combinations are randomly split into two sets. An 
in-sample set and an out-of-sample set. The surrogate is trained and its 
parameters are tuned to an in-sample set, while the out-of-sample performance 
is measured (using a selected performance metric) on the out-of-sample set. 
This out-of-sample performance is then used as a proxy for the performance 
on the full space of unevaluated parameter combinations. In the case of the 
proposed procedure, this full space is approximated by the randomly selected 
pool.
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from skopt import gp_minimize

""" performance metric """
# Mean Squared Error
from sklearn.metrics import mean_squared_error, f1_score

""" Defaults Algorithm Tuning Constants """
_N_EVALS = 15
_N_SPLITS = 5
_CALIBRATION_THRESHOLD = 1.00

# Functions
from time import time
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time()

def toc():
    return str(time() - startTime_for_tictoc)

""" Functions """
numba.jit()
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


""" FUNCTIONS FOR EVALUATING AND SCREENING ABMS, GENERAL PURPOSE"""

def build_Xy_table(ABM, size_sample, size_MC, SequenceFunction, SequenceFunctionString, as_df = True, to_file = True, **kwargs):
    
    import os
    
    problem_ABM = ABM.problem()

    if SequenceFunction == ot.MonteCarloExperiment:
        unscaled_set_X = np.array([ot.MonteCarloExperiment(ot.Uniform(0, 1), size_sample).generate() for i in range(problem_ABM['num_vars'])])
        unscaled_set_X = unscaled_set_X.reshape(size_sample, problem_ABM['num_vars'])
    
    else:
        SequenceFunction = SequenceFunction(problem_ABM['num_vars'], **kwargs)
        unscaled_set_X = np.array(SequenceFunction.generate(size_sample))

    set_X = rescale_sample(unscaled_set_X, problem_ABM['bounds'])
    
    # Evaluate
    set_y = [np.array(Parallel(n_jobs=-1)(delayed(ABM.model)(p) for p in set_X)) for i in range(size_MC)]
    
    # as DataFrame
    df = pd.DataFrame(set_X, columns = problem_ABM['names']).join(pd.DataFrame(set_y).T.add_prefix('evaluation_'))
    
    if to_file:
        # save file
        directory = 'ABM_eval_'+problem_ABM['abm_name']
        filename = SequenceFunctionString+'_ss'+str(size_sample)+'_MC'+str(size_MC)
        if not os.path.exists(directory):
            os.makedirs(directory+'/'+filename)
        df.to_csv(directory+'/'+filename, index = False)
        print 'Saved file '+filename
    if as_df:
        return df
    
def rescale_sample(sample, bounds):
    d = [b[1] - b[0] for b in bounds]
    m = [min(b) for b in bounds]
    rescaled_sample = m + (d * sample)
    return rescaled_sample





def evaluate_Xy(problem,
                params,
                function = None,
                set_X = None,
                compute_y = True,
                as_df = False):
    '''
    Generates a 2D array in which each row is an input combination for the model.
    It allows to evaluate model and surroate on these X points.
    Parameters
    ----------
    problem : dict
        contains information about the model, the name of the input parameters and their range.
    sampling : str
        Name of the sampling technique, possible values are 'saltelli' for Saltelli-Sobol sampling, 
        'latin' for latin hypercube, 'montecarlo', 'nolh' for nearly orthogonal latin hypercube or 'use set_X' 
        for providing a custom input set.
    N : int
        size of the X set. I.e. how many points you wish to evaluate on.
        Does not apply in 'nolh' and 'use set_X' sampling.
    function :    
        can be evaluate_ABM_on_set_X, which will evaluate the model in the problem dict,
        or any other function of the input, for example fitted predictors
    set_X : 2D array
        if sampling is 'use set_X', use this argument to input your custom set_X
    compute_y : boolean
        set to false if you are only interested in the sampling and do not with to obtain any output 'y'.
    as_df : boolean
        return result as a pandas DataFrame
    Returns:
    --------
    set_X, set_y (or pandas.DataFrame object if as_df==True)
        tuple with information of sampling points and evaluated function.
    Example:
    --------
    >>> set_X, set_y  = evaluate_Xy(problem = {
      'abm_name': 'Islands',
      'abm_function': island_abm,
      'num_vars': 7,
      'calibration_measure': growth_rate,
      'names': ['rho', 'alpha', 'phi','pi', 'eps', 'N','lambda'],
      'bounds': [[0,1], [0.8,2], [0.0,1.0], [0.0,1.0], [0.0,1.0], [3,15], [0.0,1.0]]
      'N': 50,
      'sampling': 'latin'
    })
    '''

    if 'sampling' in params:
        sampling = params['sampling']
    else:
        sampling = 'latin'

    if 'N' in params:
        N = params['N']
    else:
        N = 50

    if sampling == 'saltelli':
        set_X = saltelli.sample(problem, N, calc_second_order=False)
    elif sampling == 'latin':
        set_X = latin.sample(problem, N *(problem['num_vars'] + 2))
    elif sampling == 'montecarlo':
        set_X = np.vstack([np.random.uniform(p[0],p[1],size=(N, 1)).flatten() for p in problem['bounds']]).T
    elif sampling == 'nolh':
        nolh = pynolh.nolh(range(pynolh.params(problem['num_vars'])[1]))
        rescaled_nolh_T = nolh.T
        for i in range(problem['num_vars']):
            var_col = nolh.T[i]
            rescaled_nolh_T[i] = problem['bounds'][i][0] + var_col*(problem['bounds'][i][1] - problem['bounds'][i][0])
        set_X = rescaled_nolh_T.T
    elif sampling == 'use set_X':
        pass

    if compute_y:
        if function == evaluate_ABM_on_set_X:
            set_y = evaluate_ABM_on_set_X(problem['abm_function'], set_X, problem['calibration_measure'])
        elif function == None:
            print 'Need to define function (eg. evaluate_ABM_on_set_X)'
        else:
            set_y = function(set_X)
            
    else:
        set_y = None
    
    if as_df:
        df = pd.DataFrame(set_X, columns=problem['names'])
        df['y'] = pd.Series(set_y, index=df.index)
        return df
    else:
        return set_X, set_y

"""
def evaluate_ABM_on_set_X(abm_function, set_X, calibration_measure = growth_rate, parallel=True):


    if parallel:
        outputs = Parallel(n_jobs=-1)(delayed(abm_function)(*pc,_RNG_SEED=rng_seed) for rng_seed, pc in enumerate(set_X))
        y = np.array(Parallel(n_jobs=-1)(delayed(calibration_measure)(output) for output in outputs))
    else:
        y = np.zeros(set_X.shape[0])
        for i, pc in enumerate(set_X):
            output = abm_function(*pc,_RNG_SEED=i)
            y[i] = calibration_measure(output)
        
    return y
"""

def bin_variable_and_get_stats(data_for_binning, problem, var, bins = 30):
    
    """
   
    """

    df = data_for_binning
    perc = ((bins/100.)*df.rank(pct = True)).round(2)
    perc.columns = [n+'_perc' for n in (problem['names'] + ['y'])]

    df = df.join(perc)
    
    grouped = df.groupby(var+'_perc')['y']
    x, mean_y = df.groupby(var+'_perc').mean()[var].values, grouped.mean().values
    q25_y = grouped.quantile(.25).values
    q75_y = grouped.quantile(.75).values

    return x, mean_y, q25_y, q75_y

def save_2D_expectation_figure(path, df, interpolate = True, edgecolors = 'None', vmin = -0.1, vmax = 0.4, show = False):
    """
    Example:
    save_2D_expectation_figure('./2Dexpect_data/'+Y+'_data_'+str(i+1)+str(j+1), subplot_data)
    """
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
        
    # Generate data:
    x, y, z = df.as_matrix().T

    if interpolate:
        # Set up a regular grid of interpolation points
        xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate
        rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
        zi = rbf(xi, yi)

        plt.imshow(zi, vmin = vmin, vmax = vmax, origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

    plt.scatter(x, y, c=z, vmin =vmin, vmax = vmax, edgecolors='None')
#     plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.savefig(path)
    if show == True:
        plt.show()
    plt.clf()


def create_2variable_plots(df, problem, path, bins = 30, vmin=-0.1, vmax=0.4):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            pass
    
    var = problem['names']
    output_column = df.columns[-1]
    for i in range(problem['num_vars']):
        for j in range(0, i):
            if j != i:
                df['percentile_i'] = ((bins/100.)*df[var[i]].rank(pct = True)).round(2)
                df['percentile_j'] = ((bins/100.)*df[var[j]].rank(pct = True)).round(2)
                grouped = df.groupby(['percentile_i','percentile_j']).mean()
                subplot_data = grouped[[var[i], var[j],output_column]]
                subplot_data.to_csv(path+str(i+1)+str(j+1))
                save_2D_expectation_figure(path+str(i+1)+str(j+1), 
                                           subplot_data, interpolate = True, edgecolors = 'k', vmin = vmin, vmax = vmax)

def surrogate_performance_profile(train_set_X, test_set_X, test_set_y, problem, surrogate_model = 'kriging',
                                  bootstr_repeats = 150):
    
    """Input train and test set X, as well as test set y of the ABM. Tell the surrogate model name ABM name.
    It returns a tuple of training times (measured in runs of the ABM) and a bootstrap result of mse, 
    on which moments or quantiles can be computed to get confidence interval of mse.
    
    :train_set_X: Here the surrogate is trained and timed.
    :test_set_X: X set for mse.
    :test_set_y: from here mse is computed.
    :surrogate_model: either 'kriging' or 'XGBoost'
    :abm_function: name of the function of the abm
    :bootstr_repeats: how many half-samples to compute mse on

    :Example:
    >>> y = evaluate_ABM_on_set_test(island_abm, train_set_X)
    """
    #Timing of training
    tic()
    train_set_y = evaluate_ABM_on_set_X(problem['abm_function'], train_set_X, problem['calibration_measure'])
    if surrogate_model == 'kriging':
        surrogate_fit = GaussianProcessRegressor(random_state=0).fit(train_set_X, train_set_y)
    elif surrogate_model == 'XGBoost':
        surrogate_fit = fit_surrogate_model(train_set_X,train_set_y)
        
    Y_surr = surrogate_fit.predict(test_set_X)
    train_time = float(toc())
    
    #mean squared error on montecarlo sample
    mse = (test_set_y - Y_surr)**2
    bootstrap = [np.mean(mse[np.random.choice(mse.shape[0], mse.shape[0]/2, replace=True)]) for j in range(bootstr_repeats)]
    
    return (train_time, bootstrap)

def time_mse_profile(problem, budgets = [10, 15, 20], mc_size = 400, sampling = 'latin', bootstr_repeats = 150):
    """returns a 6 tuple with 
    :problem: dictionary with info of ABM function and input parameter space.
    :budgets: a list of training set sizes
    :mc_size: size of test set
    :sampling: either 'latin' or 'saltelli', sampling used for training

    :Example:
    >>> y = evaluate_ABM_on_set_test(island_abm, train_set_X)
    """
    l = len(budgets)
    krig_timing = np.zeros(l)
    XGB_timing = np.zeros(l)
    bs_krig = np.zeros((l,bootstr_repeats))
    bs_XGB = np.zeros((l,bootstr_repeats))
    

    for i in range(l):

        train_set_X = evaluate_Xy(problem, params = {'N': budgets[i],'sampling': sampling}, compute_y = False)[0]

        tic()
        test_set_X, test_set_y = evaluate_Xy(problem, params = {'N': mc_size,'sampling': 'latin'}, function=evaluate_ABM_on_set_X)
        ABM_timing = float(toc())/budgets[-1]

        krig_timing[i], bs_krig[i] = surrogate_performance_profile(train_set_X, test_set_X, test_set_y, problem, 'kriging', bootstr_repeats = bootstr_repeats)
        XGB_timing[i],  bs_XGB[i]  = surrogate_performance_profile(train_set_X, test_set_X, test_set_y, problem, 'XGBoost', bootstr_repeats = bootstr_repeats)

    bootstr_sample = int(mc_size/2)
    mse_ABM = (test_set_y - evaluate_ABM_on_set_X(problem['abm_function'], test_set_X, calibration_measure = problem['calibration_measure']))**2
    bs_ABM = [np.mean(mse_ABM[np.random.choice(mse_ABM.shape[0], bootstr_sample, replace=True)]) for j in range(bootstr_repeats)]

    XGB_timing = XGB_timing / ABM_timing
    krig_timing = krig_timing / ABM_timing
    
    return ABM_timing, krig_timing, XGB_timing, bs_ABM, bs_krig, bs_XGB



"""SURROGATES FUNCTIONS AND SUCH STUFF"""

numba.jit()
def set_surrogate_as_gbt():
    """ Set the surrogate model as Gradient Boosted Decision Trees
    Helper function to set the surrogate model and parameter space
    as Gradient Boosted Decision Trees.
    For detail, see:
    http://scikit-learn.org/stable/modules/generated/
    sklearn.ensemble.GradientBoostingRegressor.html
    Parameters
    ----------
    None
    Returns
    -------
    surrogate_model :
    surrogate_parameter_space :
    """

    surrogate_model = XGBRegressor(seed=0)

    surrogate_parameter_space = [
        (10, 100),  # n_estimators
        (0.00001, 1),  # learning_rate
        (1, 100),  # max_depth
        (0.0, 1),  # reg_alpha
        (0.0, 1),  # reg_lambda
        (0.25, 1.0)]  # subsample

    return surrogate_model, surrogate_parameter_space

numba.jit()
def custom_metric_regression(y_hat, y):
    return 'MSE', mean_squared_error(y.get_label(), y_hat)

numba.jit()
def custom_metric_binary(y_hat, y):
    return 'MSE', f1_score(y.get_label(), y_hat, average='weighted')

numba.jit()
def fit_surrogate_model(X, y):
    """ Fit a surrogate model to the X,y parameter combinations
    Parameters
    ----------
    surrogate_model :
    X :
    y :
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted
    """
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt()

    print("NEW5")

    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, reg_lambda, subsample = params

        reg = XGBRegressor(n_estimators=n_estimators,
                           learning_rate=learning_rate,
                           max_depth=max_depth,
                           reg_alpha=reg_alpha,
                           reg_lambda=reg_lambda,
                           subsample=subsample,
                           seed=0)

        kf = KFold(n_splits=_N_SPLITS, random_state=0, shuffle=True)
        kf_cv = [(train, test) for train, test in kf.split(X, y)]

        return -np.mean(cross_val_score(reg,
                                        X, y,
                                        cv=kf_cv,
                                        n_jobs=1,
                                        fit_params={'eval_metric': custom_metric_regression},
                                        scoring="neg_mean_squared_error"))

    # use Gradient Boosted Regression to optimize the Hyper-Parameters.
    surrogate_model_tuned = gp_minimize(objective,
                                        surrogate_parameter_space,
                                        n_calls=_N_EVALS,
                                        n_jobs=-1,
                                        random_state=0,
                                        verbose=0)

    surrogate_model.set_params(n_estimators=surrogate_model_tuned.x[0],
                               learning_rate=surrogate_model_tuned.x[1],
                               max_depth=surrogate_model_tuned.x[2],
                               reg_alpha=surrogate_model_tuned.x[3],
                               reg_lambda=surrogate_model_tuned.x[4],
                               subsample=surrogate_model_tuned.x[5],
                               seed=0)

    surrogate_model.fit(X, y, eval_metric=custom_metric_regression)

    return surrogate_model

def run_online_surrogate(problem, budget, N, calibration_threshold):
    print "XGBoost Started!"
    # 1. Draw the Pool
    # Set pool size

    # Draw Pool
    pool = evaluate_Xy(problem, params, compute_y = False)[0]

    pool_size = len(pool)
    
    samples_to_select = np.ceil(np.log(pool_size)).astype(int)
    
    # Set initialization samples
    initialization_samples = np.random.permutation(pool_size)[:samples_to_select]

    # Evaluate Initialization Set
    evaluated_set_X, evaluated_set_y = evaluate_Xy(problem, params, set_X = pool[initialization_samples], 
                              function = evaluate_ABM_on_set_X)
    
    # Update unevaluated_set_X
    unevaluated_set_X = pool[list(set(range(pool_size)) - set(initialization_samples))]
    surrogate_model, surrogate_parameter_space = set_surrogate_as_gbt()
    print "Evaluated set size: ", evaluated_set_y.shape[0]

    while evaluated_set_y.shape[0] < budget:
        # 3. Build Surrogate on evaluated samples
        surrogate_model_this_round = fit_surrogate_model(evaluated_set_X,evaluated_set_y)

        # 4. Predict Response over Pool
        predict_response_pool = surrogate_model_this_round.predict(unevaluated_set_X)
        predicted_positives = calibration_condition(predict_response_pool,calibration_threshold)
        num_predicted_positives = predicted_positives.sum()

        # 5. Select small subset of Pool for Evaluation
        evaluated_set_X, evaluated_set_y, unevaluated_set_X = get_round_selections(evaluated_set_X,evaluated_set_y,
                unevaluated_set_X, predicted_positives, num_predicted_positives,
                samples_to_select, calibration_threshold,
		budget)

    # 6. Output Final Surrogate Model
    surrogate_model = fit_surrogate_model(evaluated_set_X,evaluated_set_y)

    return surrogate_model, evaluated_set_X

numba.jit()
def fit_entropy_classifier(X, y, calibration_threshold):
    """ Fit a surrogate model to the X,y parameter combinations
    Parameters
    ----------
    surrogate_model :
    X :
    y :
    Output
    ------
    surrogate_model_fitted : A surrogate model fitted
    """
    y_binary = calibration_condition(y, calibration_threshold)
    _, surrogate_parameter_space = set_surrogate_as_gbt()

    def objective(params):
        n_estimators, learning_rate, max_depth, reg_alpha, \
        reg_lambda, subsample = params

        clf = XGBClassifier(n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            subsample=subsample,
                            seed=0,
                            objective="binary:logistic")

        skf = StratifiedKFold(n_splits=_N_SPLITS, random_state=0, shuffle=True)
        skf_cv = [(train, test) for train, test in skf.split(X, y_binary)]

        return -np.mean(cross_val_score(clf,
                                        X, y_binary,
                                        cv=skf_cv,
                                        n_jobs=1,
                                        fit_params={'eval_metric': custom_metric_binary},
                                        scoring="f1_weighted"))

    # use Gradient Boosted Regression to optimize the Hyper-Parameters.
    clf_tuned = gp_minimize(objective,
                            surrogate_parameter_space,
                            n_calls=_N_EVALS,
                            acq_func='gp_hedge',
                            n_jobs=-1,
                            random_state=0)

    clf = XGBClassifier(n_estimators=clf_tuned.x[0],
                        learning_rate=clf_tuned.x[1],
                        max_depth=clf_tuned.x[2],
                        reg_alpha=clf_tuned.x[3],
                        reg_lambda=clf_tuned.x[4],
                        subsample=clf_tuned.x[5],
                        seed=0)

    clf.fit(X, y_binary, eval_metric=custom_metric_binary)

    return clf

numba.jit()
def get_sobol_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)

    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return sobol_samples

numba.jit()
def get_unirand_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = np.random.rand(n_dimensions,samples).T

    # Compute the parameter mappings between the Sobol samples and supports
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return unirand_samples

numba.jit()
def get_round_selections(evaluated_set_X, evaluated_set_y,
                         unevaluated_set_X,
                         predicted_positives, num_predicted_positives,
                         samples_to_select, calibration_threshold,
                         budget):
    """
    """
    samples_to_select = np.min([abs(budget - evaluated_set_y.shape[0]),
                                samples_to_select]).astype(int)

    if num_predicted_positives >= samples_to_select:
        round_selections = int(samples_to_select)
        selections = np.where(predicted_positives == True)[0]
        selections = np.random.permutation(selections)[:round_selections]

    elif num_predicted_positives <= samples_to_select:
        # select all predicted positives
        selections = np.where(predicted_positives == True)[0]

        # select remainder according to entropy weighting
        budget_shortfall = int(samples_to_select - num_predicted_positives)

        selections = np.append(selections,
                               get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                                                      unevaluated_set_X,
                                                     calibration_threshold,
                                                      budget_shortfall))

    else:  # if we don't have any predicted positive calibrations
        selections = get_new_labels_entropy(clf, unevaluated_set_X, samples_to_select)

    to_be_evaluated = unevaluated_set_X[selections]
    unevaluated_set_X = np.delete(unevaluated_set_X, selections, 0)
    evaluated_set_X = np.vstack([evaluated_set_X, to_be_evaluated])
    evaluated_set_y = np.append(evaluated_set_y, evaluate_islands_on_set(to_be_evaluated))

    return evaluated_set_X, evaluated_set_y, unevaluated_set_X

numba.jit()
def get_new_labels_entropy(evaluated_set_X, evaluated_set_y,
                           unevaluated_X, calibration_threshold,
                           number_of_new_labels):
    """ Get a set of parameter combinations according to their predicted label entropy
    """
    clf = fit_entropy_classifier(evaluated_set_X, evaluated_set_y, calibration_threshold)

    y_hat_probability = clf.predict_proba(unevaluated_X)
    y_hat_entropy = np.array(map(entropy, y_hat_probability))
    y_hat_entropy /= y_hat_entropy.sum()
    unevaluated_X_size = unevaluated_X.shape[0]

    selections = np.random.choice(a=unevaluated_X_size,
                                  size=number_of_new_labels,
                                  replace=False,
p=y_hat_entropy)
    return selections

print ("Imported successfully")
