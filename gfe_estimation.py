import numpy as np

"""Implementing Algorithm 1 of Grouped Fixed Effect Estimator
in Bonhomme and Manresa(2015):

 1. Let (theta_0,alpha_0) be starting values.
 
 2. Compute the group membership dummy matrix (g) of size (NxT, GxT) for all individuals minimizing 
    sum(y_it - theta_0.x_it - alpha_0.g)^2.

 3. Estimate the (theta, alpha) by OLS.
 
 Iterate until (theta, alpha) converges to a value (hopefully to their true value.)
 
"""

def _grouping(outcome, groups, periods, individuals, alpha_0, theta_0, X):
    

    """Clusters the outcome(y_it(no covariates) or y_it - delta.x_it) into groups for 
    a given total group number of group and 
    a group fixed effect parameter.
    
    Args:
        outcome (np.ndarray): Dependent variable.
        groups (int): Pre-specified total number of groups counting 0 i.e. G.
        periods (int): Number of periods starting from 0 i.e. T.
        individuals(int): The number of different individual units in data i.e. N
        alpha_0 (np.ndarray): 1-d array with initial grouped fixed effect guess.
        theta_0 (np.ndarray): 1-d array with initial time invariant signal guess.
        X (np.ndarray): k-d array containing covariates with NxT rows
                        and k columns. They have to vary over time.(I might put an assert case for this.)

    Returns:
        (np.ndarray): A GxT-d array with NxT rows containing the dummy variable for each group-in-time 
                           membership. Each row represent an individual at a certain point in time and each column
                           represent the membership of a certain group in a point in time. 
                           (Group membership does not change over time.)
        
    """
    
    # i. Calculate the square error as if all individuals belong to one group for all groups.
    group_error = np.empty((outcome.size,groups))
    for g in range(groups):
            group_error[:,g] = np.square(outcome - X@theta_0 - 
            np.tile(alpha_0[g*periods:(g+1)*periods],100))
   
    # ii. Assign individuals to the group that gave the smallest square error. 
    group_assignment = np.argmin(group_error, axis =1)
    
    # iii. Create the dummy matrix
    dummy = np.zeros(((individuals*periods),(groups*periods)))
    column = 0
    for g in range(groups):
        group = 1 * (group_assignment == g)
        for t in range(periods):
            time_dummy = np.zeros(periods)
            time_dummy[t] = 1
            time_dummy = np.tile(time_dummy, individuals).astype(int)
            dummy[:,(column)] = group * time_dummy
            column += 1 
            if column == groups + periods:
                break
                
    return dummy


def _get_estimates(regression_data):
    """Estimation by OLS.
    Args:
    regression_data: the matrix that contains outcome variable in column 0 and covariates 
                     in columns 1:k and the group.time dummies in columns k:.
    
    Returns:
    estimates(np.ndarray): 1-d array with (k+G*T) columns with OLS estimates of GFE model.
                           column 0:k is the time invariant signal and k: is the group fixed
                           effect estimates.
    """
    estimates = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(regression_data[:,1:].T,regression_data[:,1:])
            ),regression_data[:,1:].T),regression_data[:,0])
    return estimates

def estimate_grouped_fixed_effect_model_parameters(outcome, groups, periods, individuals, 
alpha_0, theta_0, X,k):
    """Iterates 

    Args:
        outcome (np.ndarray): Dependent variable.
        groups (int): Pre-specified total number of groups i.e. G.
        periods (int): Number of periods i.e. T.
        individuals (int): The number of different individual units in data i.e. N.
        alpha_0 (np.ndarray): Initial starting values for group fixed effect estimates.
        theta_0 (np.ndarray): Initial starting values for 
        X (np.ndarray): k-d covariate matrix with N*T rows containing
                        arbitrarily correlated values with the group-specific unobservables, 
                        alpha_g_it.

        k (int): Number of covariates.

    Returns:
        np.ndarray: 1-d array that contains gfe model estimates. 
                    0:k time invariant coeefficients and k: grouped fixed effects' estimates.
    """

# time and group dummy that is 1 if the the outcome is in the particular time and the 
# individual belongs to the particular group.
    dummy = _grouping(outcome, groups, periods, individuals, alpha_0, theta_0, X)

# creating the matrix for OLS
    regression_data = np.empty((outcome.size, (groups*periods + k +1)))

# Column 0 contains the pooled dependent variable.    
    regression_data[:,0] = outcome

# Columns 1:k+1 contains the pooled k covariates
    regression_data[:,1:(k+1)] = X

# Column (k+1):(k+1+g*t) contains th time-group dummy variables
    regression_data[:,(k+1):] = dummy

# Get the new estimates with the estimated group structure using the initial values
# repeating until there is no update in estimate.
    update_estimates = _get_estimates(regression_data)
    regression_data[:,(k+1):] =  _grouping(outcome, groups, periods, individuals, update_estimates[k:], update_estimates[0:k], X)
    estimates = _get_estimates(regression_data)
    while all(update_estimates != estimates): #Criteria to stop updating
        regression_data[:,(k+1):] =  _grouping(outcome, groups, periods, individuals, estimates[k:], estimates[0:k], X)
        update_estimates = _get_estimates(regression_data)
        regression_data[:,(k+1):] =  _grouping(outcome, groups, periods, individuals, update_estimates[k:], update_estimates[0:k], X)
        estimates = _get_estimates(regression_data)

    return estimates
