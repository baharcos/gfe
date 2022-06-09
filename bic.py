import numpy as np


"""Implementing Algorithm 1 of Grouped Fixed Effect Estimator
in Bonhomme and Manresa(2015):

 1. Let (theta_0,alpha_0) be starting values.
 
 2. Compute the group membership dummy matrix (g) of size (NxT, GxT) for all nindividuals minimizing 
    sum(y_it - theta_0.x_it - alpha_0.g)^2.

 3. Estimate the (theta, alpha) by OLS.
 
 4. Iterate until a numerical convergence of (theta, alpha).
 
"""

def _grouping(outcome, ngroups, nperiods, nindividuals, alpha_0, theta_0, X):
    

    """Clusters the outcome(y_it(no covariates) or y_it - delta.x_it) into ngroups for 
    a given total group number of group and 
    a group fixed effect parameter.
    
    Args:
        outcome (np.ndarray): Dependent variable.
        ngroups (int): Pre-specified total number of groups counting 0 i.e. G.
        nperiods (int): Number of periods starting from 0 i.e. T.
        nindividuals(int): The number of different individual units in data i.e. N
        alpha_0 (np.ndarray): 1-d array with initial grouped fixed effect guess.
        theta_0 (np.ndarray): 1-d array with initial time invariant signal guess.
        X (np.ndarray): nfeatures-d array containing covariates with NxT rows
                        and nfeatures columns. They have to vary over time.(I might put an assert case for this.)

    Returns:
        (np.ndarray): A GxT-d array with NxT rows containing the dummy variable for each group-in-time 
                           membership. Each row represent an individual at a certain point in time and each column
                           represent the membership of a certain group in a point in time. 
                           (Group membership does not change over time.)
        
    """
    
    # i. Calculate the square error as if all individuals belong to one group for all groups.
    group_error_observation = np.empty((outcome.size,ngroups))
    for g in range(ngroups):
            group_error_observation[:,g] = np.square(outcome - X@theta_0 - 
            np.tile(alpha_0[g*nperiods:(g+1)*nperiods],nindividuals))
    
    group_error_individual = np.empty((nindividuals,ngroups))
    for n in range(nindividuals):
        group_error_individual[n,:] = np.sum(group_error_observation[(n*nperiods):((n+1)*nperiods)], 
        axis=0)

    # ii. Assign individuals to the group that gave the smallest square error. 
    group_assignment = np.argmin(group_error_individual, axis =1)
    
    # iii. Create the dummy matrix
    dummy = np.zeros(((nindividuals*nperiods),(ngroups*nperiods)))
    column = 0
    for g in range(ngroups):
        group = 1 * (group_assignment == g)
        for t in range(nperiods):
            time_dummy = np.zeros(nperiods)
            time_dummy[t] = 1
            dummy[:,(column)] = np.kron(group,time_dummy)
            column += 1 
            if column == ngroups * nperiods:
                break
                
    return dummy

#@jit(nopython=True)
def _get_estimates(regression_data):
    """Estimation by OLS.
    Args:
    regression_data: the matrix that contains outcome variable in column 0 and covariates 
                     in columns 1:nfeatures and the group.time dummies in columns nfeatures:.
    
    Returns:
    estimates(np.ndarray): 1-d array with (nfeatures+G*T) columns with OLS estimates of GFE model.
                           column 0:nfeatures is the time invariant signal and nfeatures: is the group fixed
                           effect estimates.
    """
    estimates = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(regression_data[:,1:].T,regression_data[:,1:])
            ),regression_data[:,1:].T),regression_data[:,0])
    return estimates

def gfe_bic(outcome, ngroups, nperiods, nindividuals, 
alpha_0, theta_0, X,nfeatures):
    """Iterates 

    Args:
        outcome (np.ndarray): Dependent variable.
        ngroups (int): Pre-specified total number of groups i.e. G.
        nperiods (int): Number of periods i.e. T.
        nindividuals (int): The number of different individual units in data i.e. N.
        alpha_0 (np.ndarray): Initial starting values for group fixed effect estimates.
        theta_0 (np.ndarray): Initial starting values for 
        X (np.ndarray): nfeatures-d covariate matrix with N*T rows containing
                        arbitrarily correlated values with the group-specific unobservables, 
                        alpha_g_it.

        nfeatures (int): Number of covariates.

    Returns:
        np.ndarray: 1-d array that contains gfe model estimates. 
                    0:nfeatures time invariant coeefficients and nfeatures: grouped fixed effects' estimates.
    """
    nobs = nindividuals*nperiods

# time and group dummy that is 1 if the the outcome is in the particular time and the 
# individual belongs to the particular group.
    dummy = _grouping(outcome, ngroups, nperiods, nindividuals, alpha_0, theta_0, X)

# creating the matrix for OLS
    regression_data = np.empty((outcome.size, (ngroups*nperiods + nfeatures +1)))

# Column 0 contains the pooled dependent variable.    
    regression_data[:,0] = outcome

# Columns 1:nfeatures+1 contains the pooled nfeatures covariates
    regression_data[:,1:(nfeatures+1)] = X

# Column (nfeatures+1):(nfeatures+1+g*t) contains th time-group dummy variables
    regression_data[:,(nfeatures+1):] = dummy

# Get the new estimates with the estimated group structure using the initial values
# repeating until there is no update in estimate.
    update_estimates = _get_estimates(regression_data)
    regression_data[:,(nfeatures+1):] =  _grouping(outcome, ngroups, nperiods, nindividuals, update_estimates[nfeatures:], update_estimates[0:nfeatures], X)
    estimates = _get_estimates(regression_data)
    while all(update_estimates != estimates): #Criteria to stop updating
        regression_data[:,(nfeatures+1):] =  _grouping(outcome, ngroups, nperiods, nindividuals, estimates[nfeatures:], estimates[0:nfeatures], X)
        update_estimates = _get_estimates(regression_data)
        regression_data[:,(nfeatures+1):] =  _grouping(outcome, ngroups, nperiods, nindividuals, 
        update_estimates[nfeatures:], update_estimates[0:nfeatures], X)
        estimates = _get_estimates(regression_data)
    
    # Standard deviation estimator of the estimates
    Xdemeaned = np.zeros((nobs,nfeatures))
    residuals = outcome - regression_data[:,1:]@estimates
    var_alpha = np.empty(ngroups*nperiods)
    for i in range(ngroups*nperiods):
        var_alpha[i] = sum(np.square(residuals*dummy[:,i]))/sum(dummy[:,i])**2
        xbar =  sum((X.T * dummy[:,i]).T)/sum(dummy[:,i])
        Xdemeaned = Xdemeaned + (X.T * dummy[:,i]).T - (np.tile(xbar,
         ((nindividuals*nperiods),1)).T * dummy[:,i]).T
    
    Sigma_theta = Xdemeaned.T @ Xdemeaned / nobs
    Omega_theta =  ((Xdemeaned.T * residuals) @ (Xdemeaned.T * residuals).T)/nobs
    var_theta = np.diag(np.linalg.inv(Sigma_theta) @ Omega_theta @ np.linalg.inv(Sigma_theta))/nobs
    bic = sum(residuals**2)/nobs + np.log(nobs)*(sum(residuals**2)/(nobs - ngroups*nperiods - nindividuals - nfeatures)) * (ngroups*nperiods + nindividuals + nfeatures)/nobs
    #criterion = sum(residuals**2)
    
    return {'Estimates':estimates, 'sd' : np.concatenate((np.sqrt(var_theta), np.sqrt(var_alpha))), 'bic': bic} #, 'objective': criterion}




if __name__ == "__main__":
    from simulate_bias import *

    np.random.seed(1)
    y, X = simulate_ovb(
         nindividuals=100,
        nfeatures=2,
        ngroups=2,
        nperiods=5,
        theta=np.array([0.1,0.5]),
        alpha=alpha(5),
        low=-3,
        up=3
        )
    a =  estimate_grouped_fixed_effect_model_parameters(outcome=y, ngroups=2, nperiods=5, 
    nindividuals=100, alpha_0=alpha(5), theta_0 = np.array([0.1,0.5]), X=X, nfeatures=2)
    # from simulate import simulate
    
    # def alpha_1(t):
    #     return t/10

    # def alpha_2(t):
    #     return np.exp((1)/200)

    # alpha_1t = []
    # for t in range(900):
    #     alpha_1t.append(alpha_1(t+1))
    
    # alpha_2t = []
    # for t in range(900):
    #     alpha_2t.append(alpha_2(t+1))

    # y, X = simulate(nindividuals=100, nfeatures=2, ngroups=2, nperiods=5, 
    # alpha=np.array(alpha_1t[:5] + alpha_2t[:5]), theta=np.array([0.1,0.5]),
    # low = 0, up = 15)

    # estimate_grouped_fixed_effect_model_parameters(outcome=y, X=X, nindividuals=100, 
    # nfeatures=2, ngroups=2, nperiods=5, alpha_0=np.array(alpha_1t[:5] + alpha_2t[:5]), 
    # theta_0=np.array([0.1,0.5]))
    
    # params = {"nindividuals" : 100,
    #             "nfeatures" : 2,
    #             "ngroups" : 2,
    #             "nperiods" : 2,
    #             "theta" : np.array([0.1,0.5]),
    #             "alpha" : np.array([5, 20, 15, 50, 30, 80]),
    #             "theta_0" : np.array([0.05,0.3]),
    #             "alpha_0" : np.array([3, 15, 20, 10, 40, 55]),
    #             "low" : 0,
    #             "up" : 30,
    #             "nreps" : 1000,
    #             "seed" : 1}
          