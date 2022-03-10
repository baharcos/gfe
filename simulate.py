import numpy as np
from gfe_estimation import estimate_grouped_fixed_effect_model_parameters

def simulate(
    nindividuals,
    nfeatures,
    ngroups,
    nperiods,
    theta,
    alpha,
    low,
    up
):
    """Simulate data with unobserved grouped dynamic heterogeneity:
    outcomes y (length *nobs*), 
    covariates X (*nobs* x *nfeatures*) and
    grouped fixed effects alpha by imposing a linear model.
    
       y_it = x_it'theta + alpha_g(i)t + epsilon_it
     
       y = X theta + dummy alpha + epsilon
     
     g(i) is the function assigning each individual unit into a group, 
     alpha_g(i)t is the group and time specific fixed effect 
     specifying the grouped pattern of dynamic(?) heterogeneity.
    
    Args:
        nindividuals (int): The number of different individual units in data i.e. N.
        nfeatures (int): The number of covariates i.e. p.
        ngroups (int): The number of groups i.e. G
        nperiods (int): The number of periods i.e. T.
        theta (np.ndarray): The model parameters of independent variables.
        alpha (np.ndarray): The grouped fixed-effect parameter of size *ngroups.nperiods.*.
        low (int/float): lower bound of the uniform distribution the independent variables are sampled.
        up (int/float): upper bound of the uniform distribution the independent variables are sampled..
    
    Returns:
        y (np.ndarray): shape (nobs) the outcome vector.
        X (np.ndarray): shape (nobs, nfeatures) the independent variable matrix.
    
    Raises:
        ValueError, if dimensions mismatch or data type of arguments is incorrect.
    
    """
    #Assert input dimensions
    if alpha.size != ngroups*nperiods:
        raise ValueError("Argument *alpha* should be of size ngroups*nperiods.")
    
    nobs = nindividuals*nperiods
    X = np.random.uniform(low, up,(nobs,nfeatures))
    upsilon = np.random.randn(nobs) 
    # upsilons are i.i.d. from a normal distribution which is necessary 
    # to meet estimator's asymptotic properties
    if (ngroups*nperiods) * int(nindividuals/ngroups) == nobs:
        dummy = np.tile(np.identity(ngroups*nperiods), int(nindividuals/ngroups)).T
    else:
        dummy = np.tile(np.identity(ngroups*nperiods), (1+int(nindividuals/ngroups))).T[:nobs]
    y = dummy @ alpha + X@theta + upsilon
    return y, X

def monte_carlo_simulation(
    nindividuals,
    nfeatures,
    ngroups,
    nperiods,
    theta,
    alpha,
    theta_0,
    alpha_0,
    low,
    up,
    nreps,
    seed
):
    """[summary]

    Args:
        nindividuals ([type]): [description]
        nfeatures ([type]): [description]
        ngroups ([type]): [description]
        nperiods ([type]): [description]
        theta ([type]): [description]
        alpha ([type]): [description]
        theta_0
        alpha_0
        low ([type]): [description]
        up ([type]): [description]
        nreps ([type]): [description]
        seed ([type]): [description]
    
    Return:

    """
    np.random.seed(seed)
    monte_carlo_coefficients = []
    
    for i in range(nreps):
        y, X = simulate(nindividuals, nfeatures, ngroups, nperiods, theta, alpha, low, up) 
        estimates = estimate_grouped_fixed_effect_model_parameters(outcome=y, ngroups=ngroups, nperiods=nperiods, nindividuals=nindividuals, 
                                                                   alpha_0=alpha_0, theta_0 = theta_0, X=X, nfeatures=nfeatures)
        monte_carlo_coefficients.append(estimates)
    
    return(monte_carlo_coefficients)