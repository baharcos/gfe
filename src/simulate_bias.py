import numpy as np
import pandas as pd
from gfe_estimation import estimate_grouped_fixed_effect_model_parameters
from bic import gfe_bic
import statsmodels.api as sm
from linearmodels import PanelOLS

def simulate_ovb(
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
    
       y_it = x_it'theta + alpha_g(i)t + upsilon_it
     
       y = X theta + dummy alpha + upsilon
     
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

    if theta.size != nfeatures:
        raise ValueError("Argument *alpha* should be of size nfeatures.")
    
    nobs = nindividuals*nperiods
    X = np.random.uniform(low, up, (nobs,nfeatures))
    upsilon = np.random.randn(nobs) 
    # upsilons are i.i.d. from a normal distribution which is necessary 
    # to meet estimator's asymptotic properties
    if (ngroups*nperiods) * int(nindividuals/ngroups) == nobs:
        dummy = np.tile(np.identity(ngroups*nperiods), int(nindividuals/ngroups)).T
    else:
        dummy = np.tile(np.identity(ngroups*nperiods), (1+int(nindividuals/ngroups))).T[:nobs]
    X[:,0] = X[:,0] + dummy @ alpha*0.2
    X[:,1] = X[:,1] + dummy @ (alpha**3)*0.05
    #X = X.reshape(nobs,nfeatures)
    y = dummy @ alpha + X@theta + upsilon
    #X = X.reshape(nobs,nfeatures)
    #y = dummy @ alpha + X@theta + 0.01 * upsilon
    return y, X

def monte_carlo_simulation(
    nindividuals,
    nfeatures,
    ngroups,
    specified_ngroups,
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
        specified_ngroups([type]): [description]
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
    estimates_var = []
    
    for i in range(nreps):
        y, X = simulate_ovb(nindividuals, nfeatures, ngroups, nperiods, theta, alpha, low, up) 
        estimates,std = estimate_grouped_fixed_effect_model_parameters(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, nindividuals=nindividuals, 
                                                                   alpha_0=alpha_0, theta_0 = theta_0, X=X, nfeatures=nfeatures)
        monte_carlo_coefficients.append(estimates)
        
        #variance = np.concatenate((gfe_variance(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, 
        #nindividuals=nindividuals, nfeatures=nfeatures, alpha_0=alpha_0, 
        #theta_0=theta_0, X=X)))
        estimates_var.append(std)
        
    return(monte_carlo_coefficients, estimates_var)

def monte_carlo_simulation_groups(
    nindividuals,
    nfeatures,
    ngroups,
    specified_ngroups,
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
        specified_ngroups([type]): [description]
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
    estimates_sd = []
    ic =[]
    #objective=[]
    
    for i in range(nreps):
        y, X = simulate_ovb(nindividuals, nfeatures, ngroups, nperiods, theta, alpha, low, up) 
        a = gfe_bic(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, nindividuals=nindividuals, 
                                                                   alpha_0=alpha_0, theta_0 = theta_0, X=X, nfeatures=nfeatures)
        monte_carlo_coefficients.append(a['Estimates'])
        
        #variance = np.concatenate((gfe_variance(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, 
        #nindividuals=nindividuals, nfeatures=nfeatures, alpha_0=alpha_0, 
        #theta_0=theta_0, X=X)))
        estimates_sd.append(a['sd'])
        ic.append(a['bic'])
        #objective.append(a['objective'])
        
    return({'Estimates': monte_carlo_coefficients, 'sd': estimates_sd}, {'bic': ic})


def monte_carlo_simulation_ols(
    nindividuals,
    nfeatures,
    ngroups,
    nperiods,
    theta,
    alpha,
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
        specified_ngroups([type]): [description]
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
    estimates_var = []
    
    for i in range(nreps):
        y, X = simulate_ovb(nindividuals, nfeatures, ngroups, nperiods, theta, alpha, low, up)
        X = sm.add_constant(X)
        ols = sm.OLS(y,X).fit()
        monte_carlo_coefficients.append(ols.params[1:])
        #variance = np.concatenate((gfe_variance(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, 
        #nindividuals=nindividuals, nfeatures=nfeatures, alpha_0=alpha_0, 
        #theta_0=theta_0, X=X)))
        estimates_var.append(ols.bse[1:])
        
    return(monte_carlo_coefficients, estimates_var)


def monte_carlo_simulation_fe(
    nindividuals,
    nfeatures,
    ngroups,
    nperiods,
    theta,
    alpha,
    low,
    up,
    nreps,
    seed,
    time_effects
):
    """[summary]

    Args:
        nindividuals ([type]): [description]
        nfeatures ([type]): [description]
        ngroups ([type]): [description]
        specified_ngroups([type]): [description]
        nperiods ([type]): [description]
        theta ([type]): [description]
        alpha ([type]): [description]
        theta_0
        alpha_0
        low ([type]): [description]
        up ([type]): [description]
        nreps ([type]): [description]
        seed ([type]): [description],
        time_effects(boolean): whether or not include time fe
    
    Return:

    """
    np.random.seed(seed)
    monte_carlo_coefficients = []
    estimates_var = []
    
    for i in range(nreps):
        y, X = simulate_ovb(nindividuals, nfeatures, ngroups, nperiods, theta, alpha, low, up)
        X = sm.add_constant(X)
        a = np.empty((y.size,(nfeatures+4))) #creating the data for FE reg
        a[:,0] = np.tile(np.arange(0,nindividuals,1),(nperiods,1)).T.flatten() #individual index
        a[:,1] = np.tile(np.arange(0,nperiods,1),nindividuals) #time index
        a[:,2] = y
        a[:,3:]= X
        #only for two covariates, could change this later :)
        a = pd.DataFrame(a,columns=('i','t','y','x0','x1','x2'))
        a = a.set_index(['i', 't'])
        FE = PanelOLS(a.y,a[['x0','x1','x2']], entity_effects = True, time_effects=time_effects).fit(cov_type = 'clustered',cluster_entity=True)
        monte_carlo_coefficients.append(FE.params[1:])
        #variance = np.concatenate((gfe_variance(outcome=y, ngroups=specified_ngroups, nperiods=nperiods, 
        #nindividuals=nindividuals, nfeatures=nfeatures, alpha_0=alpha_0, 
        #theta_0=theta_0, X=X)))
        estimates_var.append(FE.std_errors[1:])

    
    sd = pd.DataFrame(estimates_var).reset_index(drop=True)
    realizations = pd.DataFrame(monte_carlo_coefficients).reset_index(drop=True)
        
    return(realizations, sd)

def table(realizations, std, true_value, estimator, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std
    return pd.DataFrame({ 
        'T' : T,
        'N' : N,
        'estimator': estimator,
        'bias' : realizations.mean() - true_value,
        'rmse': np.sqrt(((realizations - true_value)**2).mean()),
        'coverage probability' : (((ci_low<=true_value)&(ci_up>=true_value))*1).mean()
        })

def _alpha_1(t):
    return np.exp(t/20)

def _alpha_2(t):
    return -t/5

def alpha(T):
    alpha_1t = []
    for t in range(T):
        alpha_1t.append(_alpha_1(t+1))
    
    alpha_2t = []
    for t in range(T):
        alpha_2t.append(_alpha_2(t+1))
    return np.array(alpha_1t + alpha_2t)




if __name__ == "__main__":
    from simulate import simulate
    
    def alpha_1(t):
        return np.exp(t/10)
    
    def alpha_2(t):
        return t

    alpha_1t = []
    for t in range(5):
         alpha_1t.append(alpha_1(t+1))
    
    alpha_2t = []
    for t in range(5):
        alpha_2t.append(alpha_2(t+1))

    alpha =  np.array(alpha_1t + alpha_2t)

    params = {"nindividuals" : 100,
          "specified_ngroups" : 2,
          "nperiods" : 10,
          "nfeatures" : 2,
          "ngroups" : 2,
          "nfeatures" : 2,
          "theta" : np.array([0.1, 0.5]),
          "alpha" : alpha,
          "theta_0" : np.array([0.2, 0.3]),
          "alpha_0" : alpha,
          "low" : -2,
          "up" : 2,
          "nreps" : 1000,
          "seed" : 1}

    monte_carlo_simulation(**params)


    # simulate_ovb(
    # nindividuals=50,
    # nfeatures=1,
    # ngroups=2,
    # nperiods=5,
    # theta=np.array([0.5]),
    # alpha=np.array(alpha_1t + alpha_2t),
    # low=0,
    # up=5
    # )