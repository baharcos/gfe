import numpy as np
import pandas as pd
from gfe_estimation import estimate_grouped_fixed_effect_model_parameters
from simulate_bias import *

""" Runs Monte-Carlo Simulations and reports the results in a table
"nindividuals" = 1000, 100, 50
"nperiods" = 20, 10, 5"""

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

def alpha_1(t):
    return np.exp(t/20)

def alpha_2(t):
    return -t/5

def alpha(T):
    alpha_1t = []
    for t in range(T):
        alpha_1t.append(alpha_1(t+1))
    
    alpha_2t = []
    for t in range(T):
        alpha_2t.append(alpha_2(t+1))
    return np.array(alpha_1t + alpha_2t)

#improve the function
# alpha_1t = []
# for t in range(20):
#     alpha_1t.append(alpha_1(t+1))
# alpha_2t = []
# for t in range(20):
#     alpha_2t.append(alpha_2(t+1))

# alpha =  np.array(alpha_1t + alpha_2t)

params = {#"specified_ngroups" : 2,
          "nfeatures" : 2,
          "ngroups" : 2,
          "theta" : np.array([0.1, 0.5]),
          #"alpha" :  alpha,
          "theta_0" : np.array([0.2, 0.4]),
          #"alpha_0" : alpha, #([ 1.1,  1.2,  1.3,  1.5 ,  1.6, -0.2, -0.4, -0.6, -0.8, -1.])
          "low" : -3,
          "up" : 3,
          "nreps" : 1000,
          "seed" : 1}

columns2x5 = ['theta1_hat', "theta2_hat", 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat', 'alpha15_hat',
                                                   'alpha21_hat', 'alpha22_hat', 'alpha23_hat','alpha_24_hat','alpha_25_hat']


columns2x10 = ['theta1_hat', "theta2_hat", 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat', 'alpha15_hat',
                                                    'alpha16_hat', 'alpha17_hat', 'alpha18_hat', 'alpha19_hat', 'alpha110_hat','alpha21_hat', 'alpha22_hat',
                                                     'alpha23_hat','alpha_24_hat','alpha_25_hat', 'alpha_26_hat', 'alpha_27_hat', 'alpha_28_hat', 'alpha_29_hat',
                                                     'alpha_210_hat']

columns2x20 =   ['theta1_hat', "theta2_hat", 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat',
 'alpha15_hat', 'alpha16_hat', 'alpha17_hat', 'alpha18_hat', 'alpha19_hat', 'alpha110_hat', 'alpha111_hat',
 'alpha112_hat', 'alpha113_hat', 'alpha114_hat', 'alpha115_hat', 'alpha116_hat', 'alpha117_hat', 
 'alpha118_hat', 'alpha119_hat', 'alpha120_hat','alpha21_hat', 'alpha22_hat', 'alpha23_hat', 
 'alpha_24_hat','alpha_25_hat', 'alpha_26_hat', 'alpha_27_hat', 'alpha_28_hat', 'alpha_29_hat', 
 'alpha_210_hat', 'alpha211_hat', 'alpha212_hat', 'alpha213_hat', 'alpha_214_hat', 'alpha_215_hat', 
 'alpha_216_hat', 'alpha_217_hat', 'alpha_218_hat', 'alpha_219_hat', 'alpha_220_hat']  


""""
GFE """


#short panel, few observations
estimates, std =  monte_carlo_simulation(**params, nindividuals=50, nperiods=5, alpha=alpha(5), alpha_0=alpha(5), specified_ngroups=2)

realizations = pd.DataFrame(estimates,  columns = columns2x5)
sd  = pd.DataFrame(std,  columns = columns2x5)


estimates1, std1 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=5,alpha=alpha(5), alpha_0=alpha(5),specified_ngroups=2)

realizations1 = pd.DataFrame(estimates1,  columns = columns2x5)
std1  = pd.DataFrame(std1,  columns = columns2x5)

estimates2, std2 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=5,alpha=alpha(5), alpha_0=alpha(5), specified_ngroups=2)

realizations2 = pd.DataFrame(estimates2,  columns = columns2x5)
std2  = pd.DataFrame(std2,  columns = columns2x5)


estimates3, std3 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations3 = pd.DataFrame(estimates3,  columns = columns2x10)
std3  = pd.DataFrame(std3,  columns = columns2x10)


estimates4, std4 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations4 = pd.DataFrame(estimates4,  columns = columns2x10)
std4  = pd.DataFrame(std4,  columns = columns2x10)


estimates5, std5 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations5 = pd.DataFrame(estimates5,  columns = columns2x10)
std5  = pd.DataFrame(std5,  columns = columns2x10)


estimates6, std6 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations6 = pd.DataFrame(estimates6,  columns = columns2x20)
std6  = pd.DataFrame(std6,  columns = columns2x20)


estimates7, std7 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations7 = pd.DataFrame(estimates7,  columns = columns2x20)
std7  = pd.DataFrame(std7,  columns = columns2x20)

estimates8, std8 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations8 = pd.DataFrame(estimates8,  columns = columns2x20)
std8  = pd.DataFrame(std8,  columns = columns2x20)
#longer panel many obs grouping improves a lot with t and it does not have to be too long

#report on bias, rmse(root mean squared error), coverage probability.


GFE = pd.concat([table(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, estimator='GFE'),
table(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, estimator='GFE'),
table(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, estimator='GFE'),
table(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, estimator='GFE'),
table(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, estimator='GFE'),
table(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, estimator='GFE'),
table(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, estimator='GFE'),
table(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, estimator='GFE'),
table(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, estimator='GFE'),
])#.sort_index()#.to_latex('table_deneme.tex')


"""OLS
"""
index= ['theta1_hat', 'theta2_hat']*9

estimates9, sd9 = monte_carlo_simulation_ols(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1)


estimates10, sd10 = monte_carlo_simulation_ols(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1)


estimates11, sd11 = monte_carlo_simulation_ols(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1)


estimates12, sd12 = monte_carlo_simulation_ols(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1)


estimates13, sd13 = monte_carlo_simulation_ols(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1)


estimates14, sd14 = monte_carlo_simulation_ols(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1)


estimates15, sd15 = monte_carlo_simulation_ols(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1)


estimates16, sd16 = monte_carlo_simulation_ols(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1)


estimates17, sd17 = monte_carlo_simulation_ols(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1)

OLS =  pd.concat([
    table(pd.DataFrame(estimates9),pd.DataFrame(sd9), true_value= [0.1, 0.5], N=50, T=5, estimator='OLS'),
    table(pd.DataFrame(estimates10),pd.DataFrame(sd10), true_value= [0.1, 0.5], N=100, T=5, estimator='OLS'),
    table(pd.DataFrame(estimates11),pd.DataFrame(sd11), true_value= [0.1, 0.5], N=1000, T=5, estimator='OLS'),
    table(pd.DataFrame(estimates12),pd.DataFrame(sd12), true_value= [0.1, 0.5], N=50, T=10, estimator='OLS'),
    table(pd.DataFrame(estimates13),pd.DataFrame(sd13), true_value= [0.1, 0.5], N=100, T=10, estimator='OLS'),
    table(pd.DataFrame(estimates14),pd.DataFrame(sd14), true_value= [0.1, 0.5], N=1000, T=10, estimator='OLS'),
    table(pd.DataFrame(estimates15),pd.DataFrame(sd15), true_value= [0.1, 0.5], N=50, T=20, estimator='OLS'),
    table(pd.DataFrame(estimates16),pd.DataFrame(sd16), true_value= [0.1, 0.5], N=100, T=20, estimator='OLS'),
    table(pd.DataFrame(estimates17),pd.DataFrame(sd17), true_value= [0.1, 0.5], N=1000, T=20, estimator='OLS')
])
OLS.index=index
                                        

"""FE
"""

estimates18, sd18 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=False)


estimates19, sd19 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates20, sd20 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates21, sd21 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates22, sd22 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates23, sd23 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates24, sd24 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates25, sd25 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=False)

estimates26, sd26 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=False)


FE =  pd.concat([
    table(estimates18,sd18, true_value= [0.1, 0.5], N=50, T=5, estimator='FE'),
    table(estimates19, sd19, true_value= [0.1, 0.5], N=100, T=5, estimator='FE'),
    table(estimates20,sd20, true_value= [0.1, 0.5], N=1000, T=5, estimator='FE'),
    table(estimates21, sd21, true_value= [0.1, 0.5], N=50, T=10, estimator='FE'),
    table(estimates22,sd22, true_value= [0.1, 0.5], N=100, T=10, estimator='FE'),
    table(estimates23,sd23, true_value= [0.1, 0.5], N=1000, T=10, estimator='FE'),
    table(estimates24,sd24, true_value= [0.1, 0.5], N=50, T=20, estimator='FE'),
    table(estimates25,sd25, true_value= [0.1, 0.5], N=100, T=20, estimator='FE'),
    table(estimates26,sd26, true_value= [0.1, 0.5], N=1000, T=20, estimator='FE')
])
FE.index= index

"""TWFE
"""

estimates27, sd27 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=True)


estimates28, sd28 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates29, sd29 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=5,theta=np.array([0.1,0.5]),alpha=alpha(5),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates30, sd30 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates31, sd31 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates32, sd32 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=10,theta=np.array([0.1,0.5]),alpha=alpha(10),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates33, sd33 = monte_carlo_simulation_fe(nindividuals=50,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates34, sd34 = monte_carlo_simulation_fe(nindividuals=100,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=True)

estimates35, sd35 = monte_carlo_simulation_fe(nindividuals=1000,nfeatures=2,ngroups=2,
nperiods=20,theta=np.array([0.1,0.5]),alpha=alpha(20),low= -3,up=3,nreps=1000,seed=1, time_effects=True)


TWFE =  pd.concat([
    table(estimates27,sd27, true_value= [0.1, 0.5], N=50, T=5, estimator='TWFE'),
    table(estimates28, sd28, true_value= [0.1, 0.5], N=100, T=5, estimator='TWFE'),
    table(estimates29,sd29, true_value= [0.1, 0.5], N=1000, T=5, estimator='TWFE'),
    table(estimates30, sd30, true_value= [0.1, 0.5], N=50, T=10, estimator='TWFE'),
    table(estimates31,sd31, true_value= [0.1, 0.5], N=100, T=10, estimator='TWFE'),
    table(estimates32,sd32, true_value= [0.1, 0.5], N=1000, T=10, estimator='TWFE'),
    table(estimates33,sd33, true_value= [0.1, 0.5], N=50, T=20, estimator='TWFE'),
    table(estimates34,sd34, true_value= [0.1, 0.5], N=100, T=20, estimator='TWFE'),
    table(estimates35,sd35, true_value= [0.1, 0.5], N=1000, T=20, estimator='TWFE')
])
TWFE.index=index


# Make the table
pd.concat([GFE, OLS, FE, TWFE]).sort_index().to_latex('table_deneme.tex')