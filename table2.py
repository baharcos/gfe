import numpy as np
import pandas as pd
from gfe_estimation import estimate_grouped_fixed_effect_model_parameters
from simulate_bias import *

""" Runs Monte-Carlo Simulations and reports the results in a bias
"nindividuals" = 1000, 100, 50
"nperiods" = 20, 10, 5
"ngroups" = 4, 3, 2, 1 """


def bias(realizations, std, true_value, G, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std
    return pd.DataFrame({ 
        'T' : T,
        'N' : N,
        'G': G,
        'bias' : realizations.mean() - true_value
        })

def rmse(realizations, std, true_value, G, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std
    return pd.DataFrame({ 
        'T' : T,
        'N' : N,
        'G': G,
        'rmse': np.sqrt(((realizations - true_value)**2).mean())
        })
def cp(realizations, std, true_value, G, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std
    return pd.DataFrame({ 
        'T' : T,
        'N' : N,
        'G': G,
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


#columns2x5 = ['$\hat{\theta_1}$', '$\hat{\theta_2}$', 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat', 'alpha15_hat',
#                                                   'alpha21_hat', 'alpha22_hat', 'alpha23_hat','alpha_24_hat','alpha_25_hat']


#columns2x10 = ['$\hat{\theta_1}$', '$\hat{\theta_2}$', 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat', 'alpha15_hat',
#                                                    'alpha16_hat', 'alpha17_hat', 'alpha18_hat', 'alpha19_hat', 'alpha110_hat','alpha21_hat', 'alpha22_hat',
#                                                     'alpha23_hat','alpha_24_hat','alpha_25_hat', 'alpha_26_hat', 'alpha_27_hat', 'alpha_28_hat', 'alpha_29_hat',
#                                                     'alpha_210_hat']

# columns2x20 =   ['$\hat{\theta_1}$', '$\hat{\theta_2}$', 'alpha11_hat', 'alpha12_hat', 'alpha13_hat', 'alpha14_hat',
#  'alpha15_hat', 'alpha16_hat', 'alpha17_hat', 'alpha18_hat', 'alpha19_hat', 'alpha110_hat', 'alpha111_hat',
#  'alpha112_hat', 'alpha113_hat', 'alpha114_hat', 'alpha115_hat', 'alpha116_hat', 'alpha117_hat', 
#  'alpha118_hat', 'alpha119_hat', 'alpha120_hat','alpha21_hat', 'alpha22_hat', 'alpha23_hat', 
#  'alpha_24_hat','alpha_25_hat', 'alpha_26_hat', 'alpha_27_hat', 'alpha_28_hat', 'alpha_29_hat', 
#  'alpha_210_hat', 'alpha211_hat', 'alpha212_hat', 'alpha213_hat', 'alpha_214_hat', 'alpha_215_hat', 
#  'alpha_216_hat', 'alpha_217_hat', 'alpha_218_hat', 'alpha_219_hat', 'alpha_220_hat']  


""""
G = 1 """


#short panel, few observations
estimates, std =  monte_carlo_simulation(**params, nindividuals=50, nperiods=5, alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2, specified_ngroups=1)

realizations = pd.DataFrame(estimates,)#  columns = columns2x5)
sd  = pd.DataFrame(std)#,  columns = columns2x5)


estimates1, std1 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=5,alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2,specified_ngroups=1)

realizations1 = pd.DataFrame(estimates1)#,  columns = columns2x5)
std1  = pd.DataFrame(std1)#,  columns = columns2x5)

estimates2, std2 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=5,alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2, specified_ngroups=1)

realizations2 = pd.DataFrame(estimates2)#,  columns = columns2x5)
std2  = pd.DataFrame(std2)#,  columns = columns2x5)


estimates3, std3 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations3 = pd.DataFrame(estimates3)#,  columns = columns2x10)
std3  = pd.DataFrame(std3)#,  columns = columns2x10)


estimates4, std4 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations4 = pd.DataFrame(estimates4)#,  columns = columns2x10)
std4  = pd.DataFrame(std4)#,  columns = columns2x10)


estimates5, std5 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations5 = pd.DataFrame(estimates5)#,  columns = columns2x10)
std5  = pd.DataFrame(std5)#,  columns = columns2x10)


estimates6, std6 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations6 = pd.DataFrame(estimates6)#,  columns = columns2x20)
std6  = pd.DataFrame(std6)#,  columns = columns2x20)


estimates7, std7 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations7 = pd.DataFrame(estimates7)#,  columns = columns2x20)
std7  = pd.DataFrame(std7)#,  columns = columns2x20)

estimates8, std8 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations8 = pd.DataFrame(estimates8)#,  columns = columns2x20)
std8  = pd.DataFrame(std8)#,  columns = columns2x20)
#longer panel many obs grouping improves a lot with t and it does not have to be too long

""""
G = 3 """

estimates9, std9 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=5, 
alpha=alpha(5), alpha_0= np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)), specified_ngroups=3)

realizations9 = pd.DataFrame(estimates9)#,  columns = columns2x5)
sd9  = pd.DataFrame(std9)#,  columns = columns2x5)


estimates10, std10 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=5,
alpha=alpha(5), alpha_0=np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)),specified_ngroups=3)

realizations10 = pd.DataFrame(estimates10)#,  columns = columns2x5)
sd10  = pd.DataFrame(std10)#,  columns = columns2x5)

estimates11, sd11 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=5,
alpha=alpha(5), alpha_0=np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)), specified_ngroups=3)

realizations11 = pd.DataFrame(estimates11)#,  columns = columns2x5)
sd11  = pd.DataFrame(sd11)#,  columns = columns2x5)


estimates12, sd12 =  monte_carlo_simulation (**params, nindividuals=50, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),((alpha(10)[:10] + alpha(10)[10:20])/2)), specified_ngroups=3)

realizations12 = pd.DataFrame(estimates12)#,  columns = columns2x10)
sd12  = pd.DataFrame(sd12)#,  columns = columns2x10)


estimates13, std13 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),((alpha(10)[:10] + alpha(10)[10:20])/2)), specified_ngroups=3)

realizations13 = pd.DataFrame(estimates13)#,  columns = columns2x10)
sd13  = pd.DataFrame(std13)#,  columns = columns2x10)


estimates14, std14 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),((alpha(10)[:10] + alpha(10)[10:20])/2)), specified_ngroups=3)

realizations14 = pd.DataFrame(estimates14)#,  columns = columns2x10)
sd14  = pd.DataFrame(std14)#,  columns = columns2x10)


estimates15, std15 =  monte_carlo_simulation(**params, nindividuals=50, nperiods=20,
alpha=alpha(20), alpha_0=np.append(alpha(20),((alpha(20)[:20] + alpha(20)[20:40])/2)), specified_ngroups=3)

realizations15 = pd.DataFrame(estimates6)#,  columns = columns2x20)
sd15  = pd.DataFrame(std15)#,  columns = columns2x20)


estimates16, std16 =  monte_carlo_simulation(**params, nindividuals=100, nperiods=20,
alpha=alpha(20), alpha_0=np.append(alpha(20),((alpha(20)[:20] + alpha(20)[20:40])/2)), specified_ngroups=3)

realizations16 = pd.DataFrame(estimates16)#,  columns = columns2x20)
sd16  = pd.DataFrame(std16)#,  columns = columns2x20)

estimates17, std17 =  monte_carlo_simulation(**params, nindividuals=1000, nperiods=20,
alpha=alpha(20), alpha_0=np.append(alpha(20),((alpha(20)[:20] + alpha(20)[20:40])/2)), specified_ngroups=3)

realizations17 = pd.DataFrame(estimates17)#,  columns = columns2x20)
sd17  = pd.DataFrame(std17)#,  columns = columns2x20)

pd.concat([bias(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=1),
bias(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=1),
bias(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=1),
bias(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=1),
bias(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=1),
bias(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=1),
bias(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=1),
bias(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=1),
bias(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=1),
bias(realizations9.iloc[:,:2],sd9.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=3),
bias(realizations10.iloc[:,:2],sd10.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=3),
bias(realizations11.iloc[:,:2],sd11.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=3),
bias(realizations12.iloc[:,:2],sd12.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=3),
bias(realizations13.iloc[:,:2],sd13.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=3),
bias(realizations14.iloc[:,:2],sd14.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=3),
bias(realizations15.iloc[:,:2],sd15.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=3),
bias(realizations16.iloc[:,:2],sd16.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=3),
bias(realizations17.iloc[:,:2],sd17.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=3)
]).reset_index().set_index(['T', 'N', 'G','index']).sort_index().to_latex('bias_groups.tex')

pd.concat([rmse(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=1),
rmse(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=1),
rmse(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=1),
rmse(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=1),
rmse(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=1),
rmse(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=1),
rmse(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=1),
rmse(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=1),
rmse(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=1),
rmse(realizations9.iloc[:,:2],sd9.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=3),
rmse(realizations10.iloc[:,:2],sd10.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=3),
rmse(realizations11.iloc[:,:2],sd11.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=3),
rmse(realizations12.iloc[:,:2],sd12.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=3),
rmse(realizations13.iloc[:,:2],sd13.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=3),
rmse(realizations14.iloc[:,:2],sd14.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=3),
rmse(realizations15.iloc[:,:2],sd15.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=3),
rmse(realizations16.iloc[:,:2],sd16.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=3),
rmse(realizations17.iloc[:,:2],sd17.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=3)
]).reset_index().set_index(['T', 'N', 'G','index']).sort_index().to_latex('rmse_groups.tex')

pd.concat([cp(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=1),
cp(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=1),
cp(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=1),
cp(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=1),
cp(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=1),
cp(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=1),
cp(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=1),
cp(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=1),
cp(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=1),
cp(realizations9.iloc[:,:2],sd9.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=3),
cp(realizations10.iloc[:,:2],sd10.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=3),
cp(realizations11.iloc[:,:2],sd11.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=3),
cp(realizations12.iloc[:,:2],sd12.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=3),
cp(realizations13.iloc[:,:2],sd13.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=3),
cp(realizations14.iloc[:,:2],sd14.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=3),
cp(realizations15.iloc[:,:2],sd15.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=3),
cp(realizations16.iloc[:,:2],sd16.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=3),
cp(realizations17.iloc[:,:2],sd17.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=3)
]).reset_index().set_index(['T', 'N', 'G','index']).sort_index().to_latex('cp_groups.tex')