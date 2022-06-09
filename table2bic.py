import numpy as np
import pandas as pd
from bic import gfe_bic
from simulate_bias import *

""" Runs Monte-Carlo Simulations and reports the results: bias, RMSE, CP, BIC 
"nindividuals" = 1000, 100, 50
"nperiods" = 20, 10, 5
"ngroups" = 4, 3, 2, 1 """



def table(realizations, std, true_value, G, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std
    return pd.DataFrame({ 
        'T' : T,
        'N' : N,
        'G': G,
        'bias' : realizations.mean() - true_value,
        'rmse': np.sqrt(((realizations - true_value)**2).mean()),
        'coverage probability' : (((ci_low<=true_value)&(ci_up>=true_value))*1).mean()
        })

def length_ci(realizations, std, true_value, G, N, T, critical_value=1.96):
    ci_low = realizations - critical_value*std
    ci_up = realizations + critical_value*std

    return pd.DataFrame({
        'T' : T,
        'N' : N,
        'G': G,
        'Expected Value' : realizations.mean(),
        'average length CI' : (ci_up - ci_low).mean()
        })

def compare_bic(a,b,c,d, G, T, N):
    return pd.DataFrame({
        'Selection rate': (((pd.DataFrame(a) < pd.DataFrame(b)) & (pd.DataFrame(a) < pd.DataFrame(c)) & (pd.DataFrame(a) < pd.DataFrame(d)))*1).mean(),
        'T':T,
        'N': N,
        'G': G
    })
    


"""DGP"""


params = {"nfeatures" : 2,
          "ngroups" : 2,
          "theta" : np.array([0.1, 0.5]),
          "theta_0" : np.array([0.2, 0.4]),
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

# def results(nindividuals,
#     nfeatures,
#     ngroups,
#     specified_ngroups,
#     nperiods,
#     theta,
#     alpha,
#     theta_0,
#     alpha_0,
#     low,
#     up,
#     nreps,
#     seed
# ):
# estimates, bic = monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=5, alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2, specified_ngroups=1)

# realizations = pd.DataFrame(estimates['Estimates'])
# sd  = pd.DataFrame(estimates['sd'])

""""
G = 1 """


#short panel, few observations
estimates, bic = monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=5, alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2, specified_ngroups=1)

realizations = pd.DataFrame(estimates['Estimates'])
sd  = pd.DataFrame(estimates['sd'])
realizations.to_csv('gfe/src/original_data/realizations.csv')

estimates1, bic1 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=5,alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2,specified_ngroups=1)

realizations1 = pd.DataFrame(estimates1['Estimates'])
std1  = pd.DataFrame(estimates1['sd'])
realizations1.to_csv('gfe/src/original_data/realizations1.csv')


estimates2, bic2 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=5,alpha=alpha(5), alpha_0=(alpha(5)[:5] + alpha(5)[5:10])/2, specified_ngroups=1)

realizations2 = pd.DataFrame(estimates2['Estimates'])
std2  = pd.DataFrame(estimates2['sd'])
realizations2.to_csv('gfe/src/original_data/realizations2.csv')

estimates3, bic3 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations3 = pd.DataFrame(estimates3['Estimates'])#,  columns = columns2x10)
std3  = pd.DataFrame(estimates3['sd'])#,  columns = columns2x10)
realizations3.to_csv('gfe/src/original_data/realizations3.csv')

estimates4, bic4 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations4 = pd.DataFrame(estimates4['Estimates'])#,  columns = columns2x10)
std4  = pd.DataFrame(estimates4['sd'])#,  columns = columns2x10)

realizations4.to_csv('gfe/src/original_data/realizations4.csv')

estimates5, bic5 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=10,alpha=alpha(10), alpha_0=(alpha(10)[:10] + alpha(10)[10:20])/2, specified_ngroups=1)

realizations5 = pd.DataFrame(estimates5['Estimates'])#,  columns = columns2x10)
std5  = pd.DataFrame(estimates5['sd'])#,  columns = columns2x10)
realizations5.to_csv('gfe/src/original_data/realizations5.csv')

estimates6, bic6 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations6 = pd.DataFrame(estimates6['Estimates'])
std6  = pd.DataFrame(estimates6['sd'])
realizations6.to_csv('gfe/src/original_data/realizations6.csv')

estimates7, bic7 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations7 = pd.DataFrame(estimates7['Estimates'])
std7  = pd.DataFrame(estimates7['sd'])

realizations7.to_csv('gfe/src/original_data/realizations7.csv')


estimates8, bic8 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=20,alpha=alpha(20), alpha_0=(alpha(20)[:20] + alpha(20)[20:40])/2, specified_ngroups=1)

realizations8 = pd.DataFrame(estimates8['Estimates'])
std8  = pd.DataFrame(estimates8['sd'])

realizations8.to_csv('gfe/src/original_data/realizations8.csv')
#longer panel many obs grouping improves a lot with t and it does not have to be too long

"""
G=2"""

estimates27, bic27 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=5, alpha=alpha(5), alpha_0=alpha(5), specified_ngroups=2)

realizations27 = pd.DataFrame(estimates['Estimates'])
sd27  = pd.DataFrame(estimates27['sd'])
realizations27.to_csv('gfe/src/original_data/realizations27.csv')

estimates28, bic28 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=5,alpha=alpha(5), alpha_0=alpha(5),specified_ngroups=2)

realizations28 = pd.DataFrame(estimates28['Estimates'])
sd28  = pd.DataFrame(estimates28['sd'])
realizations28.to_csv('gfe/src/original_data/realizations28.csv')
estimates29, bic29 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=5,alpha=alpha(5), alpha_0=alpha(5), specified_ngroups=2)

realizations29 = pd.DataFrame(estimates29['Estimates'])
sd29  = pd.DataFrame(estimates29['sd'])
realizations29.to_csv('gfe/src/original_data/realizations29.csv')

estimates30, bic30 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations30 = pd.DataFrame(estimates30['Estimates'])
sd30  = pd.DataFrame(estimates30['sd'])
realizations30.to_csv('gfe/src/original_data/realizations30.csv')

estimates31, bic31 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations31 = pd.DataFrame(estimates31['Estimates'])
sd31  = pd.DataFrame(estimates31['sd'])
realizations31.to_csv('gfe/src/original_data/realizations31.csv')

estimates32, bic32 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=10,alpha=alpha(10), alpha_0=alpha(10), specified_ngroups=2)

realizations32 = pd.DataFrame(estimates32['Estimates'])
sd32  = pd.DataFrame(estimates32['sd'])
realizations32.to_csv('gfe/src/original_data/realizations32.csv')

estimates33, bic33 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations33 = pd.DataFrame(estimates33['Estimates'])
sd33  = pd.DataFrame(estimates33['sd'])
realizations33.to_csv('gfe/src/original_data/realizations33.csv')

estimates34, bic34 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations34 = pd.DataFrame(estimates34['Estimates'])
sd34  = pd.DataFrame(estimates34['sd'])
realizations34.to_csv('gfe/src/original_data/realizations34.csv')
estimates35, bic35 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=20,alpha=alpha(20), alpha_0=alpha(20), specified_ngroups=2)

realizations35 = pd.DataFrame(estimates35['Estimates'])
sd35  = pd.DataFrame(estimates35['sd'])
realizations35.to_csv('gfe/src/original_data/realizations35.csv')
""""
G = 3 """

estimates9, bic9 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=5, 
alpha=alpha(5), alpha_0= np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)), specified_ngroups=3)

realizations9 = pd.DataFrame(estimates9['Estimates'])#,  columns = columns2x5)
sd9  = pd.DataFrame(estimates9['sd'])#,  columns = columns2x5)
realizations9.to_csv('gfe/src/original_data/realizations9.csv')

estimates10, bic10 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=5,
alpha=alpha(5), alpha_0=np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)),specified_ngroups=3)

realizations10 = pd.DataFrame(estimates10['Estimates'])#,  columns = columns2x5)
sd10  = pd.DataFrame(estimates10['sd'])#,  columns = columns2x5)
realizations10.to_csv('gfe/src/original_data/realizations10.csv')
estimates11, bic11 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=5,
alpha=alpha(5), alpha_0=np.append(alpha(5),((alpha(5)[:5] + alpha(5)[5:10])/2)), specified_ngroups=3)

realizations11 = pd.DataFrame(estimates11['Estimates'])#,  columns = columns2x5)
sd11  = pd.DataFrame(estimates11['sd'])#,  columns = columns2x5)
realizations11.to_csv('gfe/src/original_data/realizations11.csv')

estimates12, bic12 =  monte_carlo_simulation_groups (**params, nindividuals=50, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),(alpha(10)[:10]+alpha(10)[10:])-1), specified_ngroups=3)

realizations12 = pd.DataFrame(estimates12['Estimates'])
sd12  = pd.DataFrame(estimates12['sd'])
realizations12.to_csv('gfe/src/original_data/realizations12.csv')

estimates13, bic13 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),(alpha(10)[:10]+alpha(10)[10:])-1), specified_ngroups=3)

realizations13 = pd.DataFrame(estimates13['Estimates'])
sd13  = pd.DataFrame(estimates13['sd'])
realizations13.to_csv('gfe/src/original_data/realizations13.csv')

estimates14, bic14 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=10,
alpha=alpha(10), alpha_0=np.append(alpha(10),(alpha(10)[:10]+alpha(10)[10:])-1), specified_ngroups=3)

realizations14 = pd.DataFrame(estimates14['Estimates'])
sd14  = pd.DataFrame(estimates14['sd'])
realizations14.to_csv('gfe/src/original_data/realizations14.csv')

estimates15, bic15 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=20,
alpha=alpha(20), alpha_0=np.append(np.append(((alpha(20)[:20]+alpha(20)[20:])-2), (alpha(20)[:20]-0.5)), (alpha(20)[:20]+0.5)), specified_ngroups=3)

realizations15 = pd.DataFrame(estimates15['Estimates'])
sd15  = pd.DataFrame(estimates15['sd'])
realizations15.to_csv('gfe/src/original_data/realizations15.csv')

estimates16, bic16 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=20,
alpha=alpha(20), alpha_0=np.append(np.append(((alpha(20)[:20]+alpha(20)[20:])-2), (alpha(20)[:20]-0.5)), (alpha(20)[:20]+0.5)), specified_ngroups=3)

realizations16 = pd.DataFrame(estimates16['Estimates'])
sd16  = pd.DataFrame(estimates16['sd'])
realizations16.to_csv('gfe/src/original_data/realizations16.csv')
estimates17, bic17 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=20,
alpha=alpha(20), alpha_0=np.append(np.append(((alpha(20)[:20]+alpha(20)[20:])-2), (alpha(20)[:20]-0.5)), (alpha(20)[:20]+0.5)), specified_ngroups=3)

realizations17 = pd.DataFrame(estimates17['Estimates'])
sd17  = pd.DataFrame(estimates17['sd'])
realizations17.to_csv('gfe/src/original_data/realizations17.csv')
""""
G = 4 """

estimates18, bic18 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=5,alpha=alpha(5), 
alpha_0=np.append(np.append((alpha(5)[:5]+0.5),(alpha(5)[:5]-0.5)), np.append((alpha(5)[5:]+0.5),(alpha(5)[5:]-0.5))),
specified_ngroups=4)

realizations18 = pd.DataFrame(estimates18['Estimates'])
sd18  = pd.DataFrame(estimates18['sd'])
realizations18.to_csv('gfe/src/original_data/realizations18.csv')


estimates19, bic19 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=5, 
alpha=alpha(5), 
alpha_0= np.append(np.append((alpha(5)[:5]+0.5),(alpha(5)[:5]-0.5)), np.append((alpha(5)[5:]+0.5),(alpha(5)[5:]-0.5))), specified_ngroups=4)

realizations19 = pd.DataFrame(estimates19['Estimates'])#,  columns = columns2x5)
sd19  = pd.DataFrame(estimates19['sd'])#,  columns = columns2x5)
realizations19.to_csv('gfe/src/original_data/realizations19.csv')

estimates20, bic20 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=5,
alpha=alpha(5), 
alpha_0=np.append(np.append((alpha(5)[:5]+0.5),(alpha(5)[:5]-0.5)), np.append((alpha(5)[5:]+0.5),(alpha(5)[5:]-0.5))),specified_ngroups=4)

realizations20 = pd.DataFrame(estimates20['Estimates'])#,  columns = columns2x5)
sd20  = pd.DataFrame(estimates20['sd'])#,  columns = columns2x5)
realizations20.to_csv('gfe/src/original_data/realizations20.csv')
estimates21, bic21 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=10,
alpha=alpha(10), alpha_0=np.append(np.append((alpha(10)[:10]+0.5),(alpha(10)[:10]-0.5)), np.append((alpha(10)[10:]+0.5),(alpha(10)[10:]-0.5))), 
specified_ngroups=4)

realizations21 = pd.DataFrame(estimates21['Estimates'])#,  columns = columns2x5)
sd21  = pd.DataFrame(estimates21['sd'])#,  columns = columns2x5)
realizations21.to_csv('gfe/src/original_data/realizations21.csv')

estimates22, bic22 =  monte_carlo_simulation_groups (**params, nindividuals=100, nperiods=10,
alpha=alpha(10), alpha_0=np.append(np.append((alpha(10)[:10]+0.5),(alpha(10)[:10]-0.5)), np.append((alpha(10)[10:]+0.5),(alpha(10)[10:]-0.5))),
 specified_ngroups=4)

realizations22 = pd.DataFrame(estimates22['Estimates'])#,  columns = columns2x10)
sd22  = pd.DataFrame(estimates22['sd'])#,  columns = columns2x10)
realizations22.to_csv('gfe/src/original_data/realizations22.csv')

estimates23, bic23 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=10,
alpha=alpha(10), alpha_0=np.append(np.append((alpha(10)[:10]+0.5),(alpha(10)[:10]-0.5)), np.append((alpha(10)[10:]+0.5),(alpha(10)[10:]-0.5))), specified_ngroups=4)

realizations23 = pd.DataFrame(estimates23['Estimates'])#,  columns = columns2x10)
sd23  = pd.DataFrame(estimates23['sd'])#,  columns = columns2x10)
realizations23.to_csv('gfe/src/original_data/realizations23.csv')

estimates24, bic24 =  monte_carlo_simulation_groups(**params, nindividuals=50, nperiods=20,
alpha=alpha(20), alpha_0=np.append(np.append((alpha(20)[:20]+0.5),(alpha(20)[:20]-0.5)), np.append((alpha(20)[20:]+0.5),(alpha(20)[20:]-0.5))),
 specified_ngroups=4)

realizations24 = pd.DataFrame(estimates24['Estimates'])#,  columns = columns2x10)
sd24  = pd.DataFrame(estimates24['sd'])#,  columns = columns2x10)
realizations24.to_csv('gfe/src/original_data/realizations24.csv')

estimates25, bic25 =  monte_carlo_simulation_groups(**params, nindividuals=100, nperiods=20,
alpha=alpha(20), alpha_0=np.append(np.append((alpha(20)[:20]+0.5),(alpha(20)[:20]-0.5)), np.append((alpha(20)[20:]+0.5),(alpha(20)[20:]-0.5))), 
specified_ngroups=4)

realizations25 = pd.DataFrame(estimates25['Estimates'])
sd25  = pd.DataFrame(estimates25['sd'])
realizations25.to_csv('gfe/src/original_data/realizations25.csv')

estimates26, bic26 =  monte_carlo_simulation_groups(**params, nindividuals=1000, nperiods=20,
alpha=alpha(20), alpha_0= np.append(np.append((alpha(20)[:20]+0.5),(alpha(20)[:20]-0.5)), np.append((alpha(20)[20:]+0.5),(alpha(20)[20:]-0.5)))
, specified_ngroups=4)

realizations26 = pd.DataFrame(estimates26['Estimates'])
sd26  = pd.DataFrame(estimates26['sd'])
realizations26.to_csv('gfe/src/original_data/realizations26.csv')
pd.concat([table(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=1),
table(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=1),
table(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=1),
table(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=1),
table(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=1),
table(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=1),
table(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=1),
table(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=1),
table(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=1),
table(realizations9.iloc[:,:2],sd9.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=3),
table(realizations10.iloc[:,:2],sd10.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=3),
table(realizations11.iloc[:,:2],sd11.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=3),
table(realizations12.iloc[:,:2],sd12.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=3),
table(realizations13.iloc[:,:2],sd13.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=3),
table(realizations14.iloc[:,:2],sd14.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=3),
table(realizations15.iloc[:,:2],sd15.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=3),
table(realizations16.iloc[:,:2],sd16.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=3),
table(realizations17.iloc[:,:2],sd17.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=3),
table(realizations18.iloc[:,:2],sd18.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=4),
table(realizations19.iloc[:,:2],sd19.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=4),
table(realizations20.iloc[:,:2],sd20.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=4),
table(realizations21.iloc[:,:2],sd21.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=4),
table(realizations22.iloc[:,:2],sd22.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=4),
table(realizations23.iloc[:,:2],sd23.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=4),
table(realizations24.iloc[:,:2],sd24.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=4),
table(realizations25.iloc[:,:2],sd25.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=4),
table(realizations26.iloc[:,:2],sd26.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=4),
table(realizations27.iloc[:,:2],sd27.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=2),
table(realizations28.iloc[:,:2],sd28.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=2),
table(realizations29.iloc[:,:2],sd29.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=2),
table(realizations30.iloc[:,:2],sd30.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=2),
table(realizations31.iloc[:,:2],sd31.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=2),
table(realizations32.iloc[:,:2],sd32.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=2),
table(realizations33.iloc[:,:2],sd33.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=2),
table(realizations34.iloc[:,:2],sd34.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=2),
table(realizations35.iloc[:,:2],sd35.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=2)
]).reset_index().set_index(['T', 'N', 'G','index']).sort_index().to_latex('table_groups.tex')

ci = pd.concat([length_ci(realizations.iloc[:,:2],sd.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=1),
length_ci(realizations1.iloc[:,:2],std1.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=1),
length_ci(realizations2.iloc[:,:2],std2.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=1),
length_ci(realizations3.iloc[:,:2],std3.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=1),
length_ci(realizations4.iloc[:,:2],std4.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=1),
length_ci(realizations5.iloc[:,:2],std5.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=1),
length_ci(realizations6.iloc[:,:2],std6.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=1),
length_ci(realizations7.iloc[:,:2],std7.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=1),
length_ci(realizations8.iloc[:,:2],std8.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=1),
length_ci(realizations9.iloc[:,:2],sd9.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=3),
length_ci(realizations10.iloc[:,:2],sd10.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=3),
length_ci(realizations11.iloc[:,:2],sd11.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=3),
length_ci(realizations12.iloc[:,:2],sd12.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=3),
length_ci(realizations13.iloc[:,:2],sd13.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=3),
length_ci(realizations14.iloc[:,:2],sd14.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=3),
length_ci(realizations15.iloc[:,:2],sd15.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=3),
length_ci(realizations16.iloc[:,:2],sd16.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=3),
length_ci(realizations17.iloc[:,:2],sd17.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=3),
length_ci(realizations18.iloc[:,:2],sd18.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=4),
length_ci(realizations19.iloc[:,:2],sd19.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=4),
length_ci(realizations20.iloc[:,:2],sd20.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=4),
length_ci(realizations21.iloc[:,:2],sd21.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=4),
length_ci(realizations22.iloc[:,:2],sd22.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=4),
length_ci(realizations23.iloc[:,:2],sd23.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=4),
length_ci(realizations24.iloc[:,:2],sd24.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=4),
length_ci(realizations25.iloc[:,:2],sd25.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=4),
length_ci(realizations26.iloc[:,:2],sd26.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=4),
length_ci(realizations27.iloc[:,:2],sd27.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=5, G=2),
length_ci(realizations28.iloc[:,:2],sd28.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=5, G=2),
length_ci(realizations29.iloc[:,:2],sd29.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=5, G=2),
length_ci(realizations30.iloc[:,:2],sd30.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=10, G=2),
length_ci(realizations31.iloc[:,:2],sd31.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=10, G=2),
length_ci(realizations32.iloc[:,:2],sd32.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=10, G=2),
length_ci(realizations33.iloc[:,:2],sd33.iloc[:,:2], true_value= [0.1, 0.5], N=50, T=20, G=2),
length_ci(realizations34.iloc[:,:2],sd34.iloc[:,:2], true_value= [0.1, 0.5], N=100, T=20, G=2),
length_ci(realizations35.iloc[:,:2],sd35.iloc[:,:2], true_value= [0.1, 0.5], N=1000, T=20, G=2)
]).reset_index().set_index(['T', 'N', 'G','index']).sort_index()

ci.to_csv('ci_groups.csv')
ci.to_latex('ci_groups.tex')

"""Compare BIC"""
#Need to write a table function

# G = 1: 0...8 
bic_table = pd.concat([compare_bic(bic,bic9, bic18, bic27, N=50, T=5, G=1),
compare_bic(bic1,bic10, bic19, bic28, N=100, T=5, G=1),
compare_bic(bic2,bic11, bic20, bic29, N=1000, T=5, G=1),
compare_bic(bic3,bic12, bic21, bic30, N=50, T=10, G=1),
compare_bic(bic4,bic13, bic22, bic31, N=100, T=10, G=1),
compare_bic(bic5,bic14, bic23, bic32, N=1000, T=10, G=1),
compare_bic(bic6,bic15, bic24, bic33, N=50, T=20, G=1),
compare_bic(bic7,bic16, bic25, bic34, N=100, T=20, G=1),
compare_bic(bic8,bic17, bic26, bic35, N=1000, T=20, G=1),
compare_bic(bic9,bic, bic18, bic27, N=50, T=5, G=3),
compare_bic(bic10,bic1, bic19, bic28, N=100, T=5, G=3),
compare_bic(bic11,bic2, bic20, bic29, N=1000, T=5, G=3),
compare_bic(bic12,bic3, bic21, bic30, N=50, T=10, G=3),
compare_bic(bic13,bic4, bic22, bic31, N=100, T=10, G=3),
compare_bic(bic14,bic5, bic23, bic32, N=1000, T=10, G=3),
compare_bic(bic15,bic6, bic24, bic33, N=50, T=20, G=3),
compare_bic(bic16,bic7, bic25, bic34, N=100, T=20, G=3),
compare_bic(bic17,bic8, bic26, bic35, N=1000, T=20, G=3),
compare_bic(bic18,bic9, bic, bic27, N=50, T=5, G=4),
compare_bic(bic19,bic10, bic1, bic28, N=100, T=5, G=4),
compare_bic(bic20,bic11, bic2, bic29, N=1000, T=5, G=4),
compare_bic(bic21,bic12, bic3, bic30, N=50, T=10, G=4),
compare_bic(bic22,bic13, bic4, bic31, N=100, T=10, G=4),
compare_bic(bic23,bic14, bic5, bic32, N=1000, T=10, G=4),
compare_bic(bic24,bic15, bic6, bic33, N=50, T=20, G=4),
compare_bic(bic25,bic16, bic7, bic34, N=100, T=20, G=4),
compare_bic(bic26,bic17, bic8, bic35, N=1000, T=20, G=4),
compare_bic(bic27,bic9, bic18, bic, N=50, T=5, G=2),
compare_bic(bic28,bic10, bic19, bic1, N=100, T=5, G=2),
compare_bic(bic29,bic11, bic20, bic2, N=1000, T=5, G=2),
compare_bic(bic30,bic12, bic21, bic3, N=50, T=10, G=2),
compare_bic(bic31,bic13, bic22, bic4, N=100, T=10, G=2),
compare_bic(bic32,bic14, bic23, bic5, N=1000, T=10, G=2),
compare_bic(bic33,bic15, bic24, bic6, N=50, T=20, G=2),
compare_bic(bic34,bic16, bic25, bic7, N=100, T=20, G=2),
compare_bic(bic35,bic17, bic26, bic8, N=1000, T=20, G=2)
]).set_index(['T', 'N', 'G']).sort_index().to_latex('bic_groups.tex')