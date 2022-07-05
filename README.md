# gfe
---
This repository contains the code I used in my master thesis. The LaTex code of the paper can be found in the paper folder, my implementation of the Algorithm 1 in 
Bonhomme and Manresa (2015), and the code to create all the figures and tables in the paper can be found in the src folder. The notebooks folder contains the IFE
implementation using a package from julia in python, MC_Simulation which tests the algorithm when both the number of individuals(up to 10000) and the periods(up to 900) increases, sensitivity_modelselection_plots contain sensitivity
checks, migration application contains the application GFE to Angrist and Kugler (2003). The results of GFE is similar to their instrumental regression results.
However, the results are far from conclusive without further checks which the time limit on this paper has not allowed me. That is why it is not included in the 
paper. The simulation results of the IFE and GFE estimates and their standard deviation are in their respective folder. 

The pdf of the paper contains the appendix with additional figures.

## Credits
 I used statsmodels v0.13.0 for OLS results, linearmodels.panel.model.PanelOLS v4.25 for FE and TWFE results, InteractiveFixedEffectModels.jl v1.1.6 for IFE results,
 my own implementation for GFE results. 
 
 I made use of numpy and pandas libraries throughout the project. Figures are created using matplotlib.pyplot and seaborn libraries.
 
 All are python packages except for InteractiveFixedEffectModels.jl which is in julia.
