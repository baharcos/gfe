import numpy as np
import pandas as pd

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

coef1=pd.read_csv("/Users/baharcoskun/gfe/ife/coef1.csv")
sd1 = pd.read_csv("/Users/baharcoskun/gfe/ife/se1.csv")

coef2=pd.read_csv("/Users/baharcoskun/gfe/ife/coef2.csv")
sd2 = pd.read_csv("/Users/baharcoskun/gfe/ife/se2.csv")

coef3=pd.read_csv("/Users/baharcoskun/gfe/ife/coef3.csv")
sd3 = pd.read_csv("/Users/baharcoskun/gfe/ife/se3.csv")

coef4=pd.read_csv("/Users/baharcoskun/gfe/ife/coef4.csv")
sd4 = pd.read_csv("/Users/baharcoskun/gfe/ife/se4.csv")

coef5=pd.read_csv("/Users/baharcoskun/gfe/ife/coef5.csv")
sd5 = pd.read_csv("/Users/baharcoskun/gfe/ife/se5.csv")

coef6=pd.read_csv("/Users/baharcoskun/gfe/ife/coef6.csv")
sd6 = pd.read_csv("/Users/baharcoskun/gfe/ife/se6.csv")

coef7=pd.read_csv("/Users/baharcoskun/gfe/ife/coef7.csv")
sd7 = pd.read_csv("/Users/baharcoskun/gfe/ife/se7.csv")

coef8=pd.read_csv("/Users/baharcoskun/gfe/ife/coef8.csv")
sd8 = pd.read_csv("/Users/baharcoskun/gfe/ife/se8.csv")

coef8=pd.read_csv("/Users/baharcoskun/gfe/ife/coef8.csv")
sd8 = pd.read_csv("/Users/baharcoskun/gfe/ife/se8.csv")

coef9=pd.read_csv("/Users/baharcoskun/gfe/ife/coef9.csv")
sd9 = pd.read_csv("/Users/baharcoskun/gfe/ife/se9.csv")

IFE = pd.concat(table(coef1, sd1, [0.1,0.5], "IFE", 50, 5),
table(coef2, sd2, [0.1,0.5], "IFE", 100, 5),
table(coef3, sd3, [0.1,0.5], "IFE", 1000, 5),
table(coef4, sd4, [0.1,0.5], "IFE", 50, 10),
table(coef5, sd5, [0.1,0.5], "IFE", 100, 10),
table(coef6, sd6, [0.1,0.5], "IFE", 1000, 10),
table(coef7, sd7, [0.1,0.5], "IFE", 50, 20),
table(coef8, sd8, [0.1,0.5], "IFE", 100, 20),
table(coef9, sd9, [0.1,0.5], "IFE", 1000, 20)).to_latex('table_ife.tex')

