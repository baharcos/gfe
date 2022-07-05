import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns

colors = ['#547482','#C87259','#C2D8C2','#F1B05D','#818662','#6C4A4D','#7A8C87','#EE8445','#C8B05C','#3C2030','#C89D64','#2A3B49']

#When T is fixed to 10 and N increases for \hat{\theta}_1.
fig, ax = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
sns.set_palette(colors)
# T=10, N=50
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations3.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[0,0]
            )
ax[0,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations30.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,1]
            )
ax[0,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations12.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,2]
            )
ax[0,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations21.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,3]
            )
ax[0,3].axvline(0.1, c=colors[1])

# T=10, N=100

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations4.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[1,0]
            )
ax[1,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations31.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,1]
            )
ax[1,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations13.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,2]
            )
ax[1,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations22.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,3]
            )
ax[1,3].axvline(0.1, c=colors[1])


#T=10, N=1000
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations5.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[2,0]
            )
ax[2,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations32.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,1]
            )
ax[2,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations14.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,2]
            )
ax[2,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations23.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,3]
            )
ax[2,3].axvline(0.1, c=colors[1])
for i in range(3):
    for k in range(4):
        ax[i,k].get_legend().remove()
        
ax[0,0].annotate("G=1", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,1].annotate("G=2", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,2].annotate("G=3", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,3].annotate("G=4", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,0].annotate("T=10 , N=50", xy=(0, 0.5), xytext=(-ax[0,0].yaxis.labelpad - 5, 0),
                xycoords=ax[0,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center',rotation=90)
ax[1,0].annotate("T=10, N=100", xy=(0, 0.5), xytext=(-ax[1,0].yaxis.labelpad - 5, 0),
                xycoords=ax[1,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
ax[2,0].annotate("T=10, N=1000", xy=(0, 0.5), xytext=(-ax[2,0].yaxis.labelpad - 5, 0),
                xycoords=ax[2,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
fig.savefig('/Users/baharcoskun/gfe/groupssamplingplotN.png', dpi=400)

#When T is fixed to 10 and N increases for \hat{\theta}_2.
fig, ax = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
sns.set_palette(colors)
# T=10, N=50
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations3.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[0,0]
            )
ax[0,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations30.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,1]
            )
ax[0,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations12.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,2]
            )
ax[0,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations21.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,3]
            )
ax[0,3].axvline(0.5, c=colors[1])

# T=10, N=100

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations4.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[1,0]
            )
ax[1,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations31.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,1]
            )
ax[1,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations13.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,2]
            )
ax[1,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations22.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,3]
            )
ax[1,3].axvline(0.5, c=colors[1])


#T=10, N=1000
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations5.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[2,0]
            )
ax[2,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations32.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,1]
            )
ax[2,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations14.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,2]
            )
ax[2,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations23.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,3]
            )
ax[2,3].axvline(0.5, c=colors[1])
for i in range(3):
    for k in range(4):
        ax[i,k].get_legend().remove()
        
ax[0,0].annotate("G=1", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,1].annotate("G=2", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,2].annotate("G=3", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,3].annotate("G=4", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,0].annotate("T=10 , N=50", xy=(0, 0.5), xytext=(-ax[0,0].yaxis.labelpad - 5, 0),
                xycoords=ax[0,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center',rotation=90)
ax[1,0].annotate("T=10, N=100", xy=(0, 0.5), xytext=(-ax[1,0].yaxis.labelpad - 5, 0),
                xycoords=ax[1,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
ax[2,0].annotate("T=10, N=1000", xy=(0, 0.5), xytext=(-ax[2,0].yaxis.labelpad - 5, 0),
                xycoords=ax[2,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
fig.savefig('/Users/baharcoskun/gfe/groupssamplingplotN.png', dpi=400)

#When T increases and N=100 for \hat{\theta}_1.

fig, ax = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
sns.set_palette(colors)
# T=5, N=100
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations1.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[0,0]
            )
ax[0,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations28.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,1]
            )
ax[0,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations10.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,2]
            )
ax[0,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations19.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,3]
            )
ax[0,3].axvline(0.1, c=colors[1])

# T=10, N=100

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations4.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[1,0]
            )
ax[1,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations31.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,1]
            )
ax[1,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations13.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,2]
            )
ax[1,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations22.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,3]
            )
ax[1,3].axvline(0.1, c=colors[1])


#T=20, N=100
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations7.csv').iloc[:,1:2],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[2,0]
            )
ax[2,0].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations34.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,1]
            )
ax[2,1].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations16.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,2]
            )
ax[2,2].axvline(0.1, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations25.csv').iloc[:,1:2],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,3]
            )
ax[2,3].axvline(0.1, c=colors[1])
for i in range(3):
    for k in range(4):
        ax[i,k].get_legend().remove()
        
ax[0,0].annotate("G=1", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,1].annotate("G=2", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,2].annotate("G=3", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,3].annotate("G=4", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,0].annotate("T=5 , N=100", xy=(0, 0.5), xytext=(-ax[0,0].yaxis.labelpad - 5, 0),
                xycoords=ax[0,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center',rotation=90)
ax[1,0].annotate("T=10, N=100", xy=(0, 0.5), xytext=(-ax[1,0].yaxis.labelpad - 5, 0),
                xycoords=ax[1,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
ax[2,0].annotate("T=20, N=100", xy=(0, 0.5), xytext=(-ax[2,0].yaxis.labelpad - 5, 0),
                xycoords=ax[2,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
fig.savefig('/Users/baharcoskun/gfe/groupssamplingplotT1.png', dpi=400)

#When T increases and N=100 for \hat{\theta}_2.

fig, ax = plt.subplots(3, 4, figsize=(20, 15), sharex=True, sharey=True)
sns.set_palette(colors)
# T=5, N=100
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations1.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[0,0]
            )
ax[0,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations28.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,1]
            )
ax[0,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations10.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,2]
            )
ax[0,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations19.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[0,3]
            )
ax[0,3].axvline(0.5, c=colors[1])

# T=10, N=100

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations4.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[1,0]
            )
ax[1,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations31.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,1]
            )
ax[1,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations13.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,2]
            )
ax[1,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations22.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[1,3]
            )
ax[1,3].axvline(0.5, c=colors[1])


#T=20, N=100
sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations7.csv').iloc[:,2:3],
            bins=40,
            alpha=0.2,
            kde=True,
            ax=ax[2,0]
            )
ax[2,0].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations34.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,1]
            )
ax[2,1].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations16.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,2]
            )
ax[2,2].axvline(0.5, c=colors[1])

sns.histplot(data=pd.read_csv('/Users/baharcoskun/gfe/src/original_data/realizations25.csv').iloc[:,2:3],
            bins=25,
            alpha=0.2,
            kde=True,
            ax=ax[2,3]
            )
ax[2,3].axvline(0.5, c=colors[1])
for i in range(3):
    for k in range(4):
        ax[i,k].get_legend().remove()
        
ax[0,0].annotate("G=1", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,1].annotate("G=2", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,2].annotate("G=3", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,3].annotate("G=4", xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
ax[0,0].annotate("T=5 , N=100", xy=(0, 0.5), xytext=(-ax[0,0].yaxis.labelpad - 5, 0),
                xycoords=ax[0,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center',rotation=90)
ax[1,0].annotate("T=10, N=100", xy=(0, 0.5), xytext=(-ax[1,0].yaxis.labelpad - 5, 0),
                xycoords=ax[1,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
ax[2,0].annotate("T=20, N=100", xy=(0, 0.5), xytext=(-ax[2,0].yaxis.labelpad - 5, 0),
                xycoords=ax[2,0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation=90)
fig.savefig('/Users/baharcoskun/gfe/groupssamplingplotT.png', dpi=400)


