# %%

"""
Code for computing the amount of shared surface area between two neighborhoods
'cells2': Main dataframe output by neighborhoodtemplate with x,y,clusterID, and neighborhood allocation
"""

# %%

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt

cells2 = pd.read_csv('data/CRC_pathml_Neighbors.csv')

# %%

tissue_col = 'Region'
neigh_col = 'neighborhood10'
patient_col = 'patients'
group_col = 'groups'
X = 'x'
Y = 'y'

# %%

# calculate neighbors for each spot
for spot in cells2[tissue_col].unique():
    tissue = cells2[cells2[tissue_col] == spot]
    fit = NearestNeighbors(n_neighbors=1).fit(tissue[[X, Y]].values)
    m = fit.kneighbors()[1]
    cells2.loc[tissue.index, 'neigh_neigh'] = tissue.iloc[m[:, 0], :][neigh_col].values

cells2['neigh_neigh'] = cells2['neigh_neigh'].astype(int)

# %%

# compute for each patient, in each tissue and neighborhood, the number of cells in that neighborhoood
counts = cells2.groupby([group_col, patient_col, tissue_col, neigh_col]).apply(lambda x: len(x)).unstack()

# compute for each patient, in each tissue and neighborhood:  the count of how many of the cells in that neighborhood are next to a cell in the other neighborhood
neighs = cells2.groupby([group_col, patient_col, tissue_col, neigh_col]).apply(
    lambda x: x['neigh_neigh'].value_counts(sort=False)).unstack()

# %%

# specify which neighborhoods you want to calculate
neigh1, neigh2 = 0, 4

# %%

# Comment out if you wish to average each spot for each patient
N = neighs.sum(level=[group_col, patient_col, neigh_col])
N[tissue_col] = [i[1] for i in N.index]
neighs = N.set_index(tissue_col, append=True).reorder_levels([group_col, patient_col, tissue_col, neigh_col])

# %%

# index the values for the neighborhoods of interest
ix = pd.IndexSlice

# take the mean between the number of cells in neigh 1 that touch neigh4 and the number of cells in neigh4 that touch neigh1
# as not necessarily symmetric
inters = pd.concat([neighs.loc[ix[:, :, :, [neigh1]], [neigh2]], neighs.loc[ix[:, :, :, [neigh2]], [neigh1]]], 1).mean(
    level=[group_col, patient_col, tissue_col]).mean(1)

# calculate the total number of cells in both neighborhoods
wholes = neighs.sum(1).loc[ix[:, :, :, [neigh2, neigh1]]].unstack().sum(1)

combo = pd.concat([inters, wholes], 1).dropna((0))
combo['ratio'] = combo[0] / combo[1]
combo = combo.rename(columns={0: 'neighboring', 1: 'union', 2: 'ratio'})

combo = combo.reset_index()

# %%

c = combo.reset_index()
sns.stripplot(data=c, x=group_col, y='ratio')
plt.show()

# %%

a = c[c[group_col] == 'CLR']['ratio']

# %%

b = c[c[group_col] == 'DII']['ratio']

# %%

from scipy import stats

stats.ttest_ind(a, b)

# %%


