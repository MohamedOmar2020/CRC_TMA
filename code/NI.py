

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns

# Window: the 20 nearest spatial neighberous of a cell
# as measured by Euclidean distance between X/Y coordinates.


def get_windows(job, n_neighbors):
    '''
    For each region and each individual cell in dataset, return the indices of the nearest neighbors.

    'job:  meta data containing the start time,index of region, region name, indices of region in original dataframe
    n_neighbors:  the number of neighbors to find for each cell
    '''
    start_time, idx, tissue_name, indices = job
    job_start = time.time()

    # exprs: a list with unique tissue region names (140 elements)
    print("Starting:", str(idx + 1) + '/' + str(len(exps)), ': ' + exps[idx])

    # tissue_group: a grouped data frame with X and Y coordinates grouped by unique tissue regions
    # The function get_group(): i guess will construct a dataframe based on tissue_name: like using filter in dplyr
    # eg, SS = tissue_group.get_group("reg001_A") > will output a dataframe for reg001_A tissue region
    tissue = tissue_group.get_group(tissue_name)

    # .loc: Access a group of rows and columns by label(s) or a boolean array.
    # this code accesses the X and Y values from each row (each cell) belonging to a certain tissue
    # eg, SS.loc[1][['X:X','Y:Y']].values
    # Example:
    # II = SS.index
    # TF = SS.loc[II][['X:X', 'Y:Y']].values > returns a dataframe with X:X and Y:Y values (2 columns 0 and 1 which is X:X and Y:Y)
    to_fit = tissue.loc[indices][[X, Y]].values

    # Unsupervised learner for implementing neighbor searches.
    # Example: FF = NearestNeighbors(n_neighbors=20).fit(SS[['X:X', 'Y:Y']].values)
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X, Y]].values)

    # Find the nearest neighbors
    # Example: MM = FF.kneighbors(TF)
    # Returns 2 arrays (0 and 1), each is n_queries X n_neighberos dimensions
    # so 1164 (the nrow of tissue or SS/ number of cells) X 20 (the 20 neigherest points or cells)
    m = fit.kneighbors(to_fit)

    # MM = MM[0], MM[1] >> no change
    m = m[0], m[1]

    ## sort_neighbors
    # argsort: Returns the indices that would sort an array
    # Example: ARGS = MM[0].argsort(axis = 1)
    args = m[0].argsort(axis=1)

    # m[1].shape[0]: nrows of m[1]
    # m[1].shape[1]: ncol of m[1]
    # I guess here is taking the row numbers 0:1164 (np.arange(MM[1].shape[0])) then multiplying it by 20 (* MM[1].shape[1])
    # Example: ADD =  np.arange(MM[1].shape[0]) * MM[1].shape[1] > an array of 1 row (nrow * 20) and 1164 columns
    add = np.arange(m[1].shape[0]) * m[1].shape[1]

    # flatten(): collapses the array into 1 dimenstion
    # SI2 = MM[1].flatten()[ARGS + ADD[:, None]]
    sorted_indices = m[1].flatten()[args + add[:, None]]

    # NN = SS.index.values[SI]
    neighbors = tissue.index.values[sorted_indices]

    end_time = time.time()

    print("Finishing:", str(idx + 1) + "/" + str(len(exps)), ": " + exps[idx], end_time - job_start,
          end_time - start_time)
    return neighbors.astype(np.int32)

######################################################################
ks = [5,10,20] # k=5 means it collects 5 nearest neighbors for each center cell
path_to_data = './data/CRC_pathml.csv'
X = 'x'
Y = 'y'
reg = 'Region'
file_type = 'csv'

cluster_col = 'cell_types'
keep_cols = [X,Y,reg,cluster_col]
save_path = ''

#read in data and do some quick data rearrangement
n_neighbors = max(ks)
assert (file_type=='csv' or file_type =='pickle') #


if file_type == 'pickle':
    cells = pd.read_pickle(path_to_data)

# Read the file
if file_type == 'csv':
    cells = pd.read_csv(path_to_data)

##########################################################
# Add dummy variables for cell types
#Dummy = pd.get_dummies(cells["ClusterName"])

# Connect them together (like cbind in R i guess) // The axis arguemnt (0: index or 1: columns): The axis to concatenate along.
#Connect = pd.concat([cells, Dummy],axis=1)

# Original code (all together)
cells = pd.concat([cells,pd.get_dummies(cells[cluster_col])], axis = 1)
cells.shape

#cells = cells.dropna(axis=0, how="any")

#cells = cells.reset_index() #Uncomment this line if you do any subsetting of dataframe such as removing dirt etc or will throw error at end of next next code block (cell 6)

# Extract the cell types with dummy variables
sum_cols = cells[cluster_col].unique()
#sum_cols = sum_cols[0:23]
values = cells[sum_cols].values


####################################################
## find windows for each cell in each tissue region

# Keep the X and Y coordianates + the tissue regions >> then group by tissue regions (140 unique regions)
tissue_group = cells[[X,Y,reg]].groupby(reg)

# Take a look at the grouped data frame : Basically 140 dataframes (one for each region) with 3 columns: XX, YY, and region
# tissue_group.apply(print)

# Create a list of unique tissue regions
exps = list(cells[reg].unique())

# time.time(): current time is seconds !
# indices: a list of indices (rownames) of each dataframe in tissue_group
# exps.index(t) : t represents the index of each one of the indices eg, exps.index("reg001_A") is 0 and exps.index("reg001_B") is 1 and so on
# t is the name of tissue regions eg, reg001_A
tissue_chunks = [(time.time(),exps.index(t),t,a) for t,indices in tissue_group.groups.items() for a in np.array_split(indices,1)]

# View a tissue chunk (the job described in the first function):
# consists of 4 items:
# 1. the start time
# 2. the index eg, 0,1,2, etc
# 3. the name of tissue region eg, reg001_A
# 4. int64Index : mmutable sequence used for indexing and alignment. The basic object storing axis labels for all pandas objects. Int64Index is a special case of Index with purely integer labels. .
# 4. indices of region in original dataframe

#DD = tissue_chunks[1]
#SS = DD[3]

# Get the window (the 20 closest cells to each cell in each tissue region)
# Returns a list of 140 items (tissue regions), each is an array of cells X neighbors (20) and the values are the indices of these neighbors
tissues = [get_windows(job,n_neighbors) for job in tissue_chunks]

len(tissues)
TT = tissues[0]
TT.shape

#######################################################
# for each cell and its nearest neighbors, reshape and count the number of each cell type in those neighbors.
out_dict = {}
for k in ks:
    for neighbors, job in zip(tissues, tissue_chunks):
        chunk = np.arange(len(neighbors))  # indices
        tissue_name = job[2]
        indices = job[3]
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

# concatenate the summed windows and combine into one dataframe for each window size tested.
windows = {}
for k in ks:
    window = pd.concat(
        [pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1].astype(int), columns=sum_cols) for exp in
         exps], 0)
    window = window.loc[cells.index.values]
    window = pd.concat([cells[keep_cols], window], 1)
    windows[k] = window

############################
k = 10
n_neighborhoods = 10
neighborhood_name = "neighborhood"+str(k)
k_centroids = {}

windows2 = windows[10]
# windows2[cluster_col] = cells[cluster_col]

# Clustering the windows
km = MiniBatchKMeans(n_clusters = n_neighborhoods,random_state=0)

labelskm = km.fit_predict(windows2[sum_cols].values)
k_centroids[k] = km.cluster_centers_
cells['neighborhood10'] = labelskm
cells[neighborhood_name] = cells[neighborhood_name].astype('category')
#['reg064_A','reg066_A','reg018_B','reg023_A']

cell_order = ['CD45+CD4+ T cells', 'CD4+CD45RO+ T cells',
       'CD4+GATA3+ T cells', 'Naive/memory T cells',
       'TILs/TAMs', 'TILs', 'Tregs',
       'granulocytes', 'CD11b+ monocytes',
       'CD68+CD163+ macrophages', 'CD68+GzmB+ macrophages', 'CD38+CD68+ macrophages',
       'plasma cells', 'B cells', 'NK cells', 'immune cells',
       'smooth muscles', 'stroma', 'lymphatic', 'immune/vasculature',
       'tumor', 'tumor/immune', 'tumor/immune/vasculature',
       'epithelial cells', 'epithelial/immune',
       'nerves',
        'dirt', 'undefined']


# this plot shows the types of cells (ClusterIDs) in the different niches (0-7)
k_to_plot = 10
niche_clusters = (k_centroids[k_to_plot])
tissue_avgs = values.mean(axis = 0)
fc = np.log2(((niche_clusters+tissue_avgs)/(niche_clusters+tissue_avgs).sum(axis = 1, keepdims = True))/tissue_avgs)
fc = pd.DataFrame(fc,columns = sum_cols)
s=sns.clustermap(fc.loc[[0,1,2,3,4,5,6,7,8,9],cell_order], vmin =-3,vmax = 3,cmap = 'bwr',row_cluster = False)
s.savefig("figures/celltypes_perniche_10.pdf")

######
## groups here (1,2) refer to CLR and DII
cells['neighborhood10'] = cells['neighborhood10'].astype('category')
sns.lmplot(data = cells[cells['groups']=='CLR'],x = 'x',y='y',hue = 'neighborhood10',palette = 'bright',height = 8,col = reg,col_wrap = 10,fit_reg = False)
plt.savefig('figures/lmplot_CLR.png')

cells['neighborhood10'] = cells['neighborhood10'].astype('category')
sns.lmplot(data = cells[cells['groups']=='DII'],x = 'x',y='y',hue = 'neighborhood10',palette = 'bright',height = 8,col = reg,col_wrap = 10,fit_reg = False)
plt.savefig('figures/lmplot_DII.png')


###############
## Write the file with neighborhood info to a csv file
cells.to_csv('./data/CRC_pathml_Neighbors.csv')


#####################################
#plot for each group and each patient the percent of total cells allocated to each neighborhood
fc = cells.groupby(['patients','groups']).apply(lambda x: x['neighborhood10'].value_counts(sort = False,normalize = True))

fc.columns = range(10)
melt = pd.melt(fc.reset_index(),id_vars = ['patients','groups'])
melt = melt.rename(columns = {'variable':'neighborhood','value':'frequency of neighborhood'})
f,ax = plt.subplots(figsize = (10,5))
sns.stripplot(data = melt, hue = 'groups',dodge = True,alpha = .2,x ='neighborhood', y ='frequency of neighborhood')
sns.pointplot(data = melt, scatter_kws  = {'marker': 'd'},hue = 'groups',dodge = .5,join = False,x ='neighborhood', y ='frequency of neighborhood')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Groups",
          handletextpad=0, columnspacing=1,
          loc="upper left", ncol=3, frameon=True)

plt.savefig('figures/Neighborhood_Frequency.png')

#t-test to evaluate if any neighborhood is enriched in one group
from scipy.stats import ttest_ind
for i in range(10):
    n2 = melt[melt['neighborhood']==i]
    print (i,'    ',ttest_ind(n2[n2['groups']=='CLR']['frequency of neighborhood'],n2[n2['groups']=='DII']['frequency of neighborhood']))

####
#same as above except neighborhood 5 (8 in ours!) is removed from analysis.
fc = cells[cells['neighborhood10']!=8].groupby(['patients','groups']).apply(lambda x: x['neighborhood10'].value_counts(sort = False,normalize = True))

fc.columns = range(10)
melt = pd.melt(fc.reset_index(),id_vars = ['patients','groups'])
melt = melt.rename(columns = {'variable':'neighborhood','value':'frequency of neighborhood'})

f,ax = plt.subplots(figsize = (10,5))
sns.stripplot(data = melt, hue = 'groups',dodge = True,alpha = .2,x ='neighborhood', y ='frequency of neighborhood')
sns.pointplot(data = melt, scatter_kws  = {'marker': 'd'},hue = 'groups',dodge = .5,join = False,x ='neighborhood', y ='frequency of neighborhood')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Groups",
          handletextpad=0, columnspacing=1,
          loc="upper left", ncol=3, frameon=True)

plt.savefig('figures/Neighborhood_Frequency_without8.png')


for i in range(10):
    n2 = melt[melt['neighborhood']==i]
    print (i,'    ',ttest_ind(n2[n2['groups']=='CLR']['frequency of neighborhood'],n2[n2['groups']=='DII']['frequency of neighborhood']))