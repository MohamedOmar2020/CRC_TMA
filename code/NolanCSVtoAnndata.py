
# import dependencies
import os
import numpy as np
import pandas as pd
import scanpy as sc
import loompy as lp
from cytoolz import compose
import loompy
import anndata
import sys

## Load data from Noolan
NoolanData = pd.read_csv('./data/CRC_clusters_neighborhoods_markers.csv')
NoolanData.shape
NoolanData['ClusterName'].value_counts()
NoolanData['groups'].value_counts()
NoolanData['TMA_AB'].value_counts()

NoolanData_groups = NoolanData[['File Name', 'groups', 'patients']]
NoolanData_groups.rename(columns = {"File Name": "Region"}, inplace=True)





## Assemble loom file row and column attributes

col_attrs = {
    "CellID": np.array(NoolanData.index),
    "clusters": np.array(NoolanData['ClusterName'].values),
    "Neighborhoods": np.array(NoolanData['neighborhood name'].values),
    'Region': np.array(NoolanData['File Name'].values),
    'TMA': np.array(NoolanData['TMA_AB'].values),
    'patients': np.array(NoolanData['patients'].values),
    'x': np.array(NoolanData['X:X'].values),
    'y': np.array(NoolanData['Y:Y'].values),
    'size': np.array(NoolanData['size:size'].values),
    'groups': np.array(NoolanData['groups'].values)
}

MarkerNames = [
    'CD44-stroma',
    'FOXP3-Treg',
    'CD8',
    'p53',
    'GATA3-Th2',
    'CD45-hematopoietic',
    'T-bet-Th1',
    'beta-cat',
    'HLA-DR',
    'PD-L1',
    'Ki67',
    'CD45RA-naiveT',
    'CD4',
    'CD21-DCs',
    'MUC-1-epithelia',
    'CD30-costimulator',
    'CD2-Tcells',
    'Vimentin',
    'CD20',
    'LAG-3-checkpoint',
    'Na-K-ATPase',
    'CD5-Tcells',
    'IDO-1-metabolism',
    'Cytokeratin',
    'CD11b-macrophages',
    'CD56-NK cells',
    'aSMA-smooth muscle',
    'BCL-2-apoptosis',
    'CD25-IL-2Ra',
    'CD11c-DCs',
    'PD-1-checkpoint',
    'Granzyme B-cytotoxicity',
    'EGFR',
    'VISTA-costimulator',
    'CD15-granulocytes',
    'ICOS-costimulator',
    'Synaptophysin-neuroendocrine',
    'GFAP-nerves',
    'CD7-Tcells',
    'CD3-Tcells',
    'ChromograninA-neuroendocrine',
    'CD163-macrophages',
    'CD45RO-memory cells',
    'CD68-macrophages',
    'CD31-vasculature',
    'Podoplanin-lymphatics',
    'CD34-vasculature',
    'CD38-multifunctional',
    'CD138-plasma cells'
]


row_attrs = {
    "markers": np.array(MarkerNames),
}

Matrix =  np.array(NoolanData.iloc[:, 12:61], dtype='float', order='K').transpose()

lp.create('data/Nolan.loom', layers= Matrix ,row_attrs=row_attrs, col_attrs=col_attrs)

#############################
# Load the newly formed adata from Nolan's matrix
Nolan_adata = sc.read_loom('data/Nolan.loom', obs_names='CellID', var_names='markers')
Nolan_adata
Nolan_adata.obs_names
Nolan_adata.var_names

# Store the raw data
Nolan_adata.raw = Nolan_adata

# Scaling
sc.pp.log1p(Nolan_adata)
sc.pp.scale(Nolan_adata, max_value=10)
#sc.pl.violin(adata_combined_All, keys=['HOECHST1'], save='DAPI_scaled.png')

# PCA and batch correction
#bbknn.ridge_regression(adata_combined_All, batch_key=['Region'], confounder_key=['patients'])
sc.tl.pca(Nolan_adata)
sc.pl.pca_variance_ratio(Nolan_adata, n_pcs=40, log=True)
#bbknn.bbknn(adata_combined_All, batch_key='Region', neighbors_within_batch = 15, n_pcs = 30, trim = 40)
sc.external.pp.harmony_integrate(Nolan_adata, key='Region')

Nolan_adata.write(filename='./data/Nolan_adata_harmony.h5ad')

adata_combined_All = ad.read_h5ad('./data/Nolan_adata_harmony.h5ad', chunk_size=100000)

sc.pp.neighbors(Nolan_adata, n_neighbors=15, n_pcs=40, use_rep='X_pca_harmony')

# compute Umap embedding
sc.tl.umap(Nolan_adata)

# louvain clustering
with parallel_backend('threading', n_jobs=15):
    sc.tl.louvain(Nolan_adata, resolution = 2)

Nolan_adata.obs['louvain'].value_counts()


#Nolan_adata.obs['groups'] = Nolan_adata.obs['groups'].astype('category')
#Nolan_adata.obs['patients'] = Nolan_adata.obs['patients'].astype('category')

# Save results for DE with MAST
Nolan_adata.write(filename='./data/Nolan_adata_processed.h5ad')

Nolan_adata = ad.read_h5ad('./data/Nolan_adata_processed.h5ad', chunk_size=100000)

# Plot PCA embedding
#sc.pl.embedding(adata_combined_All, basis='X_pca_harmony', color=['TMA'])
#sc.pl.embedding(adata_combined_All, basis='X_pca', color=['TMA'])

# Plot UMAP embedding
sc.pl.umap(Nolan_adata, color=['Region'], save= 'umapRegions_Nolan.png')
sc.pl.umap(Nolan_adata, color=['TMA'], save= 'TMA_Nolan.png')
sc.pl.umap(Nolan_adata, color=['patients'], save= 'patients_Nolan.png')
sc.pl.umap(Nolan_adata, color=['groups'], save='groups_Nolan.png')
sc.pl.umap(Nolan_adata, color=['louvain'], save= 'Louvain_Nolan.png')
sc.pl.umap(Nolan_adata, color=['clusters'], save= 'clusters_Nolan.png')


# Cluster markers
sc.tl.rank_genes_groups(Nolan_adata, groupby = 'clusters', method='wilcoxon', pts=True)

sc.pl.rank_genes_groups_dotplot(Nolan_adata, groupby='clusters', n_genes=5, dendrogram = False, cmap='bwr', values_to_plot = 'logfoldchanges', vmin=-4, vmax=4, save='Nolan_MarkersDotplot_wilcoxon.png')

#sc.pl.rank_genes_groups(adata_combined_All, n_genes=10, sharey=False, save='LeidenMarkers0.3_ttest_scaled.png')
sc.pl.rank_genes_groups(Nolan_adata, n_genes=10, sharey=False, save='NolanMarkers_wilcoxon.png')


RankMatrix_Nolan = sc.get.rank_genes_groups_df(Nolan_adata, group=None)

sc.tl.rank_genes_groups(adata_combined_All, groupby = 'cell_types', method='wilcoxon', pts=True)

RankMatrix_Pathml = sc.get.rank_genes_groups_df(adata_combined_All, group=None)








