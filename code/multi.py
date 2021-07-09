


from os import listdir,path,getcwd
import glob
import pandas as pd
from pathml.core import SlideDataset
from pathml.core.slide_data import VectraSlide
from pathml.core.slide_data import CODEXSlide
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import SegmentMIF, QuantifyMIF, CollapseRunsCODEX

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
from dask.distributed import Client
from deepcell.utils.plot_utils import make_outline_overlay
from deepcell.utils.plot_utils import create_rgb_image

import scanpy as sc
import squidpy as sq
import anndata as ad
import re
import scipy.stats
import batchglm.api as glm
import diffxpy.api as de
import bbknn
from joblib import parallel_backend

import scvi
import scanorama
import seaborn as sb

#sc.set_figure_params(dpi=200, frameon=True, figsize=(20,15), format='png')

sc._settings.ScanpyConfig(max_memory=60, n_jobs=14)
###################################################
## Load data from Noolan
NoolanData = pd.read_csv('./data/CRC_clusters_neighborhoods_markers.csv')
NoolanData.shape
NoolanData['ClusterName'].value_counts()
NoolanData['groups'].value_counts()
NoolanData['TMA_AB'].value_counts()
NoolanData['File Name'].value_counts()

NoolanData_groups = NoolanData[['File Name', 'groups', 'patients']]
NoolanData_groups.rename(columns = {"File Name": "Region"}, inplace=True)

#######################################################
channelnames = pd.read_csv("data/channelNames.txt", header = None, dtype = str, low_memory=False)

## Load all images
dirpath_A = r"/Volumes/Mohamed/TMA_A/bestFocus"
dirpath_B = r"/Volumes/Mohamed/TMA_B/bestFocus"

# assuming that all WSIs are in a single directory, all with .tif file extension
vectra_paths_A = glob.glob(path.join(dirpath_A, "*.tif"))
vectra_paths_B = glob.glob(path.join(dirpath_B, "*.tif"))


# create a list of SlideData objects by loading each path
vectra_list_A = [CODEXSlide(p, stain='IF') for p in vectra_paths_A]
vectra_list_B = [CODEXSlide(p, stain='IF') for p in vectra_paths_B]

# initialize a SlideDataset
dataset_A = SlideDataset(vectra_list_A)
dataset_B = SlideDataset(vectra_list_B)

pipe = Pipeline([
    CollapseRunsCODEX(z=0),
    SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=29, image_resolution=0.377442),
    QuantifyMIF(segmentation_mask='cell_segmentation')
])

client = Client(n_workers = 14)
dataset_A.run(pipe, distributed = False, client = client, tile_size=(1920,1440), tile_pad=False)
dataset_B.run(pipe, distributed = False, client = client, tile_size=(1920,1440), tile_pad=False)

################
for i in dataset_A.slides:
    i.name = i.name.replace("_", "")
    i.name = i.name.replace("X01Y01", "")

#dataset_A.slides[0].name = dataset_A.slides[0].name.replace('Y01', '')


for i in dataset_B.slides:
    i.name = i.name.replace("_", "")
    i.name = i.name.replace("X01Y01", "")

#img = slidedata.tiles[0].image

def plot(slidedata, tile, channel1, channel2):
    image = np.expand_dims(slidedata.tiles[tile].image, axis=0)
    nuc_segmentation_predictions = np.expand_dims(slidedata.tiles[tile].masks['nuclear_segmentation'], axis=0)
    cell_segmentation_predictions = np.expand_dims(slidedata.tiles[tile].masks['cell_segmentation'], axis=0)
    #nuc_cytoplasm = np.expand_dims(np.concatenate((image[:,:,:,channel1,0], image[:,:,:,channel2,0]), axis=2), axis=0)
    nuc_cytoplasm = np.stack((image[:,:,:,channel1], image[:,:,:,channel2]), axis=-1)
    rgb_images = create_rgb_image(nuc_cytoplasm, channel_colors=['blue', 'green'])
    overlay_nuc = make_outline_overlay(rgb_data=rgb_images, predictions=nuc_segmentation_predictions)
    overlay_cell = make_outline_overlay(rgb_data=rgb_images, predictions=cell_segmentation_predictions)
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(rgb_images[0, ...])
    ax[1].imshow(overlay_cell[0, ...])
    ax[0].set_title('Raw data')
    ax[1].set_title('Cell Predictions')
    plt.savefig('figures/' + str(slidedata.name) + '_' + 'Ch' + str(channel1) + 'Ch' + str(channel2) + ".png", format="PNG")
    plt.show()


# DAPI + Syp
plot(dataset_A.slides[1], tile=0, channel1=12, channel2=60)
plot(dataset_A.slides[1], tile=0, channel1=29, channel2=60)
plot(dataset_A.slides[0], tile=0, channel1=29, channel2=0)
plot(dataset_A.slides[0], tile=0, channel1=33, channel2=0)
plot(dataset_A.slides[0], tile=0, channel1=33, channel2=33)
plot(dataset_A.slides[0], tile=0, channel1=29, channel2=29)
plot(dataset_A.slides[0], tile=0, channel1=23, channel2=46)

plot(dataset_B.slides[1], tile=0, channel1=12, channel2=60)
plot(dataset_B.slides[1], tile=0, channel1=29, channel2=60)
plot(dataset_B.slides[0], tile=0, channel1=29, channel2=0)
plot(dataset_B.slides[0], tile=0, channel1=33, channel2=0)
plot(dataset_B.slides[0], tile=0, channel1=33, channel2=33)
plot(dataset_B.slides[0], tile=0, channel1=29, channel2=29)


# Plot all images with DCs markers
for i in dataset_A.slides:
    plot(i, tile=0, channel1=23, channel2=46)


##############################################################
## concatenate the results in 1 file

# Initialize empty dicts
all_results_A = {}
all_results_B = {}

# extrac the count matrices from SlideDataset and save them in the corresponding dictionary
for i in dataset_A.slides:
    results = i.counts
    all_results_A[i] = results

for i in dataset_B.slides:
    results = i.counts
    all_results_B[i] = results

## Combine the count matrices into a single adata object:

# For dataset A
adata_combined_A = ad.concat(all_results_A, join="outer", label="Region", index_unique='_')
# Fix and replace the regions names
origin_A = adata_combined_A.obs['Region']
origin_A = origin_A.astype(str).str.replace("[^a-zA-Z0-9 \n\.]", "")
origin_A = origin_A.astype(str).str.replace("[\n]", "")
origin_A = origin_A.str.replace("SlideDataname", "")
origin_A = origin_A.str.replace("X.*", "")
origin_A = origin_A +'_A'
adata_combined_A.obs['Region'] = origin_A
#adata_combined_A.write_loom(filename='./data/loomCombined_A.loom', write_obsm_varm=True)
#adata_combined_A.write(filename='./data/loomCombined_A.h5ad')
adata_combined_A = ad.read_h5ad(filename='./data/loomCombined_A.h5ad', chunk_size=100000)
# Batch correction
#sc.pp.combat(adata_combined_A, key='Region')
#adata_combined_A.write(filename='./data/loomCombined_A_combat.h5ad')


# For dataset B
adata_combined_B = ad.concat(all_results_B, join="outer", label="Region", index_unique='_')
origin_B = adata_combined_B.obs['Region']
origin_B = origin_B.astype(str).str.replace("[^a-zA-Z0-9 \n\.]", "")
origin_B = origin_B.astype(str).str.replace("[\n]", "")
origin_B = origin_B.str.replace("SlideDataname", "")
origin_B = origin_B.str.replace("Z.*", "")
origin_B = origin_B+'_B'
adata_combined_B.obs['Region'] = origin_B
#adata_combined_B.write_loom(filename='./data/loomCombined_B.loom', write_obsm_varm=True)
#adata_combined_B.write(filename='./data/loomCombined_B.h5ad')
adata_combined_B = ad.read_h5ad(filename='./data/loomCombined_B.h5ad', chunk_size=100000)
# Batch correction
#sc.pp.combat(adata_combined_B, key='Region')
#adata_combined_B.write(filename='./data/loomCombined_B_combat.h5ad')

# Combine adata A and B
all_looms = [adata_combined_A, adata_combined_B]
adata_combined_All = ad.concat(all_looms, join = 'outer', label = 'TMA', index_unique='_')

# Batch correction
#sc.pp.combat(adata_combined_All, key='TMA')
#adata_combined_B.write(filename='./data/loomCombined_B_combat.h5ad')

# save the integrated loom file
adata_combined_All.write(filename='./data/loomCombined_All.h5ad')

#adata_combined_A = ad.read_loom('./data/loomCombined_A.loom', obs_names= 'obs_names')
#adata_combined_B = ad.read_loom('./data/loomCombined_B.loom', obs_names= 'obs_names')
#adata_combined_All = ad.read_loom('./data/loomCombined_All.loom', obs_names='obs_names')
adata_combined_All = ad.read_h5ad('./data/loomCombined_All.h5ad', chunk_size=100000)

# Rename the variable names (channels) in the adata object
adata_combined_All.var_names = channelnames[0]

#######################################
## Remove the empty and nuclear channels
keep = ['HOECHST1', 'CD44 - stroma', 'FOXP3 - regulatory T cells', 'CD8 - cytotoxic T cells', 'p53 - tumor suppressor', 'GATA3 - Th2 helper T cells', 'CD45 - hematopoietic cells', 'T-bet - Th1 cells', 'beta-catenin - Wnt signaling', 'HLA-DR - MHC-II', 'PD-L1 - checkpoint', 'Ki67 - proliferation', 'CD45RA - naive T cells', 'CD4 - T helper cells', 'MUC-1 - epithelia', 'CD30 - costimulator', 'CD2 - T cells', 'Vimentin - cytoplasm', 'CD20 - B cells', 'LAG-3 - checkpoint', 'Na-K-ATPase - membranes', 'CD5 - T cells', 'IDO-1 - metabolism', 'Cytokeratin - epithelia', 'CD11b - macrophages', 'CD56 - NK cells', 'aSMA - smooth muscle', 'BCL-2 - apoptosis', 'CD25 - IL-2 Ra', 'PD-1 - checkpoint', 'Granzyme B - cytotoxicity', 'EGFR - singling', 'VISTA - costimulator', 'CD15 - granulocytes', 'ICOS - costimulator', 'Synaptophysin - neuroendocrine', 'GFAP - nerves', 'CD7 - T cells', 'CD3 - T cells', 'Chromogranin A - neuroendocrine', 'CD163 - macrophages', 'CD45RO - memory cells', 'CD68 - macrophages', 'CD31 - vasculature', 'Podoplanin - lymphatics', 'CD34 - vasculature', 'CD38 - multifunctional', 'CD138 - plasma cells', 'DRAQ5']

adata_combined_All_t = adata_combined_All.transpose()
adata_combined_All = adata_combined_All_t[np.isin(adata_combined_All.var.index, keep)].copy().transpose()
adata_combined_All.obs_names_make_unique()

# Rename the markers
#adata_combined_All.var_names = ['CD44_stroma', 'FOXP3_regTcells', 'CDX2_intestinalEpith', 'CD8', 'P53', 'GATA3_TH2', 'CD45_Hemato', 'Tbet_TH1', 'beta_Cat', 'HLA_DR', 'PD_L1', 'Ki67', 'CD45RA_naiveT', 'CD4', 'CD21_DCs', 'MUC1_epith', 'CD30_costimulator', 'CD2_Tcells', 'Vimentin', 'CD20_Bcells', 'LAG3_Checkpoint', 'NaKatpase_memb', 'CD5_Tcells', 'IDO1_metabolism', 'cytokeratin_epith', 'CD11b_macrophages', 'CD56_NK', 'aSMA_smoothMuscle', 'BCL2_apoptosis', 'CD25_IL2Ra', 'Collagen4_BasMemb', 'CD11c_DCs', 'PD1_checkpoint', 'GranzymeB_Cytotoxicity', 'EGFR', 'VISTA_Costimulator', 'CD15_granulocytes', 'CD194_CCR4ChemokineR', 'ICOS_Costimulator', 'MMP9_MatrixMetalloProt', 'Synapto_NE', 'CD71_TransferrinR', 'GFAP_nerves', 'CD7_Tcells', 'CD3_Tcells', 'ChromograninA_NE', 'CD163_macrophages', 'CD57_NK', 'CD45RO_memoryCells', 'CD68_macrophages', 'CD31_vasculature', 'Podoplanin_lymphatics', 'CD34_vasculature', 'CD38_multi', 'CD138_plasmaCells', 'MMP12_MatrixMetellaProt']
adata_combined_All.var_names = ['HOECHST1', 'CD44-stroma', 'FOXP3-Treg', 'CD8', 'p53', 'GATA3-Th2', 'CD45-hematopoietic', 'T-bet-Th1', 'beta-cat', 'HLA-DR', 'PD-L1', 'Ki67', 'CD45RA-naiveT', 'CD4', 'MUC-1-epithelia', 'CD30-costimulator', 'CD2-Tcells', 'Vimentin', 'CD20', 'LAG-3-checkpoint', 'Na-K-ATPase', 'CD5-Tcells', 'IDO-1-metabolism', 'Cytokeratin', 'CD11b-macrophages', 'CD56-NK cells', 'aSMA-smooth muscle', 'BCL-2-apoptosis', 'CD25-IL-2Ra', 'PD-1-checkpoint', 'Granzyme B-cytotoxicity', 'EGFR', 'VISTA-costimulator', 'CD15-granulocytes', 'ICOS-costimulator', 'Synaptophysin-neuroendocrine', 'GFAP-nerves', 'CD7-Tcells', 'CD3-Tcells', 'ChromograninA-neuroendocrine', 'CD163-macrophages', 'CD45RO-memory cells', 'CD68-macrophages', 'CD31-vasculature', 'Podoplanin-lymphatics', 'CD34-vasculature', 'CD38-multifunctional', 'CD138-plasma cells', 'DRAQ5']

#sc.pl.violin(adata_combined_All, keys=['HOECHST1'], save='DAPI.png')
#sc.pl.violin(adata_combined_All, keys=['DRAQ5'], save='DRAQ5.png')

# Filter cells with DAPI expression < 0
adata_combined_All = adata_combined_All[adata_combined_All[: , 'HOECHST1'].X > 60, :]
adata_combined_All = adata_combined_All[adata_combined_All[: , 'DRAQ5'].X > 100, :]

# Remove DAPI
keep = ['CD44-stroma', 'FOXP3-Treg', 'CD8', 'p53', 'GATA3-Th2', 'CD45-hematopoietic', 'T-bet-Th1', 'beta-cat', 'HLA-DR', 'PD-L1', 'Ki67', 'CD45RA-naiveT', 'CD4', 'MUC-1-epithelia', 'CD30-costimulator', 'CD2-Tcells', 'Vimentin', 'CD20', 'LAG-3-checkpoint', 'Na-K-ATPase', 'CD5-Tcells', 'IDO-1-metabolism', 'Cytokeratin', 'CD11b-macrophages', 'CD56-NK cells', 'aSMA-smooth muscle', 'BCL-2-apoptosis', 'CD25-IL-2Ra', 'PD-1-checkpoint', 'Granzyme B-cytotoxicity', 'EGFR', 'VISTA-costimulator', 'CD15-granulocytes', 'ICOS-costimulator', 'Synaptophysin-neuroendocrine', 'GFAP-nerves', 'CD7-Tcells', 'CD3-Tcells', 'ChromograninA-neuroendocrine', 'CD163-macrophages', 'CD45RO-memory cells', 'CD68-macrophages', 'CD31-vasculature', 'Podoplanin-lymphatics', 'CD34-vasculature', 'CD38-multifunctional', 'CD138-plasma cells']
adata_combined_All = adata_combined_All[: , keep]

# store the raw data for further use
adata_combined_All.raw = adata_combined_All

# Save the filtered anndata
adata_combined_All.write(filename='./data/loomCombined_filtered.h5ad')

adata_combined_All = ad.read_h5ad('./data/loomCombined_filtered.h5ad', chunk_size=100000)

############################
## Add patients and groups info
NoolanData_patients = NoolanData_groups.copy()
NoolanData_groups = NoolanData_groups.groupby(['Region', 'groups']).size().unstack(fill_value=0)

## Annotation dict for patients
regions_to_patients = dict(
reg001_A = '1',
reg001_B = '1',
reg002_A = '1',
reg002_B = '1',
reg003_A = '2',
reg003_B = '2',
reg004_A = '2',
reg004_B = '2',
reg005_A = '3',
reg005_B = '3',
reg006_A = '3',
reg006_B = '3',
reg007_A = '4',
reg007_B = '4',
reg008_A = '4',
reg008_B = '4',
reg009_A = '5',
reg009_B = '5',
reg010_A = '5',
reg010_B = '5',
reg011_A = '6',
reg011_B = '6',
reg012_A = '6',
reg012_B = '6',
reg013_A = '7',
reg013_B = '7',
reg014_A = '7',
reg014_B = '7',
reg015_A = '8',
reg015_B = '8',
reg016_A = '8',
reg016_B = '8',
reg017_A = '9',
reg017_B = '9',
reg018_A = '9',
reg018_B = '9',
reg019_A = '10',
reg019_B = '10',
reg020_A = '10',
reg020_B = '10',
reg021_A = '11',
reg021_B = '11',
reg022_A = '11',
reg022_B = '11',
reg023_A = '12',
reg023_B = '12',
reg024_A = '12',
reg024_B = '12',
reg025_A = '13',
reg025_B = '13',
reg026_A = '13',
reg026_B = '13',
reg027_A = '14',
reg027_B = '14',
reg028_A = '14',
reg028_B = '14',
reg029_A = '15',
reg029_B = '15',
reg030_A = '15',
reg030_B = '15',
reg031_A = '16',
reg031_B = '16',
reg032_A = '16',
reg032_B = '16',
reg033_A = '17',
reg033_B = '17',
reg034_A = '17',
reg034_B = '17',
reg035_A = '18',
reg035_B = '18',
reg036_A = '18',
reg036_B = '18',
reg037_A = '19',
reg037_B = '19',
reg038_A = '19',
reg038_B = '19',
reg039_A = '20',
reg039_B = '20',
reg040_A = '20',
reg040_B = '20',
reg041_A = '21',
reg041_B = '21',
reg042_A = '21',
reg042_B = '21',
reg043_A = '22',
reg043_B = '22',
reg044_A = '22',
reg044_B = '22',
reg045_A = '23',
reg045_B = '23',
reg046_A = '23',
reg046_B = '23',
reg047_A = '24',
reg047_B = '24',
reg048_A = '24',
reg048_B = '24',
reg049_A = '25',
reg049_B = '25',
reg050_A = '25',
reg050_B = '25',
reg051_A = '26',
reg051_B = '26',
reg052_A = '26',
reg052_B = '26',
reg053_A = '27',
reg053_B = '27',
reg054_A = '27',
reg054_B = '27',
reg055_A = '28',
reg055_B = '28',
reg056_A = '28',
reg056_B = '28',
reg057_A = '29',
reg057_B = '29',
reg058_A = '29',
reg058_B = '29',
reg059_A = '30',
reg059_B = '30',
reg060_A = '30',
reg060_B = '30',
reg061_A = '31',
reg061_B = '31',
reg062_A = '31',
reg062_B = '31',
reg063_A = '32',
reg063_B = '32',
reg064_A = '32',
reg064_B = '32',
reg065_A = '33',
reg065_B = '33',
reg066_A = '33',
reg066_B = '33',
reg067_A = '34',
reg067_B = '34',
reg068_A = '34',
reg068_B = '34',
reg069_A = '35',
reg069_B = '35',
reg070_A = '35',
reg070_B = '35'
)


## Annotation dict for groups
regions_to_groups = dict(
reg001_A = 'CLR',
reg001_B = 'CLR',
reg002_A = 'CLR',
reg002_B = 'CLR',
reg003_A = 'DII',
reg003_B = 'DII',
reg004_A = 'DII',
reg004_B = 'DII',
reg005_A = 'DII',
reg005_B = 'DII',
reg006_A = 'DII',
reg006_B = 'DII',
reg007_A = 'DII',
reg007_B = 'DII',
reg008_A = 'DII',
reg008_B = 'DII',
reg009_A = 'DII',
reg009_B = 'DII',
reg010_A = 'DII',
reg010_B = 'DII',
reg011_A = 'CLR',
reg011_B = 'CLR',
reg012_A = 'CLR',
reg012_B = 'CLR',
reg013_A = 'DII',
reg013_B = 'DII',
reg014_A = 'DII',
reg014_B = 'DII',
reg015_A = 'DII',
reg015_B = 'DII',
reg016_A = 'DII',
reg016_B = 'DII',
reg017_A = 'DII',
reg017_B = 'DII',
reg018_A = 'DII',
reg018_B = 'DII',
reg019_A = 'CLR',
reg019_B = 'CLR',
reg020_A = 'CLR',
reg020_B = 'CLR',
reg021_A = 'CLR',
reg021_B = 'CLR',
reg022_A = 'CLR',
reg022_B = 'CLR',
reg023_A = 'CLR',
reg023_B = 'CLR',
reg024_A = 'CLR',
reg024_B = 'CLR',
reg025_A = 'CLR',
reg025_B = 'CLR',
reg026_A = 'CLR',
reg026_B = 'CLR',
reg027_A = 'DII',
reg027_B = 'DII',
reg028_A = 'DII',
reg028_B = 'DII',
reg029_A = 'DII',
reg029_B = 'DII',
reg030_A = 'DII',
reg030_B = 'DII',
reg031_A = 'DII',
reg031_B = 'DII',
reg032_A = 'DII',
reg032_B = 'DII',
reg033_A = 'CLR',
reg033_B = 'CLR',
reg034_A = 'CLR',
reg034_B = 'CLR',
reg035_A = 'DII',
reg035_B = 'DII',
reg036_A = 'DII',
reg036_B = 'DII',
reg037_A = 'CLR',
reg037_B = 'CLR',
reg038_A = 'CLR',
reg038_B = 'CLR',
reg039_A = 'CLR',
reg039_B = 'CLR',
reg040_A = 'CLR',
reg040_B = 'CLR',
reg041_A = 'CLR',
reg041_B = 'CLR',
reg042_A = 'CLR',
reg042_B = 'CLR',
reg043_A = 'DII',
reg043_B = 'DII',
reg044_A = 'DII',
reg044_B = 'DII',
reg045_A = 'DII',
reg045_B = 'DII',
reg046_A = 'DII',
reg046_B = 'DII',
reg047_A = 'CLR',
reg047_B = 'CLR',
reg048_A = 'CLR',
reg048_B = 'CLR',
reg049_A = 'DII',
reg049_B = 'DII',
reg050_A = 'DII',
reg050_B = 'DII',
reg051_A = 'DII',
reg051_B = 'DII',
reg052_A = 'DII',
reg052_B = 'DII',
reg053_A = 'DII',
reg053_B = 'DII',
reg054_A = 'DII',
reg054_B = 'DII',
reg055_A = 'CLR',
reg055_B = 'CLR',
reg056_A = 'CLR',
reg056_B = 'CLR',
reg057_A = 'CLR',
reg057_B = 'CLR',
reg058_A = 'CLR',
reg058_B = 'CLR',
reg059_A = 'DII',
reg059_B = 'DII',
reg060_A = 'DII',
reg060_B = 'DII',
reg061_A = 'DII',
reg061_B = 'DII',
reg062_A = 'DII',
reg062_B = 'DII',
reg063_A = 'CLR',
reg063_B = 'CLR',
reg064_A = 'CLR',
reg064_B = 'CLR',
reg065_A = 'CLR',
reg065_B = 'CLR',
reg066_A = 'CLR',
reg066_B = 'CLR',
reg067_A = 'CLR',
reg067_B = 'CLR',
reg068_A = 'CLR',
reg068_B = 'CLR',
reg069_A = 'CLR',
reg069_B = 'CLR',
reg070_A = 'CLR',
reg070_B = 'CLR'
)


# Map the values
adata_combined_All.obs['patients'] = (
    adata_combined_All.obs['Region']
    .map(regions_to_patients)
    .astype('category')
)

adata_combined_All.obs['groups'] = (
    adata_combined_All.obs['Region']
    .map(regions_to_groups)
    .astype('category')
)

# Save the filtered anndata
adata_combined_All.write(filename='./data/loomCombined_filtered_Annotated.h5ad')

adata_combined_All = ad.read_h5ad('./data/loomCombined_filtered_Annotated.h5ad', chunk_size=100000)

## Sanity check
adata_combined_All.obs['Region'].value_counts()
adata_combined_All.obs['patients'].value_counts()
adata_combined_All.obs['groups'].value_counts()

#######
##############
# Check the count matrix
CountMatrix = adata_combined_All.to_df()
np.isnan(CountMatrix).sum()
CountMatrix.isnull().sum()
CountMatrix.isin([0]).sum()
CountMatrix.max().max()
CountMatrix.min()
CountMatrix

################################################
## Scanpy workflow

## SCVI workflow
#sc.pp.filter_cells(adata_combined_All, min_counts=40)
#adata_combined_All.layers["counts"] = adata_combined_All.X.copy()
#sc.pp.log1p(adata_combined_All)
#sc.pp.scale(adata_combined_All, max_value=10)
#adata_combined_All.raw = adata_combined_All # freeze the state in `.raw`
# Prepare the data
#scvi.data.setup_anndata(
#    adata_combined_All,
#    layer="counts",
#    #categorical_covariate_keys=["patients", "groups"],
#    #continuous_covariate_keys=["filled_area", "x", "y"],
#    batch_key='Region'
#)
# Configure and train the model
#model = scvi.model.SCVI(adata_combined_All, dispersion='gene-batch', n_latent=10, n_layers = 2, n_hidden = 256)
#model
#model.train(train_size = 0.8, max_epochs = 50, batch_size = 256, plan_kwargs = {"n_epochs_kl_warmup":None})
#model.history["elbo_train"].plot()
#plt.show()
#model.save("data/scvi_Model/")
#model = scvi.model.SCVI.load("data/scvi_Model/", adata_combined_All)
## Obtaining model outputs
#latent = model.get_latent_representation()
#adata_combined_All.obsm["X_scVI"] = latent
# store the normalized values back in the anndata.
#adata_combined_All.layers["scvi_normalized"] = model.get_normalized_expression()
# use scVI latent space for UMAP generation
#sc.pp.neighbors(adata_combined_All, n_neighbors=30, use_rep='X_scVI')
# compute Umap embedding
#sc.tl.umap(adata_combined_All)
# louvain clustering
#with parallel_backend('threading', n_jobs=15):
#    sc.tl.louvain(adata_combined_All, resolution = 3)
#adata_combined_All.obs['louvain'].value_counts()
## DE
#de_df = model.differential_expression(adata=adata_combined_All, groupby="louvain")
#markers = {}
#cats = adata_combined_All.obs.louvain.cat.categories
#for i, c in enumerate(cats):
#    cid = "{} vs Rest".format(c)
#    cell_type_df = de_df.loc[de_df.comparison == cid]
#   cell_type_df = cell_type_df[cell_type_df.lfc_mean > 0]
#    cell_type_df = cell_type_df[cell_type_df["bayes_factor"] > 3]
#    cell_type_df = cell_type_df[cell_type_df["non_zeros_proportion1"] > 0.1]
#    markers[c] = cell_type_df.index.tolist()[:4]

############################################
# Scanorama
#batches = adata_combined_All.obs['Region'].cat.categories.tolist()
#alldata = {}
#for batch in batches:
#    alldata[batch] = adata_combined_All[adata_combined_All.obs['Region'] == batch,]

#alldata = list(alldata.values())
#alldata

#for adata in alldata:
#    sc.pp.log1p(adata)
#    sc.pp.scale(adata, max_value=10)


#alldata_cor = scanorama.correct_scanpy(alldata, return_dimred=True)
#alldata_cor

## Add the raw attributes
#alldata = {}
#for batch in batches:
#    alldata[batch] = adata_combined_All[adata_combined_All.obs['Region'] == batch,]

#for i, j in zip(alldata.keys(), range(len(alldata_cor))):
#    alldata_cor[j].raw = alldata[i]


#adata_all = ad.concat(alldata_cor, label="Region", uns_merge="unique", join = 'outer',index_unique="_")
#adata_all

# umap
#sc.pp.neighbors(adata_all, n_neighbors=10, n_pcs =30, use_rep = "X_scanorama")
#sc.tl.umap(adata_all)

# louvain clustering
#with parallel_backend('threading', n_jobs=15):
#    sc.tl.louvain(adata_all, resolution = 1)

#adata_all.obs['louvain'].value_counts()

#adata_all.write(filename='./data/adata_all_Scanorama.h5ad')
#adata_all = ad.read_h5ad('./data/adata_all_Scanorama.h5ad', chunk_size=100000)

# Cluster markers
#sc.tl.rank_genes_groups(adata_all, groupby = 'louvain', method='t-test', pts=True)
#RankMatrix_Pathml = sc.get.rank_genes_groups_df(adata_all, group=None)

##############################################
#Perform a clustering for scran normalization in clusters
#adata_pp = adata_combined_All.copy()
#sc.pp.log1p(adata_pp)
#sc.pp.scale(adata_pp, max_value=10)
#sc.tl.pca(adata_pp)
#sc.pp.neighbors(adata_pp, n_neighbors=15, n_pcs=40)
#sc.tl.louvain(adata_pp, key_added='groups', resolution=3)
#Preprocess variables for scran normalization
#input_groups = adata_pp.obs['groups']

#data_mat = adata_combined_All.X.T
#data_mat = data_mat.todense()

#input_groups.to_csv("data/input_groups.csv")
#pd.DataFrame(data_mat).to_csv("data/data_mat.csv")

#size_factors = pd.read_csv("data/size_factors.csv")
#size_factors.index = size_factors['Unnamed: 0']
#size_factors = size_factors.x

# Visualize the estimated size factors
#adata_combined_All.obs['size_factors'] = size_factors

#sc.pl.scatter(adata_combined_All, 'size_factors')
#sc.pl.scatter(adata_combined_All, 'size_factors')

#sb.distplot(size_factors, bins=50, kde=False)
#plt.show()

#Keep the count data in a counts layer
#adata_combined_All.layers["counts"] = adata_combined_All.X.copy()

#Normalize adata
#adata_combined_All.X /= adata_combined_All.obs['size_factors'].values[:,None]
#adata_combined_All.X = scipy.sparse.csr_matrix(adata_combined_All.X)

# Store the full data set in 'raw' as log-normalised data for statistical testing
#adata_combined_All.raw = adata_combined_All

#sc.pp.log1p(adata_combined_All)

# Batch correction
#sc.pp.combat(adata_combined_All, key='Region')

#sc.pp.scale(adata_combined_All, max_value=10)

#sc.tl.pca(adata_combined_All)
#sc.pp.neighbors(adata_combined_All, n_neighbors=15, n_pcs=40)
#sc.tl.umap(adata_combined_All)
# louvain clustering
#with parallel_backend('threading', n_jobs=15):
#    sc.tl.louvain(adata_combined_All, resolution = 3)
#adata_combined_All.obs['louvain'].value_counts()

#sc.tl.rank_genes_groups(adata_combined_All, groupby = 'louvain', method='t-test', pts=True)

###################################
## Normal scanpy pipline with harmony
sc.pp.normalize_total(adata_combined_All, target_sum=1e3)
sc.pp.log1p(adata_combined_All)
sc.pp.scale(adata_combined_All, max_value=10)

#pd.crosstab(adata_combined_All.obs["Region"], adata_combined_All.obs["patients"])
#sc.pp.combat(adata_combined_All, key='Region')

# PCA and batch correction
sc.tl.pca(adata_combined_All)
#sc.pl.pca_variance_ratio(adata_combined_All, n_pcs=40, log=True)
sc.external.pp.harmony_integrate(adata_combined_All, key='Region')

adata_combined_All.write(filename='./data/loomCombined_harmony.h5ad')

adata_combined_All = ad.read_h5ad('./data/loomCombined_harmony.h5ad', chunk_size=100000)

# use scVI latent space for UMAP generation
sc.pp.neighbors(adata_combined_All, n_neighbors=15, n_pcs=30, use_rep='X_pca_harmony')

# compute Umap embedding
sc.tl.umap(adata_combined_All)

# louvain clustering
with parallel_backend('threading', n_jobs=15):
    sc.tl.louvain(adata_combined_All, resolution = 5)

adata_combined_All.obs['louvain'].value_counts()

# Save results
adata_combined_All.write(filename='./data/loomCombined_clusters_res3.h5ad')
adata_combined_All = ad.read_h5ad('./data/loomCombined_clusters_res3.h5ad', chunk_size=100000)


# Plot UMAP embedding
#adata_combined_All.obs['Region'] = adata_combined_All.obs['Region'].astype(str)
sc.pl.umap(adata_combined_All, color=['Region'])
sc.pl.umap(adata_combined_All, color=['TMA'])
sc.pl.umap(adata_combined_All, color=['patients'])
sc.pl.umap(adata_combined_All, color=['groups'])
sc.pl.umap(adata_combined_All, color=['louvain'])

with rc_context({'figure.figsize': (10, 10)}):
sc.pl.umap(adata_combined_All, color='louvain', save= 'Umap_LouvainClusters3.png')

# Cluster markers
sc.tl.rank_genes_groups(adata_combined_All, groupby = 'louvain', method='wilcoxon', pts=True)
RankMatrix_Pathml = sc.get.rank_genes_groups_df(adata_combined_All, group=None)


marker_genes_dict = {
    'tumor cells': ['p53', 'beta-cat', 'Cytokeratin', 'Ki67', 'MUC-1-epithelia'],
    'stroma': ['Vimentin', 'Podoplanin-lymphatics', 'CD44-stroma'],
    'smooth muscles': ['aSMA-smooth muscle', 'Podoplanin-lymphatics'],
    'adipocytes': ['p53', 'Vimentin'],
    'CD68+CD163+ macrophages': ['CD68-macrophages', 'CD163-macrophages'],
    'CD68+ macrophages': ['CD68-macrophages'],
    'CD68+ macrophages GzmB+': ['CD68-macrophages', 'Granzyme B-cytotoxicity'],
    'CD11b+CD68+ macrophages': ['CD11b-macrophages', 'CD68-macrophages'],
    'CD163+ macrophages': ['CD163-macrophages'],
    'granulocytes': ['CD15-granulocytes', 'CD11b-macrophages'],
    'CD11b+ monocytes': ['CD11b-macrophages', 'GATA3-Th2', 'T-bet-Th1'],
    'CD3+ T cells': ['CD3-Tcells', 'GATA3-Th2', "CD2-Tcells"],
    'CD4+ T cells': ['CD3-Tcells', 'CD4', 'CD45-hematopoietic', 'BCL-2-apoptosis'],
    'CD4+ T cells CD45RO+': ['CD3-Tcells', 'CD4', 'CD45RO-memory cells'],
    'CD4+ T cells GATA3+': ['CD4', 'GATA3-Th2'],
    'CD8+ T cells': ['CD8', 'CD3-Tcells'],
    'Tregs': ['CD25-IL-2Ra', 'FOXP3-Treg'],
    'B cells': ['CD20', 'CD45-hematopoietic', 'BCL-2-apoptosis'],
    'plasma cells': ['CD38-multifunctional', 'VISTA-costimulator', 'CD138-plasma cells'],
    'vasculature': ['CD31-vasculature', 'CD34-vasculature', 'Vimentin'],
    'NK cells': ['CD56-NK cells'],
    'immune cells': ['CD45RA-naiveT', 'Granzyme B-cytotoxicity', 'CD2-Tcells', 'PD-L1', 'PD-1-checkpoint'],
    'immune cells / vasculature': ['CD45RA-naiveT', 'CD30-costimulator', 'PD-L1', 'CD31-vasculature', 'CD34-vasculature'],
    'tumor cells / immune cells': ['LAG-3-checkpoint', 'T-bet-Th1', 'CD44-stroma', 'CD138-plasma cells', 'p53', 'Cytokeratin', 'Ki67'],
    'nerves': ['GFAP-nerves', 'Synaptophysin-neuroendocrine', 'ChromograninA-neuroendocrine'],
    'lymphatics': ['Podoplanin-lymphatics', 'CD31-vasculature', 'CD34-vasculature']
}


cell_annotation = sc.tl.marker_gene_overlap(adata_combined_All, marker_genes_dict, top_n_markers=10)
cell_annotation_norm = sc.tl.marker_gene_overlap(adata_combined_All, marker_genes_dict, top_n_markers=20, normalize='reference')



sc.pl.rank_genes_groups_dotplot(adata_combined_All, groupby='louvain', n_genes=5, dendrogram = False, cmap='bwr', values_to_plot = 'logfoldchanges', vmin=-4, vmax=4, save='LouvainMarkersDotplot_5_wilcoxon.png')

# Save the processed anndata object
adata_combined_All.write(filename='./data/adataCombined_All_processed.h5ad')

# read the processed anndata object
adata_combined_All = ad.read_h5ad('./data/adataCombined_All_processed.h5ad', chunk_size=100000)


####################
## Annotate the clusters

#sc.pl.violin(Nolan_adata, ['MUC-1-epithelia', 'p53', 'Ki67', 'aSMA-smooth muscle'], groupby='clusters')
#sc.pl.violin(adata_combined_All, ['CD68-macrophages', 'CD163-macrophages', 'CD3-Tcells'], groupby='leiden')



# Fill in the clusters that belong to each cell type based on each marker in the plot above
cell_dict = {'tumor': ['8'],
             'stroma': ['0'],
             'B cells': ['1', '5', '10'],
             'CD68+CD163+ macrophages': ['2'],
             'smooth muscles': ['3', '7'],
             'tumor/immune': ['4'],
             'CD38+CD68+ macrophages': ['6'],
             'granulocytes': ['9'],
             'unknown': ['11'],
             'immune': ['12'],
             'artifact':['13'],
             'plasma cells':['14'],

             }

# Initialize empty column in cell metadata
adata.obs['merged'] = np.nan

# Generate new assignments
for i in cell_dict.keys():
    ind = pd.Series(adata.obs.leiden).isin(cell_dict[i])
    adata.obs.loc[ind,'merged'] = i

new_cluster_names = [
    'immune0',
    'granulocytes1',
    'unknown2',
    'unknown3',
    'Bcells4',
    'smooth_muscle5',
    'stroma6',
    'Bcells7',
    'Bcells8',
    'CD4TcellsCD45RO_9',
    'granulocytes10',
    'tumor_immune11',
    'vasculature12',
    'CD4TcellsCD45RO_13',
    'vasculature14',
    'tumor_immune15',
    'smooth_muscle16',
    'tumor17',
    'tumor_immune18',
    'tumor_immune19',
    'granulocytes20',
    'immune_vasculature21',
    'plasma_cells22',
    'immune_vasculature23',
    'unknown24',
    'dirt25',
    'CD68CD163macrophages26',
    'CD68CD163macrophages27',
    'lymphatic28',
    'plasma_cells29',
    'stroma30',
    'dirt31',
    'immune32',
    'epithelial_cells33',
    'epithelial_immune34',
    'dirt35',
    'CD4TcellsGATA3_36',
    'epithelial_immune37',
    'CD4TcellsGATA3_38',
    'Naive_memory_Tcells39',
    'nerves40',
    'CD68macrophagesGzmB41'
]


adata_combined_All.rename_categories('louvain',  new_cluster_names)

## Cell types Annotation
old_to_new = dict(
    CD45_CD4_Tcells0='CD45+CD4+ T cells',
    TILs_TAMs1='TILs/TAMs',
    smooth_muscle2='smooth muscles',
    CD38_CD68_macrophages3='CD38+CD68+ macrophages',
    tumor4='tumor',
    Bcells5='B cells',
    NK_cells6='NK cells',
    tumor_immune7='tumor/immune',
    TILs8='TILs',
    TILs9='TILs',
    Tregs10='Tregs',
    unknown11='undefined',
    immune_vasculature12='immune/vasculature',
    tumor_immune_vasculature13='tumor/immune/vasculature',
    granulocytes14='granulocytes',
    immune_vasculature15='immune/vasculature',
    CD11bmonocytes16='CD11b+ monocytes',
    Tregs17='Tregs',
    Bcells18='B cells',
    CD4TcellsCD45RO19='CD4+CD45RO+ T cells',
    granulocytes20='granulocytes',
    immune_vasculature21='immune/vasculature',
    plasma_cells22='plasma cells',
    immune_vasculature23='immune/vasculature',
    unknown24='undefined',
    dirt25='dirt',
    CD68CD163macrophages26='CD68+CD163+ macrophages',
    CD68CD163macrophages27='CD68+CD163+ macrophages',
    lymphatic28='lymphatic',
    plasma_cells29='plasma cells',
    stroma30='stroma',
    dirt31='dirt',
    immune32='immune cells',
    epithelial_cells33='epithelial cells',
    epithelial_immune34='epithelial/immune',
    dirt35='dirt',
    CD4TcellsGATA3_36= 'CD4+GATA3+ T cells',
    epithelial_immune37= 'epithelial/immune',
    CD4TcellsGATA3_38 = 'CD4+GATA3+ T cells',
    Naive_memory_Tcells39= 'Naive/memory T cells',
    nerves40= 'nerves',
    CD68macrophagesGzmB41= 'CD68+GzmB+ macrophages'
)

adata_combined_All.obs['cell_types'] = (
    adata_combined_All.obs['louvain']
    .map(old_to_new)
    .astype('category')
)

adata_combined_All.obs['louvain'].value_counts()
adata_combined_All.obs['cell_types'].value_counts()

# Save the anndata object with cell types
adata_combined_All.write(filename='./data/adataCombined_cell_types.h5ad')

# read the processed anndata object
adata_combined_All = ad.read_h5ad('./data/adataCombined_cell_types.h5ad', chunk_size=100000)



##################

##############################################
#import squidpy as sq
#sc.set_figure_params(dpi=200, frameon= True, figsize=(15,15), format='png', fontsize=12)
#sc.pl.spatial(adata_combined_All, color='louvain', spot_size=15)
#sc.pl.spatial(adata_combined_All, color='Region', spot_size=15)
#sc.pl.spatial(dataset_A.slides[0].counts, color='leiden', spot_size=15)
#sc.pl.spatial(adata_combined_All, color="leiden", groups=["2","4"], spot_size=15)


#sq.gr.spatial_neighbors(adata_combined_All)
#sq.gr.nhood_enrichment(adata_combined_All, cluster_key="louvain")
#sq.pl.nhood_enrichment(adata_combined_All, cluster_key="louvain", method="ward", figsize=(30,15), dpi=200, save='NeighberhoodEnrichment.png')
#plt.show()

###################################
## Our data using pathml
LoomData = adata_combined_All.to_df()
LoomData['CellID'] = np.linspace(0, 282577, num= 282578).astype(int)
LoomData.set_index('CellID', inplace=True)

LoomObs = adata_combined_All.obs
LoomObs['CellID'] = np.linspace(0, 282577, num= 282578).astype(int)
LoomObs.set_index('CellID', inplace=True)

loomDataAnn = pd.concat([LoomObs, LoomData], axis = 1)
loomDataAnn['CellID'] = loomDataAnn.index
loomDataAnn.shape

loomDataAnn['TMA'].value_counts()

# How many cell types we have?
len(loomDataAnn['cell_types'].unique())

# save
loomDataAnn.to_csv('./data/CRC_pathml.csv')




