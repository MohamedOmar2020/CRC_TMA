
# perform tensor decomposition after each single cell has been allocated to a Cellular Neighborhood and Cell Type.

from tensorly.decomposition import non_negative_parafac,parafac,non_negative_tucker,partial_tucker,tucker
from tensorly.regression import cp_regression
import tensorly as tl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import itertools
from tensorly.regression.cp_regression import KruskalRegressor

'''
x needs to be a dataframe which says 
for each patient, neighborhood and cell type, how many cells of that type in that neighborhood there are

cells of interest are the cells to be used in the decomposition
CNs is the CNs to be used in the decomposition
patients are the patients to be used in the decomposition
'''

x = pd.read_csv('data/CRC_pathml_Neighbors.csv')
x['patients'].value_counts()

clr = x[x["groups"] == "CLR"]
dii = x[x["groups"] == "DII"]

neigh_col = 'neighborhood10'

cells_of_interest = [#'CD45+CD4+ T cells',
       'CD4+CD45RO+ T cells',
       #'CD4+GATA3+ T cells',
       #'Naive/memory T cells',
       'TILs/TAMs', 'TILs', 'Tregs',
       'granulocytes', #'CD11b+ monocytes',
       'CD68+CD163+ macrophages', #'CD68+GzmB+ macrophages',
       #'CD38+CD68+ macrophages',
       'plasma cells', 'B cells', #'NK cells',
       'immune cells',
       'smooth muscles', 'stroma',
       #'lymphatic',
       'immune/vasculature',
       'tumor', 'tumor/immune', 'tumor/immune/vasculature',
       'epithelial cells', 'epithelial/immune'
       #'nerves',
       #'dirt', 'undefined'
    ]

#['B cells', 'TILs/TAMs','TILs', 'Tregs']
nbs = [0,1,2,3,4,5,6,7,8,9]
#patients = [1,2,3,4,5,6,7]

patients = np.linspace(1, 35, num= 35).astype(int)
patients_clr = clr['patients'].unique()
patients_dii = dii['patients'].unique()

def build_tensor(patients, x, nbs, cells_of_interest):
    T = np.zeros((len(patients), len(nbs), len(cells_of_interest)))
    for i, nb in enumerate(nbs):
        for j, chk in enumerate(cells_of_interest):
            interm = x.loc[x['neighborhood10'] == nb, :]
            interm = interm.set_index('patients')
            try:
                interm = interm.loc[patients, chk].fillna(0).values
                T[:, i, j] = np.sum(interm)
            except KeyError:
                T[:, i, j] = 0

    # normalize each patient's frequencies
    dat = np.nan_to_num(T / T.sum((1, 2), keepdims=True))
    return dat

def decomposition_elbow(dat):
    pal = sns.color_palette('bright',10)
    palg = sns.color_palette('Greys',10)
    mat1 = np.zeros((5,15))
    #finding the elbow point
    for i in range(2,15):
        for j in range(1,5):
            facs_overall = non_negative_tucker(dat,rank=[j,i,i],random_state = 2336)
            mat1[j,i] = np.mean((dat- tl.tucker_to_tensor(tucker_tensor = (facs_overall[0],facs_overall[1])))**2)

    #figsize(10,5)
    plt.plot(2+np.arange(13),mat1[2][2:],c = 'red',label = 'rank = (2,x,x)')
    plt.plot(2+np.arange(13),mat1[1][2:],c = 'blue',label = 'rank = (1,x,x)')
    plt.xlabel('x')
    plt.ylabel('error')
    plt.show()


def tissue_module_plots(dat, person_rank, rank, nbs, cells_of_interest, random_state=0):
    facs_overall = non_negative_tucker(dat, rank=[person_rank, rank, rank], random_state=random_state)
    print(facs_overall[0].shape)
    sns.heatmap(pd.DataFrame(facs_overall[1][1], index=nbs))
    plt.ylabel('CN')
    plt.xlabel('CN module')
    plt.title('CN modules')
    plt.show()

    sns.heatmap(pd.DataFrame(facs_overall[1][2], index=cells_of_interest))
    plt.ylabel('CT')
    plt.xlabel('CT module')
    plt.title('CT modules')
    plt.show()

    print('--------Tissue modules ---------')
    for i in range(person_rank):
        sns.heatmap(pd.DataFrame(facs_overall[0][i]))
        plt.ylabel('CN module')
        plt.xlabel('CT module')
        plt.title('Tissue module {}'.format(i))
        plt.show()

    return facs_overall


dat_clr = build_tensor(patients_clr,clr,nbs,cells_of_interest)
dat_dii = build_tensor(patients_dii,dii,nbs,cells_of_interest)

#compute elbow point for decomposition
decomposition_elbow(dat_clr)
decomposition_elbow(dat_dii)

#compute CN modules, CT modules and couplings
facs_overall_clr = tissue_module_plots(dat_clr,2,7,nbs,cells_of_interest)
facs_overall_dii = tissue_module_plots(dat_dii,2,7,nbs,cells_of_interest)


#######################################################################################
def tensor_plots(dat, scale=0.4,  person_rank = 2, rank = 7, savename=None):
    pal = sns.color_palette('bright', 10)
    palg = sns.color_palette('Greys', 10)
    mat1 = np.zeros((5, 15))
    # finding the elbow point
    for i in range(2, 15):
        for j in range(1, 5):
            facs_overall = non_negative_tucker(dat, rank=[j, i, i], random_state=2336)
            mat1[j, i] = np.mean((dat - tl.tucker_to_tensor(facs_overall) ** 2))

    facs_overall = non_negative_tucker(dat, rank=[person_rank, rank, rank], random_state=32)

    #figsize(10, 5)
    plt.plot(2 + np.arange(13), mat1[2][2:], c='red', label='rank = (2,x,x)')
    plt.plot(2 + np.arange(13), mat1[1][2:], c='blue', label='rank = (1,x,x)')
    plt.xlabel('x')
    plt.ylabel('error')
    plt.show()

    #figsize(3.67 * scale, 2.00 * scale)
    nb_scatter_size = scale * scale * 45
    cel_scatter_size = scale * scale * 15

    # script to draw the tissue modules (requires fine tuning for rescaling/positioning)
    for p in range(person_rank):
        for idx in range(rank):
            an = float(np.max(facs_overall[0][p][idx, :]) > 0.1) + (np.max(facs_overall[0][p][idx, :]) <= 0.1) * 0.05
            ac = float(np.max(facs_overall[0][p][:, idx]) > 0.1) + (np.max(facs_overall[0][p][:, idx]) <= 0.1) * 0.05

            nb_fac = facs_overall[1][1][:, idx]
            cel_fac = facs_overall[1][2][:, idx]

            cols_alpha = [(*pal[nb], an * np.minimum(nb_fac, 1.0)[i]) for i, nb in enumerate(nbs)]
            cols = [(*pal[nb], np.minimum(nb_fac, 1.0)[i]) for i, nb in enumerate(nbs)]
            cell_cols_alpha = [(0, 0, 0, an * np.minimum(cel_fac, 1.0)[i]) for i, _ in enumerate(cel_fac)]
            cell_cols = [(0, 0, 0, np.minimum(cel_fac, 1.0)[i]) for i, _ in enumerate(cel_fac)]

            plt.scatter(0.5 * np.arange(len(nb_fac)), 5 * idx + np.zeros(len(nb_fac)), c=cols_alpha,
                        s=nb_scatter_size)  # ,edgecolors=len(cols_alpha)*[(0,0,0,min(1.0,max(0.1,2*an)))], linewidths= 0.05)
            offset = 9
            for i, k in enumerate(nbs):
                pass
                # plt.text(0.5*i, 5*idx, k,fontsize = scale*2,ha = 'center', va = 'center',alpha = an)

            plt.scatter(-4.2 + 0.25 * np.arange(len(cel_fac)) + offset, 5 * idx + np.zeros(len(cel_fac)),
                        c=cell_cols_alpha,
                        s=0.5 * cel_scatter_size)  # ,vmax = 0.5,edgecolors=len(cell_cols_alpha)*[(0,0,0,min(1.0,max(0.1,2*an)))], linewidths= 0.05)

            # For the left rect
            rect = plt.Rectangle((-0.35, 5 * idx - 2), 4.5, 4, linewidth=scale * scale * 1, edgecolor='black',
                                 facecolor='none', zorder=0, alpha=an, linestyle='--')
            ax = plt.gca()
            ax.add_artist(rect)
            plt.scatter([offset - 5], [5 * idx], c='black', marker='D', s=scale * scale * 5, zorder=5, alpha=an)
                         #plt.text(offset-5,5*idx,idx,color = 'white',alpha = an, ha = 'center', va = 'center',zorder = 6,fontsize = 4.5)
            plt.scatter([offset - 4.5], [5 * idx], c='black', marker='D', s=scale * scale * 5, zorder=5, alpha=ac)
                         #plt.text(offset-4.5,5*idx,idx,color = 'white',alpha = ac, ha = 'center', va = 'center', zorder = 6,fontsize = 4.5)

            # For the right rect
            rect = plt.Rectangle((4.5, 5 * idx - 2), 4.5, 4, linewidth=scale * scale * 1, edgecolor='black',
                                 facecolor='none', zorder=0, alpha=ac, linestyle='-.')
            ax.add_artist(rect)

        #        cols = lcols
        #         for i,chk in enumerate(cell_cols):
        #                 plt.text(-4+offset+0.2*i, 27.5, chk, rotation = 45, color = 'black',ha = 'left', va = 'bottom',fontsize = scale*15,alpha = ac)
        for nb_i in range(rank):
            for cel_i in range(rank):
                plt.plot([4.5, 4.1], [5 * nb_i, 5 * cel_i], color='black',
                         linewidth=2 * scale * scale * 1 * min(1.0, max(0, -0.00 + facs_overall[0][p][nb_i, cel_i])),
                         alpha=min(1.0, max(0.000, -0.00 + 10 * facs_overall[0][p][nb_i, cel_i])))  # max(an,ac))

        plt.ylim(-5, 35)
        plt.axis('off')

        if savename:
            #plt.savefig(savename + '_%d.pdf' % p)
            plt.savefig(savename + '_%d.png' % p)
        plt.show()



# Person rank refers to how many compartments you want (eg, tumor compartment, immune compartment, etc)
tensor_plots(dat_clr,scale = 1, person_rank = 2, rank = 7, savename = 'clr_updated')

tensor_plots(dat_dii,scale = 1, person_rank = 2, rank = 7, savename = 'dii_updated')











