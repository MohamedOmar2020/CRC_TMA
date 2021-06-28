

rm(list = ls())

library(Seurat)
library(SeuratDisk)
library(zellkonverter)
library(MAST)
library(dplyr)

##########################
## Convert the h5ad anndata file to h5 seurat
Convert('./data/loomCombined_processed.h5ad', dest = "h5seurat", overwrite = FALSE)

# Load the converted file
adata_combined_All <- LoadH5Seurat("./data/loomCombined_processed.h5seurat")

Idents(adata_combined_All) <- adata_combined_All$louvain
table(adata_combined_All$louvain)

# Find differentially expressed features between CD14+ and FCGR3A+ Monocytes
CRC_Mast_Markers <- FindAllMarkers(adata_combined_All, test.use = 'MAST')

save(CRC_Mast_Markers, file = './objs/CRC_Mast_Markers.rda')

# view results
head(CRC_Mast_Markers)

CRC_Mast_Markers_grouped <- CRC_Mast_Markers %>%
                            group_by(cluster) %>%
                            arrange(desc(avg_log2FC), .by_group = TRUE) %>%  
                            top_n(n = 10, wt = avg_log2FC)


FeaturePlot(adata_combined_All, features = c("p53", "beta-cat", "MUC-1-epithelia", "Vimentin"))

CRC_Mast_Markers %>%
  group_by(cluster) %>%
  top_n(n = 10, wt = avg_log2FC) -> top10

DoHeatmap(adata_combined_All, features = top10$gene) + NoLegend()

##########################################
## Annotate the clusters into cell types
new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", "FCGR3A+ Mono",
"NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(adata_combined_All)
adata_combined_All <- RenameIdents(adata_combined_All, new.cluster.ids)
DimPlot(adata_combined_All, reduction = "umap", label = TRUE, pt.size = 0.5) + NoLegend()








