

rm(list = ls())

library(scran)
library(data.table)

input_groups = read.csv("data/input_groups.csv")
rownames(input_groups) <- input_groups$X
input_groups$X <- NULL
input_groups <- as.matrix(input_groups)

data_mat = fread("data/data_mat.csv", header = T, data.table = F)
rownames(data_mat) <- data_mat$V1
data_mat$V1 <- NULL

data_mat <- as.matrix(data_mat)

size_factors = calculateSumFactors(data_mat, clusters=input_groups)
names(size_factors) <- rownames(input_groups)
write.csv(size_factors, file = "data/size_factors.csv")



