## SMDPG: Identifying Metabolite-Disease Associations via Optimized Negative Sampling and Sparse Graph Convolutional Network



**requirement:**

python						3.8.6

numpy                        1.24.1

pandas                        2.0.3

torch                           1.8.1+cu101

torch-geometric        2.2.0



**Run our code:**

1. Run `LowSimilarityNegativeSampling.cpp` to get new negative samples.

2. Run `MDPGTConstruction.cpp` to construct the sparse homogeneous graph MDPGT.
3. Run `PCA.py` to downscale node features by PCA.
4. Finally, run `main_balanced.py` or `main_unbalanced.py` to build model and 10-cross validation. AUPR、AUC、F1 and ACC would be given.

