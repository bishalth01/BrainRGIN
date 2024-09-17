# BrainRGIN (Brain ROI Aware Graph Isomorphism-Networks )

Description:

Extending from the existing graph convolution networks, our approach incorporates a clustering-based embedding and graph isomorphism network method in the graph convolutional layer to reflect nature of the brain sub-network organization and efficient expression, in combination with TopK pooling and attention-based readout functions.

This work is accepted in Medical Image Analysis. The DOI for citation will be updated soon.

Creating Graphs:

We used DGL to create the graphs from the pytorch Data object. The sample details are in save_dgl_graphs/* repository.



Tips to run the code:

Main File: hcp_main/main_rgin
Hyperparameters: wandb_sweeps/example_config

