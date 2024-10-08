# BrainRGIN (Brain ROI Aware Graph Isomorphism Networks)

## Description

BrainRGIN extends existing Graph Convolution Networks (GCNs) by incorporating clustering-based embeddings and a Graph Isomorphism Network (GIN) in the graph convolutional layers. This method better captures the structure and organization of brain sub-networks, providing efficient and expressive representations. Our approach combines **TopK pooling** and **attention-based readout** functions.

This work has been accepted for publication in *Medical Image Analysis*. The DOI for citation will be updated soon.

## Creating Graphs

We used **DGL (Deep Graph Library)** to create graphs from the PyTorch Data object. Sample details and graphs can be found in the `save_dgl_graphs/*` directory.

## How to Run the Code

- **Main File**: `hcp_main/main_rgin`
- **Hyperparameters**: Configurations for hyperparameter tuning are available in `wandb_sweeps/example_config`.

---

For questions or contributions, feel free to open an issue or create a pull request!
