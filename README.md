# Graph Convolutional Network for Ship Noise Prediction

## Project Overview

This study presents an innovative framework combining mobile observations with machine learning to predict river shipping noise spatial distribution. The model constructs a Graph Convolutional Network (GCN) based on surface airborne noise data collected along the Yangtze River using self-developed Acoustic Drifters. This framework enables the generation of high-resolution, near-real-time noise maps, providing scientific evidence for assessing the cumulative impacts of shipping noise on ecosystems and human communities.

## Model Architecture

The model employs a graph neural network architecture with the following core components:

- **Graph Structure**: Each timestamp-geolocation combination forms a graph containing one hydrophone node and multiple ship nodes
- **Node Features**:
  - Vessel nodes: distance to test point, ship speed, ship length, direction cosine, etc.
  - Acoustics Drifter node: geographic location information
- **Edge Features**: Distance weight (distanceW = 1/distance)
- **Network Layers**: 3 SAGEConv graph convolution layers + output MLP
- **Loss Function**: Mean Squared Error (MSE)

## Requirements

### Dependencies

Python >= 3.8
PyTorch >= 1.10.0
PyTorch Geometric >= 2.0.0
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
