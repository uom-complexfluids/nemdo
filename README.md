# NeMDO: Learning Mesh-Free Discrete Differential Operators 

[![Paper](https://img.shields.io/badge/OpenReview-vPQ9dpQ2rE-B31B1B.svg)](https://openreview.net/forum?id=vPQ9dpQ2rE)
[![Institution](https://img.shields.io/badge/University%20of%20Manchester-Complex%20Fluids%20Group-blue)](https://www.manchester.ac.uk/)

This repository contains the official implementation for the paper:  
**"Learning Mesh-Free Discrete Differential Operators with Self-Supervised Graph Neural Networks"**, presented in the *AI&PDE Workshop at ICLR 2026*.

## 📖 Overview
NEMDO introduces a self-supervised approach using Graph Neural Networks (GNNs) to learn discrete differential operators in mesh-free frameworks. By leveraging the local topology of particle distributions, the model provides an accurate, data-driven alternative to traditional numerical approximations of PDEs.

## 📂 Repository Structure
The project is organized into two main modules:

* **`/training`**: Contains scripts and configurations for training the Self-Supervised GNNs. This includes data pre-processing, graph construction, and the core training loops.
* **`/testing`**: Contains evaluation scripts to validate the learned operators against analytical solutions and traditional mesh-free methods (e.g., SPH or LABFM).

## 🚀 Getting Started

This repository is designed to be modular. Each module (`training` and `testing`) contains its own `requirements.txt` file. You can install everything into a single environment or maintain separate environments for each phase of the pipeline.

The requirements are located at **`training/requirements.txt`** and **`testing/requirements.txt`**.

## 💻 Usage
Once your environment(s) are set up, follow the specific instructions located in each subdirectory:

Refer to training/README.md for data generation and model optimization.
Refer to testing/README.md for evaluation and comparative analysis.
