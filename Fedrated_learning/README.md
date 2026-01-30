# Saving, Visualizing, and Deploying a Federated Learning Model

## Overview
This module focuses on post-training workflows for a Federated Learning system.  
Rather than improving model accuracy, the emphasis is on usability, observability, and reusability of an already trained global model.

The implementation demonstrates how trained models are stored, inspected, and visualized in real-world machine learning systems.

---

## Objectives
The goals of this module are to:
- Persist the final global federated model to disk
- Log training metrics for post-training analysis
- Visualize federated training behavior using a lightweight dashboard
- Separate training, evaluation, and deployment workflows
- Mimic production-oriented ML engineering practices

---

## Project Structure
Fedrated_learning/
│
├── models/
│ └── global_model.pth
│
├── logs/
│ └── metrics.csv
│
├── app.py
├── train_federated.py
├── requirements.txt
└── README.md


---

## Model Persistence
After federated training completes, the global model parameters are saved using PyTorch.

```python
torch.save(global_model.state_dict(), "models/global_model.pth")
Visualization and Monitoring

A lightweight Streamlit application is used to visualize federated training metrics and summarize model performance.
This separates training from analysis and provides an easy-to-understand interface for inspecting results.

Features

Loads saved training metrics from disk

Plots global model accuracy across federated rounds

Displays final model performance summary

Safely handles missing or empty log files

Running the Application

To launch the Streamlit dashboard, run:

streamlit run app.py

Dependencies

All required dependencies for the project can be installed using:

pip install -r requirements.txt


Key libraries used include:

PyTorch – Model training, evaluation, and persistence

Flower (flwr) – Federated learning framework

Pandas and Matplotlib – Data analysis and visualization

Streamlit – Deployment and monitoring interface
