# Data Quality Driven MLOps
This repository contains the code and resources used in the research study "Data Quality driven MLOps: Investigating the Telco Customer Churn Dataset". The study focuses on data-centric strategies in Machine Learning Operations (MLOps), emphasizing the importance of data quality in enhancing machine learning performance.

## Abstract
In the contemporary data-driven landscape, the quality of data emerges as a critical factor for optimizing machine learning (ML) operations. This research shifts the focus from traditional model-centric approaches to data-centric strategies, emphasizing the role of data quality in enhancing ML performance. By investigating the Telco customer churn dataset, the study simulates real-world data anomalies and assesses their impact on ML outcomes. Three pivotal tasks in the MLOps process are explored, leading to the formulation of data-centric strategies that demonstrably improve predictive accuracy, even amidst significant data anomalies. The findings serve as a practical guide for practitioners, advocating for a robust data pipeline that can navigate data quality challenges and unlock the full potential of data assets. The study’s contributions include the identification of data quality challenges, the development of data-centric strategies, and the provision of a pragmatic guide for industry application. The research thus bridges the gap between data quality and ML operations, offering a resilient and evidence-backed approach to navigating data anomalies, with tangible improvements in predictive accuracy.

Key Contributions
The key contributions of this study include:

1. Identification of data quality challenges in MLOps.
2. Development of data-centric strategies for dealing with these challenges.
3. Provision of a practical guide for industry application of these strategies.

The research bridges the gap between data quality and machine learning operations, offering a resilient and evidence-backed approach to navigating data anomalies, with tangible improvements in predictive accuracy.

## Getting Started
To get started with the project, clone the repository, install the required dependencies listed in requirements.txt as separate environments, and follow the instructions provided in the scripts to replicate the experiments conducted in the study.

## Components of Data Quality Driven MLOps

| Note: The following have been customised for the project and the real output of the framework might be slighly different.

1. Snoopy

Snoopy is a tool used in the study, which aims to estimate the Bayes Error Rate (BER) of a given dataset. It operates as follows:

    Input: Data (tabular or text) and target accuracy.
    Output: Feasible or Not Feasible.
    
    How it works:
    1. Embedding: Snoopy first transforms the data into a high-dimensional space using pre-trained embeddings. This step converts raw data into a format that can be used for machine learning.

    2. Nearest Neighbors Search: For each point in the dataset, Snoopy finds its k nearest neighbors. The neighbors are found in the high-dimensional space created in the embedding step.

    3. Error Estimation: Snoopy then estimates the error rate of each point based on its nearest neighbors. If a point and its neighbors have different labels, it is considered an error. The BER is the average error rate across all points in the dataset.

2. CPClean (Certain Prediction Cleaning)
CPClean is a component used in the study for data cleaning based on the concept of entropy change and model fit and validation.

    Input: Dataset with predicted labels.
    Output: f1-score for Cleaned dataset with corrected labels

    How it works:
    1. Entropy Change: CPClean measures the entropy change in the predicted labels when certain predictions are modified or corrected. By analyzing the impact of these changes on the overall entropy of the dataset, CPClean identifies potential errors.
    2. Model Fit and Validation: CPClean uses machine learning models to validate the predictions and identify incorrect or uncertain predictions. It may utilize techniques such as cross-validation or model evaluation metrics to assess the quality of the predictions.

3. Modelpicker
The Modelpicker component mentioned in the study aims to assist in the selection of the most suitable machine learning model for a given task. 

    Input: Dataset and evaluation metrics.
    Output: Recommended machine learning model.
    
    How it works:
    Modelpicker analyzes the given dataset and evaluates the performance of different machine learning models using the provided evaluation metrics. By employing an online selective sampling approach, ease.ml Modelpicker can output the best model with high probability at any given round.

## Citation
If you use this code or the findings of the study in your work, please cite the following papers:

1. A data quality-driven view of MLOps
2. The effect of data quality on ML performances