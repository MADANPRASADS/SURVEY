# Survey Personalization Project

## Overview
This project aims to personalize survey questions based on dynamic user responses using machine learning techniques. It involves clustering users into personality types and adapting survey questions to improve engagement and data accuracy.

## Contents
- `data-final.csv`: Dataset used for analysis (not included in this repository).
- `scripts/`: Python scripts for various stages of the project.
  - `data_loading.py`: Script to load and preprocess the dataset.
  - `data_visualization.py`: Script for visualizing data and personality traits.
  - `clustering.py`: Script for K-Means clustering and PCA analysis.
  - `model_training.py`: Script for training the Random Forest Classifier.
  - `survey_personalization.py`: Script for user interaction and survey personalization.
- `visualizations/`: Folder containing key visualizations generated during analysis.
  - `histograms.png`: Histograms showing distribution of personality traits.
  - `pca_plot.png`: PCA plot visualizing clusters in 2D space.

## Requirements
- Python 3
- Required Python packages are listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/survey-personalization.git
   cd survey-personalization
-