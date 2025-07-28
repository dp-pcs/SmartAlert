# SmartAlert AI

This project trains a machine learning model to predict critical incidents from Splunk-style logs.

## Getting Started

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate.bat on Windows
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook to train models:
    ```bash
    jupyter notebook notebooks/01_Train_Models.ipynb
    ```

## Project Structure

- `data/`: Contains synthetic Splunk log dataset
- `scripts/`: CLI training and evaluation scripts
- `notebooks/`: Interactive training and visualization
- `models/`: Folder for saved models
- `utils/`: Feature engineering utilities
