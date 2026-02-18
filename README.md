# AI-Based Water Quality Monitoring System

A Machine Learning project to predict water safety using pH and Solids (TDS) values.

## Project Structure
- `model_training.ipynb`: Jupyter Notebook to train the Random Forest model.
- `app.py`: Streamlit web application for predictions.
- `water_model.pkl`: Trained model file (generated after running the notebook).
- `requirements.txt`: List of Python dependencies.

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Dataset**:
    - Download `water_potability.csv` from Kaggle.
    - Place it in this directory.

3.  **Train the Model**:
    - Open `model_training.ipynb` in Jupyter Notebook.
    - Run all cells to generate `water_model.pkl`.

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
