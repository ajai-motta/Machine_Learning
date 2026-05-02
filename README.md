# Student Performance Prediction ML Pipeline

A machine learning project for predicting student performance using multiple regression models. The pipeline automates data ingestion, transformation, and model training with automatic model selection based on R² score.

## Features

- **Data Ingestion**: Automated data loading and train-test splitting
- **Data Transformation**: Preprocessing and feature engineering pipeline
- **Multiple Models**: Trains and compares 6 different regression models
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors Regressor
  - CatBoost Regressor
- **Automatic Model Selection**: Selects best model based on R² score
- **Production Pipeline**: Separate training and prediction pipelines
- **Error Handling**: Custom exception handling and comprehensive logging

## Project Structure

```
project1/
├── src/
│   ├── components/
│   │   ├── data_ingetion.py        # Data loading and splitting
│   │   ├── data_transformation.py   # Feature preprocessing
│   │   └── model_traing.py          # Model training and selection
│   ├── pipeline/
│   │   ├── training_pipeline.py     # Training workflow orchestration
│   │   └── predict_pipeline.py      # Prediction workflow
│   ├── logger.py                    # Logging configuration
│   ├── exception.py                 # Custom exception handling
│   ├── utils.py                     # Utility functions
│   └── __init__.py
├── notebooks/
│   └── data/
│       └── student_data.csv         # Input data
├── artifact/                        # Generated artifacts
│   ├── train.csv                    # Training data
│   ├── test.csv                     # Test data
│   ├── data.csv                     # Raw data
│   └── mymodel.pkl                  # Trained model
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project1
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate      # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- pandas
- numpy
- seaborn
- dill
- catboost
- scikit-learn

## Usage

### Training a Model

Run the data ingestion component to load data, transform it, and train models:

```python
from src.components.data_ingetion import DataIngestion
from src.components.data_transformation import DataTransformer
from src.components.model_traing import ModelTriner

# Data Ingestion
ingestion = DataIngestion()
raw_data, test_data, train_data = ingestion.initiate_data_ingection()

# Data Transformation
transformer = DataTransformer()
train_arr, test_arr, preprocesser_path = transformer.int_data_transform(train_data, test_data)

# Model Training
trainer = ModelTriner()
trainer.initate_model_train(train_arr, test_arr, preprocesser_path)
```

### Making Predictions

Use the prediction pipeline for new data:

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Create pipeline and make predictions
pipeline = PredictionPipeline()
predictions = pipeline.predict(new_data)
```

## Data Requirements

The input dataset should contain:
- Student performance metrics
- Features relevant to student outcomes
- Target variable for regression

Expected format: CSV file with headers

## Model Evaluation

Models are evaluated using:
- **R² Score**: Primary metric for model selection
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

The model with the highest R² score on the test set is automatically selected as the best model.

## Output

- **Trained Model**: `artifact/mymodel.pkl`
- **Preprocessor**: Saved for consistent data transformation
- **Logs**: Detailed execution logs with training information

## Logging

All processes are logged to track:
- Data ingestion steps
- Data transformation progress
- Model training results
- Model evaluation metrics
- Errors and exceptions

## Error Handling

Custom exception handling ensures:
- Clear error messages
- Complete stack traces
- Graceful failure recovery

## Future Improvements

- [ ] Add hyperparameter tuning
- [ ] Implement cross-validation
- [ ] Add feature importance analysis
- [ ] Develop API for model serving
- [ ] Add model versioning
- [ ] Implement A/B testing framework

## Contributing

1. Create a feature branch
2. Make your changes
3. Ensure tests pass
4. Submit a pull request

## License

[Specify your license here]

## Author

Ajai Joseph

## Contact

For questions or issues, please open a GitHub issue.