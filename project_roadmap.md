# Project Syllabus: Palmer Penguins Classification Model

## Academic Objective
Develop a Supervised Machine Learning pipeline to classify penguin species based on physical biometric measurements. This project introduces core Scikit-learn functionality, specifically data splitting and classification algorithms, building upon existing Pandas data manipulation fundamentals.

## Phase 1: Environment & Data Ingestion
**Objective:** Establish the workspace and load the raw dataset into memory.
* Initialize the local repository (`penguin-classifier`).
* Configure the Python virtual environment (`.venv`) and install required libraries (`pandas`, `scikit-learn`).
* Ingest the raw CSV data from the provided source URL via Pandas.
* Output the initial `DataFrame` head and structural information (`df.info()`) to identify missing values and data types.
* **Checkpoint:** Initialize Git repository and commit the initial setup.

## Phase 2: Data Cleaning & Preprocessing
**Objective:** Prepare the dataset for algorithmic processing. Machine learning models require clean, numerical data; they cannot process null values or string characters.
* Isolate and remove or impute any rows containing missing values (`NaN`).
* Identify categorical string columns (e.g., "Island", "Sex") and biometric numerical columns.
* Filter the dataset to retain only the essential features required for the classification model (Target: Species. Features: Flipper length, bill depth, body mass, etc.).
* **Checkpoint:** Verify the cleaned `DataFrame` contains no nulls and commit changes.

## Phase 3: The Train-Test Split
**Objective:** Divide the data to prevent algorithmic memorization (overfitting).
* Separate the dataset into a feature matrix ($X$) and a target array ($y$).
* Import `train_test_split` from `sklearn.model_selection`.
* Partition $X$ and $y$ into training and testing subsets (standard distribution is typically 80% training, 20% testing).
* Output the dimensions (shape) of the resulting sets to verify the mathematical split.
* **Checkpoint:** Commit the data splitting logic.

## Phase 4: Model Initialization & Training
**Objective:** Instantiate the classification algorithm and fit it to the historical data.
* Import a classification algorithm from Scikit-learn (e.g., `DecisionTreeClassifier`).
* Instantiate the model object.
* Execute the `.fit()` method using the training matrices ($X\_train$, $y\_train$) to allow the algorithm to learn the mathematical boundaries between species.
* **Checkpoint:** Commit the model training execution.

## Phase 5: Evaluation & Prediction
**Objective:** Test the model's predictive accuracy on unseen data.
* Execute the `.predict()` method using the isolated test matrix ($X\_test$).
* Compare the model's predictions against the actual answers ($y\_test$) using Scikit-learn's `accuracy_score`.
* Construct a mock array of biometric measurements for a theoretical penguin and pass it to the model to output a final species classification.
* **Final Checkpoint:** Commit the evaluation logic and push to the remote repository.