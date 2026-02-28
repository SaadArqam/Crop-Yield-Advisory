Crop-Yield-Advisory/
│
├── .git/                      # Git version control metadata
├── .gitignore                 # Lists files to be ignored by Git (like the virtual environment or large data files)
├── README.md                  # The main documentation file explaining the project's purpose and how to run it
├── requirements.txt           # Lists all the Python packages needed to run the app (e.g., streamlit, pandas, scikit-learn)
├── app.py                     # The main application script that runs the Streamlit web user interface
│
├── data/                      # Contains all dataset files used for building the model
│   ├── pesticides.csv         # Raw data describing pesticide usage
│   ├── rainfall.csv           # Raw data containing rainfall statistics
│   ├── temp.csv               # Raw data containing temperature readings
│   ├── yield.csv              # Raw data indicating the crop yields
│   ├── yield_df.csv           # The main preprocessed and combined dataset used for model training
│   └── processed_data.csv     # Another processed version of the dataset
│
├── ml/                        # Put all your Machine Learning source code scripts here
│   ├── inspect_data.py        # A small script used for exploratory data analysis (EDA) to understand the dataset
│   ├── preprocess.py          # Contains functions that clean, merge, and transform raw CSVs into training-ready data
│   ├── train_model.py         # Trains a simple Linear model (likely Linear Regression or Ridge/Lasso)
│   ├── train_tree_model.py    # Trains a tree-based model (like a Decision Tree or Random Forest)
│   └── train_pipeline.py      # An automated script that combines preprocessing and training into one cohesive pipeline
│
└── models/                    # Stores the exported ML artifacts for the application to consume
    ├── crop_yield_model.pkl                 # The saved/trained Linear Regression model
    ├── crop_yield_tree.pkl                  # The saved/trained Tree-based model
    ├── label_encoders.pkl                   # Saved encoders to convert categorical data into numbers for predictions
    ├── label_encoders_pipeline.pkl          # Pipeline-version of the encoders
    ├── feature_order.json                   # Keeps track of the exact features and columns passed to the model
    ├── medians.json                         # Saved median values for numeric columns to handle missing input values
    ├── metrics_crop_yield_model.json        # Evaluation metrics for the linear model (e.g., MSE, R2 score)
    ├── metrics_crop_yield_tree.json         # Evaluation metrics for the tree model
    ├── linear_feature_importances.csv       # Tells you which features impacted the linear predictions the most
    └── tree_feature_importances.csv         # Tells you which features impacted the tree predictions the most
