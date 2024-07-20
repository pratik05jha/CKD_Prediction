import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define preprocessing function
def preprocess_input(features):
    # Assuming 'features' is a numpy array of shape (1, -1)
    df = pd.DataFrame(features, columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

    # Apply necessary preprocessing steps (e.g., scaling, encoding)
    # This should match the preprocessing applied during training
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Add any other preprocessing steps as required
    # ...

    return df.values
