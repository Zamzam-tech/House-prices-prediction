import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_and_preprocess_data(df):
    """Performs data cleaning, preprocessing, and feature engineering."""
    # Drop irrelevant columns
    columns_to_drop = ['Plot Area', 'Dimensions', 'Super Area', 'Car Parking', 'Society', 'Ownership', 'Balcony', 'overlooking', 'facing', 'Carpet Area']
    df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Convert 'Bathroom' to numerical and handle missing values
    df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')
    df['Bathroom'].fillna(df['Bathroom'].median(), inplace=True)
    
    # Handle missing values in numerical and categorical columns
    numerical_cols_to_impute = ['Price (in rupees)']
    categorical_cols_to_impute = ['Description', 'Status', 'Floor', 'Transaction', 'Furnishing']
    
    for column in numerical_cols_to_impute:
        df[column].fillna(df[column].median(), inplace=True)
    
    for column in categorical_cols_to_impute:
        df[column].fillna(df[column].mode()[0], inplace=True)

    # Convert 'Amount(in rupees)'
    def convert_amount(amount):
        if isinstance(amount, float):
            return amount
        elif 'Lac' in str(amount):
            return float(str(amount).replace('Lac', '').strip()) * 100000
        elif 'Cr' in str(amount):
            return float(str(amount).replace('Cr', '').strip()) * 10000000
        else:
            try:
                return float(amount)
            except (ValueError, TypeError):
                return None
    
    df['Amount(in rupees)'] = df['Amount(in rupees)'].apply(convert_amount)
    df['Amount(in rupees)'].fillna(df['Amount(in rupees)'].median(), inplace=True)
    
    # Drop highly correlated columns
    df.drop(['Amount(in rupees)', 'Title', 'Description'], axis=1, inplace=True)
    
    # One-hot encode categorical columns
    categorical_cols_to_one_hot_code = ['Status', 'Transaction', 'Furnishing']
    df = pd.get_dummies(df, columns=categorical_cols_to_one_hot_code, drop_first=True)
    
    # Handle high cardinality in 'location'
    location_counts = df['location'].value_counts()
    location_threshold = 100
    rare_locations = location_counts[location_counts < location_threshold].index
    df['location'] = df['location'].apply(lambda x: 'Other' if x in rare_locations else x)
    df = pd.get_dummies(df, columns=['location'], drop_first=True)
    
    # Clean 'Floor' column
    def clean_floor(floor):
        floor = str(floor).lower()
        if 'ground' in floor:
            return 0
        try:
            return int(floor.split()[0])
        except (ValueError, IndexError):
            return None
    
    df['Floor'] = df['Floor'].apply(clean_floor)
    df['Floor'].fillna(df['Floor'].median(), inplace=True)
    df['Floor'] = df['Floor'].astype(int)
    
    # Remove outliers based on price
    lower_bound = df['Price (in rupees)'].quantile(0.01)
    upper_bound = df['Price (in rupees)'].quantile(0.99)
    df_filtered = df[(df['Price (in rupees)'] > lower_bound) & (df['Price (in rupees)'] < upper_bound)].copy()
    
    return df_filtered

def train_and_evaluate_models(X_train, Y_train, X_test, Y_test):
    """Trains and evaluates Linear Regression and Random Forest models."""
    # Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)
    y_predicted_lr = linear_model.predict(X_test)
    mse_lr = mean_squared_error(Y_test, y_predicted_lr)
    r2_lr = r2_score(Y_test, y_predicted_lr)
    
    print(f'Linear Regression Metrics:')
    print(f'Mean Squared Error: {mse_lr}')
    print(f'R-squared value: {r2_lr}')
    
    # Random Forest Regressor Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, Y_train)
    y_predicted_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(Y_test, y_predicted_rf)
    r2_rf = r2_score(Y_test, y_predicted_rf)
    
    print(f'\nRandom Forest Regressor Metrics:')
    print(f'Mean Squared Error: {mse_rf}')
    print(f'R-squared value: {r2_rf}')

if __name__ == "__main__":
    file_path = 'data\house_prices.csv'
    df = load_data(file_path)
    df_processed = clean_and_preprocess_data(df)

    X = df_processed.drop('Price (in rupees)', axis=1)
    Y = df_processed['Price (in rupees)']
    
    # Drop 'Index' column if it exists and is not needed for modeling
    if 'Index' in X.columns:
        X.drop('Index', axis=1, inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    train_and_evaluate_models(X_train, Y_train, X_test, Y_test)