import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from surprise import Dataset, Reader, SVD
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def revenue_analysis(df):
    """
    Perform revenue analysis using MapReduce-like operations
    """
    # Map phase: Calculate revenue per product
    df['revenue'] = df['actual_price'] * df['no_of_ratings']
    
    # Reduce phase: Aggregate by main_category and sub_category
    revenue_by_category = df.groupby(['main_category', 'sub_category'])['revenue'].sum().reset_index()
    revenue_trend = df.groupby(['main_category'])['revenue'].sum().sort_values(ascending=False)
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    revenue_trend.plot(kind='bar')
    plt.title('Revenue by Main Category')
    plt.xlabel('Category')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig('revenue_analysis.png')
    
    return revenue_by_category

def inventory_forecast(df):
    """
    Implement inventory demand forecasting using Random Forest and XGBoost
    """
    # Feature engineering
    le = LabelEncoder()
    df['main_category_encoded'] = le.fit_transform(df['main_category'])
    df['sub_category_encoded'] = le.fit_transform(df['sub_category'])
    
    # Prepare features and target
    features = ['main_category_encoded', 'sub_category_encoded', 'ratings', 'discount_price', 'actual_price']
    X = df[features]
    y = df['no_of_ratings']  # Using number of ratings as a proxy for demand
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    
    # Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_score = xgb_model.score(X_test, y_test)
    
    return {'random_forest_score': rf_score, 'xgboost_score': xgb_score}

def product_recommendations(df):
    """
    Implement a simple product recommendation system
    """
    # Create a reader object
    reader = Reader(rating_scale=(1, 5))
    
    # Prepare data for Surprise
    ratings_data = pd.DataFrame({
        'user_id': range(len(df)),  # Using index as user_id for demonstration
        'item_id': df.index,
        'rating': df['ratings']
    })
    
    # Load data into Surprise format
    data = Dataset.load_from_df(ratings_data, reader)
    
    # Train SVD model
    model = SVD(n_factors=100, random_state=42)
    trainset = data.build_full_trainset()
    model.fit(trainset)
    
    return model

def main():
    # Load data
    df = load_data('your_data.csv')  # Replace with your data file path
    
    # Perform revenue analysis
    print("Performing revenue analysis...")
    revenue_results = revenue_analysis(df)
    print("\nTop revenue generating categories:")
    print(revenue_results.head())
    
    # Perform inventory forecasting
    print("\nTraining inventory forecasting models...")
    forecast_results = inventory_forecast(df)
    print("\nModel Performance:")
    print(f"Random Forest R² Score: {forecast_results['random_forest_score']:.4f}")
    print(f"XGBoost R² Score: {forecast_results['xgboost_score']:.4f}")
    
    # Build recommendation system
    print("\nBuilding recommendation system...")
    recommender = product_recommendations(df)
    print("Recommendation system ready!")

if __name__ == "__main__":
    main()