import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from credit_model.config.config import config
from credit_model.processing.data_manager import load_data
from credit_model.processing.features import AgeTransformer, EmploymentTransformer
from credit_model.pipeline import credit_pipeline

def run_training():
    # Load data
    df = load_data()
    
    # Apply custom transformers
    df = AgeTransformer().fit_transform(df)
    df = EmploymentTransformer().fit_transform(df)
    
    # Drop unused columns
    df = df.drop(columns=config.features_to_drop)
    
    # Split
    X = df.drop(columns=[config.target])
    y = df[config.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, 
        random_state=config.random_state, stratify=y
    )
    
    # Train
    credit_pipeline.fit(X_train, y_train)
    
    # Save
    save_path = Path(__file__).parent / config.pipeline_save_file
    joblib.dump(credit_pipeline, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    run_training()