"""
Threshold Prediction Module using Random Forest Regressor.

Predicts optimal Canny edge detection thresholds based on image quality features.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from dataclasses import dataclass


@dataclass
class ThresholdPrediction:
    """Container for threshold predictions."""
    low_threshold: float
    high_threshold: float
    confidence: Optional[float] = None
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return thresholds as tuple."""
        return (self.low_threshold, self.high_threshold)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        """Return thresholds as integer tuple for cv2.Canny."""
        return (int(round(self.low_threshold)), int(round(self.high_threshold)))


class ThresholdPredictor:
    """
    Predicts optimal Canny thresholds using Random Forest regression.
    
    The model learns the mapping from image quality features to optimal
    (low_threshold, high_threshold) pairs for Canny edge detection.
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the threshold predictor.
        
        Args:
            model_path: Path to load a pre-trained model (optional)
        """
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.feature_names = ['brightness', 'contrast', 'noise_level', 
                              'entropy', 'blur_level', 'edge_density']
        
        if model_path is not None:
            self.load(model_path)
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              tune_hyperparams: bool = False,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the threshold prediction model.
        
        Args:
            X: Feature matrix of shape (n_samples, 6)
            y: Target thresholds of shape (n_samples, 2) - [low, high]
            tune_hyperparams: Whether to perform hyperparameter tuning
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training metrics
        """
        if verbose:
            print(f"Training on {len(X)} samples...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if tune_hyperparams:
            if verbose:
                print("Performing hyperparameter tuning...")
            self.model, best_params = self._tune_hyperparams(X_scaled, y, verbose)
        else:
            # Default parameters
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            best_params = None
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, 
                                    scoring='neg_mean_absolute_error')
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'cv_mae': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'n_samples': len(X),
            'best_params': best_params,
            'feature_importances': dict(zip(self.feature_names, 
                                           self.model.feature_importances_))
        }
        
        if verbose:
            print(f"Training complete!")
            print(f"  MAE: {mae:.2f}")
            print(f"  CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
            print(f"  Feature importances:")
            for name, imp in sorted(metrics['feature_importances'].items(), 
                                   key=lambda x: -x[1]):
                print(f"    {name}: {imp:.3f}")
        
        return metrics
    
    def _tune_hyperparams(self, X: np.ndarray, y: np.ndarray, 
                          verbose: bool) -> Tuple[RandomForestRegressor, Dict]:
        """
        Tune hyperparameters using grid search.
        
        Args:
            X: Scaled feature matrix
            y: Target thresholds
            verbose: Print progress
            
        Returns:
            Tuple of (best model, best parameters)
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5,
            scoring='neg_mean_absolute_error',
            verbose=2 if verbose else 0,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.2f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def predict(self, features: np.ndarray) -> ThresholdPrediction:
        """
        Predict thresholds for a single image's features.
        
        Args:
            features: 1D array of 6 features
            
        Returns:
            ThresholdPrediction object
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")
        
        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Ensure valid threshold range and relationship
        low = np.clip(prediction[0], 10, 200)
        high = np.clip(prediction[1], 50, 300)
        
        # Ensure high > low
        if high <= low:
            high = low + 50
        
        return ThresholdPrediction(
            low_threshold=low,
            high_threshold=high
        )
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Predict thresholds for multiple images.
        
        Args:
            features: 2D array of shape (n_samples, 6)
            
        Returns:
            2D array of shape (n_samples, 2) with [low, high] thresholds
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        # Clip to valid ranges
        predictions[:, 0] = np.clip(predictions[:, 0], 10, 200)  # low
        predictions[:, 1] = np.clip(predictions[:, 1], 50, 300)  # high
        
        # Ensure high > low
        mask = predictions[:, 1] <= predictions[:, 0]
        predictions[mask, 1] = predictions[mask, 0] + 50
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True
        
        print(f"Model loaded from {path}")
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
