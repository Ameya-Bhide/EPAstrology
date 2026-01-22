"""Improved ML models with hyperparameter tuning and better features."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedMLRoleModel:
    """Improved ML model for role prediction with hyperparameter tuning."""
    
    def __init__(self, model_type: str = "ridge", tune_hyperparams: bool = True):
        self.model_type = model_type
        self.tune_hyperparams = tune_hyperparams
        self.targets_model = None
        self.carries_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.feature_importance_targets = None
        self.feature_importance_carries = None
    
    def _get_feature_cols(self, features: pd.DataFrame, players: pd.DataFrame) -> list:
        """Get feature columns for modeling, including position encoding."""
        cols = [
            # Rolling role stats
            "targets_last1", "targets_last3", "targets_last6",
            "carries_last1", "carries_last3", "carries_last6",
            # Shares
            "target_share_last3", "target_share_last6",
            "carry_share_last3", "carry_share_last6",
            # Trends
            "target_share_slope_3", "carry_share_slope_3",
            # Context
            "team_pass_rate_last6", "team_plays_last6",
            # Opponent
            "opp_def_epa_allowed_last6",
            # Efficiency (might help predict role)
            "epa_per_target_last6", "epa_per_rush_last6",
            "success_rate_last6",
        ]
        
        # Only include columns that exist
        available_cols = [c for c in cols if c in features.columns]
        
        # Add position encoding if available
        if "position" in features.columns or "position" in players.columns:
            available_cols.append("position_encoded")
        
        return available_cols
    
    def _encode_position(self, features: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        """Add position encoding to features."""
        features = features.copy()
        
        # Merge position if not already there
        if "position" not in features.columns:
            features = pd.merge(
                features,
                players[["player_id", "position"]],
                on="player_id",
                how="left"
            )
        
        # One-hot encode position (simpler than full encoding for now)
        position_dummies = pd.get_dummies(features["position"], prefix="pos", sparse=False)
        features = pd.concat([features, position_dummies], axis=1)
        
        # Also add a simple numeric encoding
        position_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "FB": 4}
        features["position_encoded"] = features["position"].map(position_map).fillna(5)
        
        return features
    
    def fit(self, features: pd.DataFrame, player_game: pd.DataFrame, players: pd.DataFrame):
        """Fit ML models with improved hyperparameters."""
        # Encode position
        features_enhanced = self._encode_position(features, players)
        
        self.feature_cols = self._get_feature_cols(features_enhanced, players)
        
        # Merge with actuals
        merged = pd.merge(
            features_enhanced,
            player_game[["game_id", "player_id", "targets", "carries"]],
            on=["game_id", "player_id"],
            how="inner"
        )
        
        # Sort by date for time series split (if date available)
        if "date" not in merged.columns:
            # Try to get date from games if available
            try:
                from .config import PARQUET_DIR
                games = pd.read_parquet(PARQUET_DIR / "games.parquet")
                merged = pd.merge(merged, games[["game_id", "date"]], on="game_id", how="left")
                merged = merged.sort_values("date")
            except:
                pass
        
        X = merged[self.feature_cols].fillna(0)
        y_targets = merged["targets"]
        y_carries = merged["carries"]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Hyperparameter grids
        if self.model_type == "ridge":
            param_grid = {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            }
            base_model = Ridge
        elif self.model_type == "poisson":
            param_grid = {
                "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
            }
            base_model = PoissonRegressor
        elif self.model_type == "gbm":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.05, 0.1]
            }
            base_model = GradientBoostingRegressor
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit targets model
        if self.tune_hyperparams and len(X_scaled) > 100:
            # Use time series split for cross-validation
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_scaled) // 20))
            grid_search = GridSearchCV(
                base_model(),
                param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_scaled, y_targets)
            self.targets_model = grid_search.best_estimator_
            logger.info(f"Best targets model params: {grid_search.best_params_}, score: {grid_search.best_score_:.4f}")
        else:
            # Use default or simple params
            if self.model_type == "ridge":
                self.targets_model = Ridge(alpha=0.5)  # Lower than before
            elif self.model_type == "poisson":
                self.targets_model = PoissonRegressor(alpha=0.5)
            elif self.model_type == "gbm":
                self.targets_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
            self.targets_model.fit(X_scaled, y_targets)
        
        # Fit carries model
        if self.tune_hyperparams and len(X_scaled) > 100:
            grid_search = GridSearchCV(
                base_model(),
                param_grid,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_scaled, y_carries)
            self.carries_model = grid_search.best_estimator_
            logger.info(f"Best carries model params: {grid_search.best_params_}, score: {grid_search.best_score_:.4f}")
        else:
            if self.model_type == "ridge":
                self.carries_model = Ridge(alpha=0.5)
            elif self.model_type == "poisson":
                self.carries_model = PoissonRegressor(alpha=0.5)
            elif self.model_type == "gbm":
                self.carries_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05)
            self.carries_model.fit(X_scaled, y_carries)
        
        # Store feature importance if available
        if hasattr(self.targets_model, "feature_importances_"):
            self.feature_importance_targets = dict(zip(self.feature_cols, self.targets_model.feature_importances_))
        elif hasattr(self.targets_model, "coef_"):
            # For linear models, use absolute coefficients
            self.feature_importance_targets = dict(zip(self.feature_cols, np.abs(self.targets_model.coef_)))
        
        if hasattr(self.carries_model, "feature_importances_"):
            self.feature_importance_carries = dict(zip(self.feature_cols, self.carries_model.feature_importances_))
        elif hasattr(self.carries_model, "coef_"):
            self.feature_importance_carries = dict(zip(self.feature_cols, np.abs(self.carries_model.coef_)))
    
    def predict_targets(self, features: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
        """Predict targets."""
        if self.targets_model is None:
            raise ValueError("Model not fitted")
        
        features_enhanced = self._encode_position(features, players)
        X = features_enhanced[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        preds = self.targets_model.predict(X_scaled)
        return pd.Series(np.maximum(0, preds), index=features.index)
    
    def predict_carries(self, features: pd.DataFrame, players: pd.DataFrame) -> pd.Series:
        """Predict carries."""
        if self.carries_model is None:
            raise ValueError("Model not fitted")
        
        features_enhanced = self._encode_position(features, players)
        X = features_enhanced[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        preds = self.carries_model.predict(X_scaled)
        return pd.Series(np.maximum(0, preds), index=features.index)
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance for both models."""
        return {
            "targets": self.feature_importance_targets,
            "carries": self.feature_importance_carries
        }
