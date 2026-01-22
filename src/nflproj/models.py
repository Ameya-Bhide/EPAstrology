"""Models for role and efficiency prediction."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging

from .config import SHRINKAGE_K

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_shrinkage(
    player_mean: float,
    position_mean: float,
    n_opportunities: int,
    k: int,
    position: str,
    metric_type: str = "targets"
) -> float:
    """
    Apply empirical Bayes shrinkage to per-opportunity metrics.
    
    shrunk = (n * player_mean + k * position_mean) / (n + k)
    
    Args:
        player_mean: Player's observed mean
        position_mean: Position's mean
        n_opportunities: Number of opportunities for player
        k: Shrinkage parameter
        position: Player position
        metric_type: "targets" or "carries" (to get appropriate k)
        
    Returns:
        Shrunk estimate
    """
    if n_opportunities == 0:
        return position_mean
    
    k_dict = SHRINKAGE_K.get(metric_type, SHRINKAGE_K["targets"])
    k_value = k_dict.get(position, k_dict.get("default", k))
    
    shrunk = (n_opportunities * player_mean + k_value * position_mean) / (n_opportunities + k_value)
    return shrunk


def compute_positional_averages(
    player_game: pd.DataFrame,
    players: pd.DataFrame,
    metric: str = "epa_per_target"
) -> Dict[str, float]:
    """
    Compute positional averages for a metric.
    
    Args:
        player_game: Player game stats
        players: Players table with positions
        metric: Metric to average (e.g., "epa_per_target", "epa_per_rush")
        
    Returns:
        Dictionary mapping position to average
    """
    merged = pd.merge(player_game, players[["player_id", "position"]], on="player_id", how="left")
    
    # Filter to relevant opportunities
    if "target" in metric:
        merged = merged[merged["targets"] > 0].copy()
        weight_col = "targets"
    elif "rush" in metric:
        merged = merged[merged["carries"] > 0].copy()
        weight_col = "carries"
    else:
        weight_col = None
    
    if weight_col is not None and weight_col in merged.columns:
        # Compute weighted average by position
        pos_avg = {}
        for position, group in merged.groupby("position"):
            if len(group) > 0 and weight_col in group.columns:
                weights = group[weight_col].values
                values = group[metric].values
                # Filter out any NaN values
                mask = ~(np.isnan(values) | np.isnan(weights))
                if mask.sum() > 0:
                    pos_avg[position] = np.average(values[mask], weights=weights[mask])
                else:
                    pos_avg[position] = 0.0
            else:
                pos_avg[position] = 0.0
    else:
        pos_avg = merged.groupby("position")[metric].mean().to_dict()
    
    return pos_avg


class BaselineRoleModel:
    """Baseline model for role prediction (targets, carries)."""
    
    def __init__(self):
        self.positional_avgs = {}
        self.team_volume_avgs = {}
    
    def fit(self, player_game: pd.DataFrame, team_game: pd.DataFrame, players: pd.DataFrame):
        """Fit baseline model (compute averages)."""
        # Compute positional averages for target share and carry share
        merged = pd.merge(player_game, players[["player_id", "position"]], on="player_id", how="left")
        
        # Target share by position
        player_with_team = pd.merge(
            merged,
            team_game[["game_id", "dropbacks"]],
            on="game_id",
            how="left"
        )
        player_with_team = player_with_team[player_with_team["dropbacks"] > 0]
        player_with_team["target_share"] = player_with_team["targets"] / player_with_team["dropbacks"]
        
        self.positional_avgs["target_share"] = player_with_team.groupby("position")["target_share"].mean().to_dict()
        
        # Carry share by position
        player_with_team = pd.merge(
            merged,
            team_game[["game_id", "rushes"]],
            on="game_id",
            how="left"
        )
        player_with_team = player_with_team[player_with_team["rushes"] > 0]
        player_with_team["carry_share"] = player_with_team["carries"] / player_with_team["rushes"]
        
        self.positional_avgs["carry_share"] = player_with_team.groupby("position")["carry_share"].mean().to_dict()
        
        # Team volume averages (dropbacks and rushes per game)
        if "dropbacks" not in team_game.columns:
            raise KeyError("'dropbacks' column not found in team_game table")
        if "rushes" not in team_game.columns:
            raise KeyError("'rushes' column not found in team_game table. Available columns: " + str(list(team_game.columns)))
        self.team_volume_avgs["dropbacks"] = team_game["dropbacks"].mean()
        self.team_volume_avgs["rushes"] = team_game["rushes"].mean()
    
    def predict_targets(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        team_game: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """
        Predict targets using baseline: weighted avg of last3 + (season share × projected team volume).
        """
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            # Get player position
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0:
                position = "WR"  # Default
            else:
                position = player_info.iloc[0]["position"]
            
            # Get player's last 3 games
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            # Weighted average of last 3 (more recent = higher weight)
            if len(prior_games) > 0:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["targets"].values * weights).sum()
            else:
                last3_avg = 0
            
            # Season-to-date target share
            season_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[(games["date"] < game_date) & (games["season"] == row["season"])]["game_id"]
                ))
            ]
            
            if len(season_games) > 0:
                # Get team dropbacks for these games
                season_with_team = pd.merge(
                    season_games,
                    team_game[["game_id", "dropbacks"]],
                    on="game_id",
                    how="left"
                )
                # Filter to games where player had targets AND team had dropbacks
                season_with_team = season_with_team[
                    (season_with_team["targets"] > 0) & 
                    (season_with_team["dropbacks"].notna()) &
                    (season_with_team["dropbacks"] > 0)
                ]
                
                if len(season_with_team) > 0 and season_with_team["dropbacks"].sum() > 0:
                    target_share_season = season_with_team["targets"].sum() / season_with_team["dropbacks"].sum()
                else:
                    target_share_season = self.positional_avgs["target_share"].get(position, 0.15)
            else:
                target_share_season = self.positional_avgs["target_share"].get(position, 0.15)
            
            # Projected team volume (use team's last 6 games average)
            team = row["team"]
            team_prior = team_game[
                (team_game["team"] == team) &
                (team_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].tail(6)
            
            if len(team_prior) > 0:
                proj_dropbacks = team_prior["dropbacks"].mean()
            else:
                proj_dropbacks = self.team_volume_avgs["dropbacks"]
            
            # Combine: 60% last3, 40% season share × volume
            pred = 0.6 * last3_avg + 0.4 * (target_share_season * proj_dropbacks)
            predictions.append(max(0, pred))
        
        return pd.Series(predictions, index=features.index)
    
    def predict_carries(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        team_game: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """Predict carries using same baseline approach."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0:
                position = "RB"
            else:
                position = player_info.iloc[0]["position"]
            
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            if len(prior_games) > 0:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["carries"].values * weights).sum()
            else:
                last3_avg = 0
            
            season_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[(games["date"] < game_date) & (games["season"] == row["season"])]["game_id"]
                ))
            ]
            
            if len(season_games) > 0:
                season_with_team = pd.merge(
                    season_games,
                    team_game[["game_id", "rushes"]],
                    on="game_id",
                    how="left"
                )
                # Filter to games where player had carries AND team had rushes
                season_with_team = season_with_team[
                    (season_with_team["carries"] > 0) & 
                    (season_with_team["rushes"].notna()) &
                    (season_with_team["rushes"] > 0)
                ]
                
                if len(season_with_team) > 0 and season_with_team["rushes"].sum() > 0:
                    carry_share_season = season_with_team["carries"].sum() / season_with_team["rushes"].sum()
                else:
                    carry_share_season = self.positional_avgs["carry_share"].get(position, 0.20)
            else:
                carry_share_season = self.positional_avgs["carry_share"].get(position, 0.20)
            
            team = row["team"]
            team_prior = team_game[
                (team_game["team"] == team) &
                (team_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].tail(6)
            
            if len(team_prior) > 0:
                proj_rushes = team_prior["rushes"].mean()
            else:
                proj_rushes = self.team_volume_avgs["rushes"]
            
            pred = 0.6 * last3_avg + 0.4 * (carry_share_season * proj_rushes)
            predictions.append(max(0, pred))
        
        return pd.Series(predictions, index=features.index)


class BaselineEfficiencyModel:
    """Baseline model for efficiency prediction (EPA per target/rush) with shrinkage."""
    
    def __init__(self):
        self.positional_avgs = {}
    
    def fit(self, player_game: pd.DataFrame, players: pd.DataFrame):
        """Fit efficiency model (compute positional averages)."""
        self.positional_avgs["epa_per_target"] = compute_positional_averages(
            player_game, players, "epa_per_target"
        )
        self.positional_avgs["epa_per_rush"] = compute_positional_averages(
            player_game, players, "epa_per_rush"
        )
    
    def predict_epa_per_target(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        players: pd.DataFrame
    ) -> pd.Series:
        """Predict EPA per target with shrinkage."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0:
                position = "WR"
            else:
                position = player_info.iloc[0]["position"]
            
            # Get player's last 6 games for targets
            n_targets = row.get("targets_last6", 0)
            player_mean = row.get("epa_per_target_last6", 0.0)
            
            if n_targets == 0 or player_mean == 0:
                # No history, use positional average
                pos_mean = self.positional_avgs["epa_per_target"].get(position, 0.5)
                predictions.append(pos_mean)
            else:
                # Apply shrinkage
                pos_mean = self.positional_avgs["epa_per_target"].get(position, 0.5)
                shrunk = apply_shrinkage(
                    player_mean, pos_mean, int(n_targets), 30, position, "targets"
                )
                predictions.append(shrunk)
        
        return pd.Series(predictions, index=features.index)
    
    def predict_epa_per_rush(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        players: pd.DataFrame
    ) -> pd.Series:
        """Predict EPA per rush with shrinkage."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0:
                position = "RB"
            else:
                position = player_info.iloc[0]["position"]
            
            n_carries = row.get("carries_last6", 0)
            player_mean = row.get("epa_per_rush_last6", 0.0)
            
            if n_carries == 0 or player_mean == 0:
                pos_mean = self.positional_avgs["epa_per_rush"].get(position, 0.3)
                predictions.append(pos_mean)
            else:
                pos_mean = self.positional_avgs["epa_per_rush"].get(position, 0.3)
                shrunk = apply_shrinkage(
                    player_mean, pos_mean, int(n_carries), 40, position, "carries"
                )
                predictions.append(shrunk)
        
        return pd.Series(predictions, index=features.index)


class MLRoleModel:
    """ML model for role prediction."""
    
    def __init__(self, model_type: str = "ridge", position_specific: bool = False):
        self.model_type = model_type
        self.position_specific = position_specific
        self.targets_model = None
        self.carries_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        
        # Position-specific models
        if position_specific:
            self.position_models = {
                "WR": {"targets": None, "carries": None, "scaler": StandardScaler()},
                "RB": {"targets": None, "carries": None, "scaler": StandardScaler()},
                "TE": {"targets": None, "carries": None, "scaler": StandardScaler()},
            }
    
    def _get_feature_cols(self, features: pd.DataFrame) -> list:
        """Get feature columns for modeling."""
        cols = [
            # Rolling role stats
            "targets_last1", "targets_last3", "targets_last6",
            "carries_last1", "carries_last3", "carries_last6",
            # Shares (important for role prediction)
            "target_share_last3", "target_share_last6",
            "carry_share_last3", "carry_share_last6",
            # Trends
            "target_share_slope_3", "carry_share_slope_3",
            # Context
            "team_pass_rate_last6", "team_plays_last6",
            # Opponent
            "opp_def_epa_allowed_last6",
            # Efficiency features (can help predict role - efficient players get more opportunities)
            "epa_per_target_last6", "epa_per_rush_last6",
            "success_rate_last6",
        ]
        # Only include columns that exist
        return [c for c in cols if c in features.columns]
    
    def fit(self, features: pd.DataFrame, player_game: pd.DataFrame, players: Optional[pd.DataFrame] = None):
        """Fit ML models for targets and carries."""
        # Merge with players to get position if not already there
        if players is not None and "position" not in features.columns:
            features = pd.merge(
                features,
                players[["player_id", "position"]],
                on="player_id",
                how="left"
            )
        
        self.feature_cols = self._get_feature_cols(features)
        
        # Merge with actuals
        merged = pd.merge(
            features,
            player_game[["game_id", "player_id", "targets", "carries"]],
            on=["game_id", "player_id"],
            how="inner"
        )
        
        if self.position_specific and "position" in merged.columns:
            # Fit separate models for each position
            for position in ["WR", "RB", "TE"]:
                pos_data = merged[merged["position"] == position].copy()
                if len(pos_data) < 10:  # Need minimum data
                    logger.warning(f"Not enough data for {position} model ({len(pos_data)} samples), skipping")
                    continue
                
                X_pos = pos_data[self.feature_cols].fillna(0)
                y_targets_pos = pos_data["targets"]
                y_carries_pos = pos_data["carries"]
                
                # Scale features
                X_pos_scaled = self.position_models[position]["scaler"].fit_transform(X_pos)
                
                # Fit models
                if self.model_type == "ridge":
                    self.position_models[position]["targets"] = Ridge(alpha=0.5)
                    self.position_models[position]["carries"] = Ridge(alpha=0.5)
                elif self.model_type == "poisson":
                    self.position_models[position]["targets"] = PoissonRegressor(alpha=0.5)
                    self.position_models[position]["carries"] = PoissonRegressor(alpha=0.5)
                elif self.model_type == "gbm":
                    self.position_models[position]["targets"] = GradientBoostingRegressor(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, min_samples_split=20
                    )
                    self.position_models[position]["carries"] = GradientBoostingRegressor(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, min_samples_split=20
                    )
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                self.position_models[position]["targets"].fit(X_pos_scaled, y_targets_pos)
                self.position_models[position]["carries"].fit(X_pos_scaled, y_carries_pos)
                
                logger.info(f"Fitted {position} model with {len(pos_data)} samples")
        
        # Also fit global model (for positions not in position_specific or as fallback)
        X = merged[self.feature_cols].fillna(0)
        y_targets = merged["targets"]
        y_carries = merged["carries"]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit models with improved default hyperparameters
        if self.model_type == "ridge":
            # Lower alpha for less regularization
            self.targets_model = Ridge(alpha=0.5)
            self.carries_model = Ridge(alpha=0.5)
        elif self.model_type == "poisson":
            # Poisson is better for count data
            self.targets_model = PoissonRegressor(alpha=0.5)
            self.carries_model = PoissonRegressor(alpha=0.5)
        elif self.model_type == "gbm":
            # Better defaults for GBM
            self.targets_model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=20
            )
            self.carries_model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=20
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.targets_model.fit(X_scaled, y_targets)
        self.carries_model.fit(X_scaled, y_carries)
    
    def predict_targets(self, features: pd.DataFrame, players: Optional[pd.DataFrame] = None) -> pd.Series:
        """Predict targets."""
        if self.targets_model is None:
            raise ValueError("Model not fitted")
        
        # Get position if available
        if self.position_specific and players is not None:
            if "position" not in features.columns:
                features = pd.merge(
                    features,
                    players[["player_id", "position"]],
                    on="player_id",
                    how="left"
                )
        
        X = features[self.feature_cols].fillna(0)
        preds = []
        
        if self.position_specific and "position" in features.columns:
            # Use position-specific models where available
            for idx, row in features.iterrows():
                position = row.get("position", None)
                if position in self.position_models and self.position_models[position]["targets"] is not None:
                    # Use position-specific model
                    X_pos = X.iloc[[idx]]
                    X_pos_scaled = self.position_models[position]["scaler"].transform(X_pos)
                    pred = self.position_models[position]["targets"].predict(X_pos_scaled)[0]
                    preds.append(pred)
                else:
                    # Fall back to global model
                    X_scaled = self.scaler.transform(X.iloc[[idx]])
                    pred = self.targets_model.predict(X_scaled)[0]
                    preds.append(pred)
            preds = np.array(preds)
        else:
            # Use global model
            X_scaled = self.scaler.transform(X)
            preds = self.targets_model.predict(X_scaled)
        
        return pd.Series(np.maximum(0, preds), index=features.index)
    
    def predict_carries(self, features: pd.DataFrame, players: Optional[pd.DataFrame] = None) -> pd.Series:
        """Predict carries."""
        if self.carries_model is None:
            raise ValueError("Model not fitted")
        
        # Get position if available
        if self.position_specific and players is not None:
            if "position" not in features.columns:
                features = pd.merge(
                    features,
                    players[["player_id", "position"]],
                    on="player_id",
                    how="left"
                )
        
        X = features[self.feature_cols].fillna(0)
        preds = []
        
        if self.position_specific and "position" in features.columns:
            # Use position-specific models where available
            for idx, row in features.iterrows():
                position = row.get("position", None)
                if position in self.position_models and self.position_models[position]["carries"] is not None:
                    # Use position-specific model
                    X_pos = X.iloc[[idx]]
                    X_pos_scaled = self.position_models[position]["scaler"].transform(X_pos)
                    pred = self.position_models[position]["carries"].predict(X_pos_scaled)[0]
                    preds.append(pred)
                else:
                    # Fall back to global model
                    X_scaled = self.scaler.transform(X.iloc[[idx]])
                    pred = self.carries_model.predict(X_scaled)[0]
                    preds.append(pred)
            preds = np.array(preds)
        else:
            # Use global model
            X_scaled = self.scaler.transform(X)
            preds = self.carries_model.predict(X_scaled)
        
        return pd.Series(np.maximum(0, preds), index=features.index)


def project_player(
    player_id: str,
    game_id: str,
    role_model,
    efficiency_model,
    features: pd.DataFrame,
    player_game: pd.DataFrame,
    team_game: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame
) -> Dict:
    """
    Project a player's EPA contribution for a game.
    
    Returns:
        Dictionary with projections
    """
    # Get features for this player-game
    player_features = features[
        (features["player_id"] == player_id) &
        (features["game_id"] == game_id)
    ]
    
    if len(player_features) == 0:
        raise ValueError(f"No features found for player {player_id} in game {game_id}")
    
    player_features = player_features.iloc[0]
    
    # Predict role
    if isinstance(role_model, BaselineRoleModel):
        proj_targets = role_model.predict_targets(
            pd.DataFrame([player_features]),
            player_game, team_game, players, games
        ).iloc[0]
        proj_carries = role_model.predict_carries(
            pd.DataFrame([player_features]),
            player_game, team_game, players, games
        ).iloc[0]
    else:
        proj_targets = role_model.predict_targets(pd.DataFrame([player_features])).iloc[0]
        proj_carries = role_model.predict_carries(pd.DataFrame([player_features])).iloc[0]
    
    # Predict efficiency
    if isinstance(efficiency_model, BaselineEfficiencyModel):
        proj_epa_per_target = efficiency_model.predict_epa_per_target(
            pd.DataFrame([player_features]),
            player_game, players
        ).iloc[0]
        proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
            pd.DataFrame([player_features]),
            player_game, players
        ).iloc[0]
    else:
        # For ML efficiency models, would need to implement
        proj_epa_per_target = efficiency_model.predict_epa_per_target(
            pd.DataFrame([player_features]),
            player_game, players
        ).iloc[0]
        proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
            pd.DataFrame([player_features]),
            player_game, players
        ).iloc[0]
    
    # Combine
    proj_epa_total = proj_targets * proj_epa_per_target + proj_carries * proj_epa_per_rush
    
    return {
        "player_id": player_id,
        "game_id": game_id,
        "proj_targets": proj_targets,
        "proj_carries": proj_carries,
        "proj_epa_per_target": proj_epa_per_target,
        "proj_epa_per_rush": proj_epa_per_rush,
        "proj_epa_total": proj_epa_total
    }
