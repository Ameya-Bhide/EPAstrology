"""Models for role and efficiency prediction."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import logging
import joblib
from pathlib import Path

# Try to import XGBoost, but don't fail if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost not available. Install with: pip install xgboost")

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
    elif "attempt" in metric:
        merged = merged[merged["pass_attempts"] > 0].copy() if "pass_attempts" in merged.columns else pd.DataFrame()
        weight_col = "pass_attempts"
    elif "completion" in metric:
        merged = merged[merged["pass_attempts"] > 0].copy() if "pass_attempts" in merged.columns else pd.DataFrame()
        weight_col = "pass_attempts"
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
        
        # QB pass attempt share (for QBs, their share of team dropbacks)
        if "pass_attempts" in merged.columns:
            qb_with_team = pd.merge(
                merged[merged["position"] == "QB"],
                team_game[["game_id", "dropbacks"]],
                on="game_id",
                how="left"
            )
            qb_with_team = qb_with_team[qb_with_team["dropbacks"] > 0]
            if len(qb_with_team) > 0:
                qb_with_team["pass_attempt_share"] = qb_with_team["pass_attempts"] / qb_with_team["dropbacks"]
                self.positional_avgs["pass_attempt_share"] = {"QB": qb_with_team["pass_attempt_share"].mean()}
        
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
    
    def predict_pass_attempts(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        team_game: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """
        Predict pass attempts for QBs using baseline: weighted avg of last3 + (season share × projected team volume).
        """
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            # Get player position - only predict for QBs
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0 or player_info.iloc[0]["position"] != "QB":
                predictions.append(0.0)
                continue
            
            # Get player's last 3 games
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            # Weighted average of last 3 (more recent = higher weight)
            if len(prior_games) > 0 and "pass_attempts" in prior_games.columns:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["pass_attempts"].values * weights).sum()
            else:
                last3_avg = 0
            
            # Season-to-date pass attempt share
            season_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[(games["date"] < game_date) & (games["season"] == row["season"])]["game_id"]
                ))
            ]
            
            if len(season_games) > 0 and "pass_attempts" in season_games.columns:
                # Get team dropbacks for these games
                season_with_team = pd.merge(
                    season_games,
                    team_game[["game_id", "dropbacks"]],
                    on="game_id",
                    how="left"
                )
                season_with_team = season_with_team[season_with_team["dropbacks"] > 0]
                
                if len(season_with_team) > 0:
                    season_pass_attempts = season_with_team["pass_attempts"].sum()
                    season_dropbacks = season_with_team["dropbacks"].sum()
                    season_share = season_pass_attempts / season_dropbacks if season_dropbacks > 0 else 0
                else:
                    season_share = self.positional_avgs.get("pass_attempt_share", {}).get("QB", 0.95)  # Default QB share
            else:
                season_share = self.positional_avgs.get("pass_attempt_share", {}).get("QB", 0.95)
            
            # Projected team dropbacks (average of last 3 games)
            team_prior = team_game[
                (team_game["team"] == row["team"]) &
                (team_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            if len(team_prior) > 0:
                proj_team_dropbacks = team_prior["dropbacks"].mean()
            else:
                proj_team_dropbacks = self.team_volume_avgs.get("dropbacks", 35.0)
            
            # Combine: 60% last3 avg, 40% (season share × projected team volume)
            pred = 0.6 * last3_avg + 0.4 * (season_share * proj_team_dropbacks)
            predictions.append(max(0.0, pred))
        
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
        
        # Compute positional averages for QB metrics
        if "epa_per_attempt" in player_game.columns:
            self.positional_avgs["epa_per_attempt"] = compute_positional_averages(
                player_game, players, "epa_per_attempt"
            )
        if "completion_rate" in player_game.columns:
            self.positional_avgs["completion_rate"] = compute_positional_averages(
                player_game, players, "completion_rate"
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
    
    def predict_epa_per_attempt(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """
        Predict EPA per attempt for QBs using shrinkage.
        """
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            # Get player position - only predict for QBs
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0 or player_info.iloc[0]["position"] != "QB":
                predictions.append(0.0)
                continue
            
            position = "QB"
            
            # Get player's last 6 games
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(6)
            
            # Compute player's observed EPA per attempt
            if len(prior_games) > 0 and "pass_attempts" in prior_games.columns:
                total_attempts = prior_games["pass_attempts"].sum()
                if total_attempts > 0:
                    player_mean = prior_games["epa_passing"].sum() / total_attempts
                else:
                    player_mean = 0.0
            else:
                player_mean = 0.0
            
            # Get positional average
            position_mean = self.positional_avgs.get("epa_per_attempt", {}).get(position, 0.0)
            
            # Apply shrinkage
            n_opportunities = int(total_attempts) if len(prior_games) > 0 and "pass_attempts" in prior_games.columns else 0
            shrunk = apply_shrinkage(
                player_mean, position_mean, n_opportunities, 50, position, "pass_attempts"
            )
            
            predictions.append(shrunk)
        
        return pd.Series(predictions, index=features.index)
    
    def predict_completion_rate(
        self,
        features: pd.DataFrame,
        player_game: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """
        Predict completion rate for QBs using shrinkage.
        """
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            # Get player position - only predict for QBs
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0 or player_info.iloc[0]["position"] != "QB":
                predictions.append(0.0)
                continue
            
            position = "QB"
            
            # Get player's last 6 games
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game[
                (player_game["player_id"] == player_id) &
                (player_game["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(6)
            
            # Compute player's observed completion rate
            if len(prior_games) > 0 and "pass_attempts" in prior_games.columns:
                total_attempts = prior_games["pass_attempts"].sum()
                if total_attempts > 0:
                    player_mean = prior_games["completions"].sum() / total_attempts
                else:
                    player_mean = 0.0
            else:
                player_mean = 0.0
            
            # Get positional average
            position_mean = self.positional_avgs.get("completion_rate", {}).get(position, 0.65)
            
            # Apply shrinkage
            n_opportunities = int(total_attempts) if len(prior_games) > 0 and "pass_attempts" in prior_games.columns else 0
            shrunk = apply_shrinkage(
                player_mean, position_mean, n_opportunities, 50, position, "pass_attempts"
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
        self.pass_attempts_model = None  # QB model
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.kwargs = {}  # Store kwargs for model initialization
        
        # Position-specific models
        if position_specific:
            self.position_models = {
                "WR": {"targets": None, "carries": None, "scaler": StandardScaler()},
                "RB": {"targets": None, "carries": None, "scaler": StandardScaler()},
                "TE": {"targets": None, "carries": None, "scaler": StandardScaler()},
                "QB": {"pass_attempts": None, "scaler": StandardScaler()},
            }
    
    def _get_feature_cols(self, features: pd.DataFrame, for_qb: bool = False) -> list:
        """Get feature columns for modeling."""
        if for_qb:
            # QB-specific features
            cols = [
                # Rolling role stats
                "pass_attempts_last1", "pass_attempts_last3", "pass_attempts_last6",
                # Shares (QB's share of team dropbacks)
                "pass_attempt_share_last3", "pass_attempt_share_last6",
                # Trends
                "pass_attempt_share_slope_3",
                # Team context
                "team_pass_rate_last6", "team_plays_last6",
                # Opponent defense
                "opp_def_pass_epa_allowed_last6",
                # Efficiency features
                "completion_rate_last6", "epa_per_attempt_last6",
            ]
        else:
            # Skill position features
            cols = [
                # Rolling role stats (raw counts)
                "targets_last1", "targets_last3", "targets_last6",
                "carries_last1", "carries_last3", "carries_last6",
                # Shares (CRITICAL for target prediction - player's share of team targets)
                "target_share_last3", "target_share_last6",
                "carry_share_last3", "carry_share_last6",
                # Trends (is share increasing/decreasing?)
                "target_share_slope_3", "carry_share_slope_3",
                # Team context (CRITICAL - projected team volume)
                "team_pass_rate_last6", "team_plays_last6",
                # Opponent defense
                "opp_def_epa_allowed_last6",
                # Efficiency features (efficient players may get more targets)
                "epa_per_target_last6", "epa_per_rush_last6",
                "success_rate_last6",
            ]
        # Only include columns that exist
        return [c for c in cols if c in features.columns]
    
    def fit(self, features: pd.DataFrame, player_game: pd.DataFrame, players: Optional[pd.DataFrame] = None,
            team_game: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None):
        """Fit ML models for targets and carries."""
        # Merge with players to get position if not already there
        if players is not None and "position" not in features.columns:
            features = pd.merge(
                features,
                players[["player_id", "position"]],
                on="player_id",
                how="left"
            )
        
        # Add team volume features (critical for target prediction)
        if team_game is not None and games is not None:
            # Get projected team dropbacks for each game
            features = pd.merge(features, games[["game_id", "date"]], on="game_id", how="left")
            
            # For each feature row, get team's projected dropbacks (last 6 games before this game)
            proj_dropbacks = []
            proj_rushes = []
            
            for idx, row in features.iterrows():
                team = row["team"]
                game_date = row["date"]
                
                # Get team's last 6 games before this game
                team_prior = team_game[
                    (team_game["team"] == team) &
                    (team_game["game_id"].isin(
                        games[games["date"] < game_date]["game_id"]
                    ))
                ].tail(6)
                
                if len(team_prior) > 0:
                    proj_dropbacks.append(team_prior["dropbacks"].mean())
                    proj_rushes.append(team_prior["rushes"].mean())
                else:
                    # Fallback to overall average
                    proj_dropbacks.append(team_game["dropbacks"].mean() if len(team_game) > 0 else 35.0)
                    proj_rushes.append(team_game["rushes"].mean() if len(team_game) > 0 else 25.0)
            
            features["proj_team_dropbacks"] = proj_dropbacks
            features["proj_team_rushes"] = proj_rushes
            
            # Create interaction features: share × projected volume (like baseline does)
            if "target_share_last6" in features.columns:
                features["target_share_x_proj_dropbacks"] = (
                    features["target_share_last6"].fillna(0) * features["proj_team_dropbacks"]
                )
            if "carry_share_last6" in features.columns:
                features["carry_share_x_proj_rushes"] = (
                    features["carry_share_last6"].fillna(0) * features["proj_team_rushes"]
                )
        
        self.feature_cols = self._get_feature_cols(features)
        
        # Add team volume features to feature list
        if "proj_team_dropbacks" in features.columns:
            self.feature_cols.append("proj_team_dropbacks")
        if "proj_team_rushes" in features.columns:
            self.feature_cols.append("proj_team_rushes")
        if "target_share_x_proj_dropbacks" in features.columns:
            self.feature_cols.append("target_share_x_proj_dropbacks")
        if "carry_share_x_proj_rushes" in features.columns:
            self.feature_cols.append("carry_share_x_proj_rushes")
        
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
                    # Use kwargs if provided, otherwise use defaults
                    gbm_kwargs = self.kwargs.copy() if self.kwargs else {}
                    default_gbm = {
                        "n_estimators": 100,
                        "max_depth": 3,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "min_samples_split": 20
                    }
                    default_gbm.update(gbm_kwargs)
                    
                    self.position_models[position]["targets"] = GradientBoostingRegressor(**default_gbm)
                    self.position_models[position]["carries"] = GradientBoostingRegressor(**default_gbm)
                elif self.model_type == "xgb" and XGBOOST_AVAILABLE:
                    # XGBoost with tuned hyperparameters
                    xgb_kwargs = self.kwargs.copy() if self.kwargs else {}
                    default_xgb = {
                        "n_estimators": 150,
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "min_child_weight": 3,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                        "random_state": 42
                    }
                    default_xgb.update(xgb_kwargs)
                    
                    self.position_models[position]["targets"] = xgb.XGBRegressor(**default_xgb)
                    self.position_models[position]["carries"] = xgb.XGBRegressor(**default_xgb)
                elif self.model_type == "nn":
                    # Neural network with tuned hyperparameters
                    nn_kwargs = self.kwargs.copy() if self.kwargs else {}
                    default_nn = {
                        "hidden_layer_sizes": (100, 50),
                        "activation": "relu",
                        "solver": "adam",
                        "alpha": 0.01,
                        "learning_rate": "adaptive",
                        "learning_rate_init": 0.001,
                        "max_iter": 500,
                        "early_stopping": True,
                        "validation_fraction": 0.1,
                        "random_state": 42
                    }
                    default_nn.update(nn_kwargs)
                    
                    self.position_models[position]["targets"] = MLPRegressor(**default_nn)
                    self.position_models[position]["carries"] = MLPRegressor(**default_nn)
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
            # Use kwargs if provided, otherwise use defaults
            gbm_kwargs = self.kwargs.copy() if self.kwargs else {}
            default_gbm = {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_split": 20
            }
            default_gbm.update(gbm_kwargs)
            
            self.targets_model = GradientBoostingRegressor(**default_gbm)
            self.carries_model = GradientBoostingRegressor(**default_gbm)
        elif self.model_type == "xgb" and XGBOOST_AVAILABLE:
            # XGBoost with tuned hyperparameters
            xgb_kwargs = self.kwargs.copy() if self.kwargs else {}
            default_xgb = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42
            }
            default_xgb.update(xgb_kwargs)
            
            self.targets_model = xgb.XGBRegressor(**default_xgb)
            self.carries_model = xgb.XGBRegressor(**default_xgb)
        elif self.model_type == "nn":
            # Neural network with tuned hyperparameters
            nn_kwargs = self.kwargs.copy() if self.kwargs else {}
            default_nn = {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.01,
                "learning_rate": "adaptive",
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42
            }
            default_nn.update(nn_kwargs)
            
            self.targets_model = MLPRegressor(**default_nn)
            self.carries_model = MLPRegressor(**default_nn)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.targets_model.fit(X_scaled, y_targets)
        self.carries_model.fit(X_scaled, y_carries)
        
        # Fit QB pass attempts model if we have QB data
        if "pass_attempts" in merged.columns:
            qb_data = merged[merged["position"] == "QB"] if "position" in merged.columns else pd.DataFrame()
            if len(qb_data) > 0:
                qb_feature_cols = self._get_feature_cols(features, for_qb=True)
                if len(qb_feature_cols) > 0:
                    # Add team volume features for QBs
                    if team_game is not None and games is not None:
                        if "proj_team_dropbacks" not in qb_data.columns:
                            qb_data = pd.merge(qb_data, games[["game_id", "date"]], on="game_id", how="left")
                            proj_dropbacks = []
                            for idx, row in qb_data.iterrows():
                                team = row["team"]
                                game_date = row["date"]
                                team_prior = team_game[
                                    (team_game["team"] == team) &
                                    (team_game["game_id"].isin(
                                        games[games["date"] < game_date]["game_id"]
                                    ))
                                ].tail(6)
                                proj_dropbacks.append(team_prior["dropbacks"].mean() if len(team_prior) > 0 else 35.0)
                            qb_data["proj_team_dropbacks"] = proj_dropbacks
                            if "pass_attempt_share_last6" in qb_data.columns:
                                qb_data["pass_attempt_share_x_proj_dropbacks"] = (
                                    qb_data["pass_attempt_share_last6"].fillna(0) * qb_data["proj_team_dropbacks"]
                                )
                    
                    X_qb = qb_data[qb_feature_cols].fillna(0)
                    for col in qb_feature_cols:
                        if col not in X_qb.columns:
                            X_qb[col] = 0.0
                    
                    if "proj_team_dropbacks" in qb_data.columns:
                        qb_feature_cols.append("proj_team_dropbacks")
                        X_qb["proj_team_dropbacks"] = qb_data["proj_team_dropbacks"]
                    if "pass_attempt_share_x_proj_dropbacks" in qb_data.columns:
                        qb_feature_cols.append("pass_attempt_share_x_proj_dropbacks")
                        X_qb["pass_attempt_share_x_proj_dropbacks"] = qb_data["pass_attempt_share_x_proj_dropbacks"]
                    
                    X_qb = X_qb[qb_feature_cols].fillna(0)
                    y_pass_attempts = qb_data["pass_attempts"]
                    
                    if len(X_qb) > 0:
                        qb_scaler = StandardScaler()
                        X_qb_scaled = qb_scaler.fit_transform(X_qb)
                        
                        if self.model_type == "ridge":
                            self.pass_attempts_model = Ridge(alpha=0.5)
                        elif self.model_type == "poisson":
                            self.pass_attempts_model = PoissonRegressor(alpha=0.5)
                        elif self.model_type == "gbm":
                            gbm_kwargs = self.kwargs.copy() if self.kwargs else {}
                            default_gbm = {
                                "n_estimators": 100,
                                "max_depth": 3,
                                "learning_rate": 0.05,
                                "subsample": 0.8,
                                "min_samples_split": 20
                            }
                            default_gbm.update(gbm_kwargs)
                            self.pass_attempts_model = GradientBoostingRegressor(**default_gbm)
                        elif self.model_type == "xgb" and XGBOOST_AVAILABLE:
                            xgb_kwargs = self.kwargs.copy() if self.kwargs else {}
                            default_xgb = {
                                "n_estimators": 150,
                                "max_depth": 4,
                                "learning_rate": 0.05,
                                "subsample": 0.8,
                                "colsample_bytree": 0.8,
                                "min_child_weight": 3,
                                "reg_alpha": 0.1,
                                "reg_lambda": 1.0,
                                "random_state": 42
                            }
                            default_xgb.update(xgb_kwargs)
                            self.pass_attempts_model = xgb.XGBRegressor(**default_xgb)
                        elif self.model_type == "nn":
                            nn_kwargs = self.kwargs.copy() if self.kwargs else {}
                            default_nn = {
                                "hidden_layer_sizes": (100, 50),
                                "activation": "relu",
                                "solver": "adam",
                                "alpha": 0.01,
                                "learning_rate": "adaptive",
                                "learning_rate_init": 0.001,
                                "max_iter": 500,
                                "early_stopping": True,
                                "validation_fraction": 0.1,
                                "random_state": 42
                            }
                            default_nn.update(nn_kwargs)
                            self.pass_attempts_model = MLPRegressor(**default_nn)
                        else:
                            raise ValueError(f"Unknown model type: {self.model_type}")
                        
                        self.pass_attempts_model.fit(X_qb_scaled, y_pass_attempts)
                        self.qb_feature_cols = qb_feature_cols
                        self.qb_scaler = qb_scaler
                        logger.info(f"Fitted QB pass attempts model with {len(X_qb)} samples")
    
    def predict_targets(self, features: pd.DataFrame, players: Optional[pd.DataFrame] = None,
                        team_game: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None) -> pd.Series:
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
        
        # Add team volume features if not already present and we have the data
        if team_game is not None and games is not None:
            if "proj_team_dropbacks" not in features.columns:
                features = pd.merge(features, games[["game_id", "date"]], on="game_id", how="left")
                
                proj_dropbacks = []
                for idx, row in features.iterrows():
                    team = row["team"]
                    game_date = row["date"]
                    
                    team_prior = team_game[
                        (team_game["team"] == team) &
                        (team_game["game_id"].isin(
                            games[games["date"] < game_date]["game_id"]
                        ))
                    ].tail(6)
                    
                    if len(team_prior) > 0:
                        proj_dropbacks.append(team_prior["dropbacks"].mean())
                    else:
                        proj_dropbacks.append(team_game["dropbacks"].mean() if len(team_game) > 0 else 35.0)
                
                features["proj_team_dropbacks"] = proj_dropbacks
                
                # Add interaction features
                if "target_share_last6" in features.columns:
                    features["target_share_x_proj_dropbacks"] = (
                        features["target_share_last6"].fillna(0) * features["proj_team_dropbacks"]
                    )
        
        # Ensure all feature columns exist - add missing ones with zeros
        X = features.copy()
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        
        # Select only the feature columns we need, in the right order
        # Only select columns that actually exist (in case feature_cols has extras)
        available_cols = [c for c in self.feature_cols if c in X.columns]
        X = X[available_cols].fillna(0)
        
        # If we're missing columns, add zeros for them (model was trained with them)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        
        # Now select in the right order
        X = X[self.feature_cols].fillna(0)
        
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
    
    def predict_carries(self, features: pd.DataFrame, players: Optional[pd.DataFrame] = None,
                       team_game: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None) -> pd.Series:
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
        
        # Ensure all feature columns exist - add missing ones with zeros
        X = features.copy()
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        
        # Select only the feature columns we need, in the right order
        # Only select columns that actually exist (in case feature_cols has extras)
        available_cols = [c for c in self.feature_cols if c in X.columns]
        X = X[available_cols].fillna(0)
        
        # If we're missing columns, add zeros for them (model was trained with them)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        
        # Now select in the right order
        X = X[self.feature_cols].fillna(0)
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
    
    def predict_pass_attempts(self, features: pd.DataFrame, players: Optional[pd.DataFrame] = None,
                              team_game: Optional[pd.DataFrame] = None, games: Optional[pd.DataFrame] = None) -> pd.Series:
        """Predict pass attempts for QBs."""
        # If no QB model, return zeros
        if not hasattr(self, 'pass_attempts_model') or self.pass_attempts_model is None:
            return pd.Series([0.0] * len(features), index=features.index)
        
        # Get position if available
        if players is not None:
            if "position" not in features.columns:
                features = pd.merge(
                    features,
                    players[["player_id", "position"]],
                    on="player_id",
                    how="left"
                )
        
        # Filter to QBs only
        qb_features = features[features["position"] == "QB"].copy() if "position" in features.columns else pd.DataFrame()
        if len(qb_features) == 0:
            return pd.Series([0.0] * len(features), index=features.index)
        
        # Add team volume features if not already present
        if team_game is not None and games is not None:
            if "proj_team_dropbacks" not in qb_features.columns:
                qb_features = pd.merge(qb_features, games[["game_id", "date"]], on="game_id", how="left")
                proj_dropbacks = []
                for idx, row in qb_features.iterrows():
                    team = row["team"]
                    game_date = row["date"]
                    team_prior = team_game[
                        (team_game["team"] == team) &
                        (team_game["game_id"].isin(
                            games[games["date"] < game_date]["game_id"]
                        ))
                    ].tail(6)
                    proj_dropbacks.append(team_prior["dropbacks"].mean() if len(team_prior) > 0 else 35.0)
                qb_features["proj_team_dropbacks"] = proj_dropbacks
                if "pass_attempt_share_last6" in qb_features.columns:
                    qb_features["pass_attempt_share_x_proj_dropbacks"] = (
                        qb_features["pass_attempt_share_last6"].fillna(0) * qb_features["proj_team_dropbacks"]
                    )
        
        # Ensure all QB feature columns exist
        X_qb = qb_features.copy()
        qb_feature_cols = getattr(self, 'qb_feature_cols', self._get_feature_cols(qb_features, for_qb=True))
        for col in qb_feature_cols:
            if col not in X_qb.columns:
                X_qb[col] = 0.0
        
        # Add team volume features if they exist
        if "proj_team_dropbacks" in qb_features.columns and "proj_team_dropbacks" not in qb_feature_cols:
            qb_feature_cols.append("proj_team_dropbacks")
        if "pass_attempt_share_x_proj_dropbacks" in qb_features.columns and "pass_attempt_share_x_proj_dropbacks" not in qb_feature_cols:
            qb_feature_cols.append("pass_attempt_share_x_proj_dropbacks")
        
        X_qb = X_qb[qb_feature_cols].fillna(0)
        
        # Scale and predict
        qb_scaler = getattr(self, 'qb_scaler', StandardScaler())
        X_qb_scaled = qb_scaler.transform(X_qb)
        qb_preds = self.pass_attempts_model.predict(X_qb_scaled)
        
        # Create full predictions series (zeros for non-QBs)
        all_preds = pd.Series([0.0] * len(features), index=features.index)
        all_preds.loc[qb_features.index] = np.maximum(0, qb_preds)
        
        return all_preds
    
    def save(self, model_path: Path):
        """Save the model to disk."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model attributes
        model_data = {
            "model_type": self.model_type,
            "position_specific": self.position_specific,
            "targets_model": self.targets_model,
            "carries_model": self.carries_model,
            "pass_attempts_model": getattr(self, 'pass_attempts_model', None),
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "kwargs": self.kwargs,
            "position_models": getattr(self, 'position_models', {}),
            "qb_feature_cols": getattr(self, 'qb_feature_cols', None),
            "qb_scaler": getattr(self, 'qb_scaler', None),
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Saved model to {model_path}")
    
    @classmethod
    def load(cls, model_path: Path):
        """Load a model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        # Create model instance
        model = cls(
            model_type=model_data["model_type"],
            position_specific=model_data["position_specific"]
        )
        
        # Restore attributes
        model.targets_model = model_data["targets_model"]
        model.carries_model = model_data["carries_model"]
        model.pass_attempts_model = model_data.get("pass_attempts_model")
        model.scaler = model_data["scaler"]
        model.feature_cols = model_data["feature_cols"]
        model.kwargs = model_data.get("kwargs", {})
        model.position_models = model_data.get("position_models", {})
        model.qb_feature_cols = model_data.get("qb_feature_cols")
        model.qb_scaler = model_data.get("qb_scaler")
        
        logger.info(f"Loaded model from {model_path}")
        return model


class BaselineDefenseModel:
    """Baseline model for defense prediction (tackles, sacks, interceptions)."""
    
    def __init__(self):
        self.positional_avgs = {}
        self.team_volume_avgs = {}
    
    def fit(self, player_game_defense: pd.DataFrame, team_game_defense: pd.DataFrame, players: pd.DataFrame):
        """Fit baseline defense model."""
        merged = pd.merge(player_game_defense, players[["player_id", "position"]], on="player_id", how="left")
        
        # Tackles per game by position
        self.positional_avgs["tackles_per_game"] = merged.groupby("position")["tackles"].mean().to_dict()
        
        # Sacks per game by position
        self.positional_avgs["sacks_per_game"] = merged.groupby("position")["sacks"].mean().to_dict()
        
        # Interceptions per game by position
        self.positional_avgs["interceptions_per_game"] = merged.groupby("position")["interceptions"].mean().to_dict()
        
        # Team volume averages
        if "plays_faced" in team_game_defense.columns:
            self.team_volume_avgs["plays_faced"] = team_game_defense["plays_faced"].mean()
        else:
            self.team_volume_avgs["plays_faced"] = 65.0  # Default
    
    def predict_tackles(
        self,
        features: pd.DataFrame,
        player_game_defense: pd.DataFrame,
        team_game_defense: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """Predict tackles using baseline: weighted avg of last 3 games."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            player_info = players[players["player_id"] == player_id]
            position = player_info.iloc[0]["position"] if len(player_info) > 0 else "LB"
            
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game_defense[
                (player_game_defense["player_id"] == player_id) &
                (player_game_defense["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            if len(prior_games) > 0:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["tackles"].values * weights).sum()
            else:
                last3_avg = self.positional_avgs.get("tackles_per_game", {}).get(position, 5.0)
            
            predictions.append(max(0.0, last3_avg))
        
        return pd.Series(predictions, index=features.index)
    
    def predict_sacks(
        self,
        features: pd.DataFrame,
        player_game_defense: pd.DataFrame,
        team_game_defense: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """Predict sacks using baseline: weighted avg of last 3 games."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            player_info = players[players["player_id"] == player_id]
            position = player_info.iloc[0]["position"] if len(player_info) > 0 else "DE"
            
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game_defense[
                (player_game_defense["player_id"] == player_id) &
                (player_game_defense["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            if len(prior_games) > 0:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["sacks"].values * weights).sum()
            else:
                last3_avg = self.positional_avgs.get("sacks_per_game", {}).get(position, 0.5)
            
            predictions.append(max(0.0, last3_avg))
        
        return pd.Series(predictions, index=features.index)
    
    def predict_interceptions(
        self,
        features: pd.DataFrame,
        player_game_defense: pd.DataFrame,
        team_game_defense: pd.DataFrame,
        players: pd.DataFrame,
        games: pd.DataFrame
    ) -> pd.Series:
        """Predict interceptions using baseline: weighted avg of last 3 games."""
        predictions = []
        
        for idx, row in features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            player_info = players[players["player_id"] == player_id]
            position = player_info.iloc[0]["position"] if len(player_info) > 0 else "DB"
            
            game_date = games[games["game_id"] == game_id]["date"].iloc[0]
            prior_games = player_game_defense[
                (player_game_defense["player_id"] == player_id) &
                (player_game_defense["game_id"].isin(
                    games[games["date"] < game_date]["game_id"]
                ))
            ].sort_values("game_id").tail(3)
            
            if len(prior_games) > 0:
                weights = np.array([0.5, 0.3, 0.2][-len(prior_games):])
                weights = weights / weights.sum()
                last3_avg = (prior_games["interceptions"].values * weights).sum()
            else:
                last3_avg = self.positional_avgs.get("interceptions_per_game", {}).get(position, 0.2)
            
            predictions.append(max(0.0, last3_avg))
        
        return pd.Series(predictions, index=features.index)


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
