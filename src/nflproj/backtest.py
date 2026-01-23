"""Walk-forward backtesting framework."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .models import BaselineRoleModel, BaselineEfficiencyModel, MLRoleModel
from .config import PARQUET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def walk_forward_backtest(
    season: int,
    player_game: pd.DataFrame,
    team_game: pd.DataFrame,
    features: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    model_type: str = "baseline"
) -> Dict:
    """
    Walk-forward backtest: train on past weeks, predict next week.
    
    Args:
        season: Season to backtest
        player_game: Player game stats
        team_game: Team game stats
        features: Features table
        games: Games table
        players: Players table
        model_type: "baseline" or "ml"
        
    Returns:
        Dictionary with predictions and metrics
    """
    logger.info(f"Starting walk-forward backtest for season {season}")
    
    # Get games for this season, sorted by week
    season_games = games[games["season"] == season].sort_values(["week", "date"])
    weeks = sorted(season_games["week"].unique())
    
    all_predictions = []
    all_actuals = []
    
    # Walk forward by week
    for week in weeks:
        logger.info(f"Backtesting week {week}")
        
        # Get games for this week
        week_games = season_games[season_games["week"] == week]
        week_game_ids = week_games["game_id"].tolist()
        
        # Get the date of the first game in this week (for temporal filtering)
        week_date = week_games["date"].min()
        
        # Get training data: all games before this week's date (from any season)
        # This allows training on multiple seasons of historical data
        train_games = games[games["date"] < week_date]
        train_game_ids = train_games["game_id"].tolist()
        
        if len(train_game_ids) == 0:
            logger.warning(f"No training data for week {week}, skipping")
            continue
        
        logger.info(f"Training on {len(train_game_ids)} games from seasons {sorted(train_games['season'].unique())}")
        
        # Get features for this week (to predict)
        week_features = features[
            (features["game_id"].isin(week_game_ids)) &
            (features["season"] == season)
        ]
        
        if len(week_features) == 0:
            logger.warning(f"No features for week {week}, skipping")
            continue
        
        # Train models on training data (from all previous seasons)
        train_features = features[features["game_id"].isin(train_game_ids)]
        train_player_game = player_game[player_game["game_id"].isin(train_game_ids)]
        train_team_game = team_game[team_game["game_id"].isin(train_game_ids)]
        
        # Initialize and fit models
        if model_type == "baseline":
            role_model = BaselineRoleModel()
            role_model.fit(train_player_game, train_team_game, players)
            
            efficiency_model = BaselineEfficiencyModel()
            efficiency_model.fit(train_player_game, players)
        else:
            # Use position-specific models if model_type includes it
            position_specific = model_type.endswith("_pos") or model_type == "gbm_pos" or model_type == "ridge_pos"
            model_type_clean = model_type.replace("_pos", "")
            
            # Tune hyperparameters based on model type and training data size
            ml_kwargs = {}
            if model_type_clean == "ridge":
                ml_kwargs = {"alpha": 0.5}
            elif model_type_clean == "gbm":
                # Balanced tuning for target prediction with more data
                # Reduced from 200 to 150 to speed up training
                ml_kwargs = {
                    "n_estimators": 150,  # More trees but not too many
                    "max_depth": 4,  # Moderate depth
                    "learning_rate": 0.05,  # Standard learning rate
                    "subsample": 0.8,
                    "min_samples_split": 20,  # Standard splits
                    "min_samples_leaf": 5
                }
            elif model_type_clean == "xgb":
                # XGBoost hyperparameters optimized for larger dataset
                ml_kwargs = {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42
                }
            elif model_type_clean == "nn":
                # Neural network hyperparameters
                ml_kwargs = {
                    "hidden_layer_sizes": (128, 64, 32),
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
            
            role_model = MLRoleModel(model_type=model_type_clean, position_specific=position_specific)
            # Store kwargs for model initialization
            role_model.kwargs = ml_kwargs
            # Pass players, team_game, and games for team volume features
            try:
                role_model.fit(train_features, train_player_game, players, train_team_game, games)
            except Exception as e:
                logger.warning(f"Error fitting ML model: {e}, falling back to baseline")
                role_model = BaselineRoleModel()
                role_model.fit(train_player_game, train_team_game, players)
            
            efficiency_model = BaselineEfficiencyModel()  # Still use baseline for efficiency
            efficiency_model.fit(train_player_game, players)
        
        # Make predictions for this week
        for _, row in week_features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            # Get player position
            player_info = players[players["player_id"] == player_id]
            if len(player_info) == 0:
                position = None
            else:
                position = player_info.iloc[0]["position"]
            
            # Skip QBs for target/carry predictions (they use pass attempts)
            is_qb = position == "QB"
            
            try:
                # Predict role
                proj_pass_attempts = 0.0  # Initialize for non-QBs
                if isinstance(role_model, BaselineRoleModel):
                    if is_qb:
                        # QBs use pass attempts, not targets/carries
                        proj_targets = 0.0
                        proj_pass_attempts = role_model.predict_pass_attempts(
                            pd.DataFrame([row]),
                            train_player_game, train_team_game, players, games
                        ).iloc[0]
                        proj_carries = role_model.predict_carries(
                            pd.DataFrame([row]),
                            train_player_game, train_team_game, players, games
                        ).iloc[0]
                    else:
                        proj_targets = role_model.predict_targets(
                            pd.DataFrame([row]),
                            train_player_game, train_team_game, players, games
                        ).iloc[0]
                        proj_carries = role_model.predict_carries(
                            pd.DataFrame([row]),
                            train_player_game, train_team_game, players, games
                        ).iloc[0]
                else:
                    # ML model - check if QB and use pass attempts model
                    if is_qb and hasattr(role_model, 'pass_attempts_model') and role_model.pass_attempts_model is not None:
                        # QB: predict pass attempts instead of targets
                        proj_pass_attempts = role_model.predict_pass_attempts(
                            pd.DataFrame([row]), players, train_team_game, games
                        ).iloc[0]
                        proj_targets = 0.0
                        proj_carries = role_model.predict_carries(
                            pd.DataFrame([row]), players, train_team_game, games
                        ).iloc[0]
                    else:
                        # Skill position: predict targets and carries
                        proj_targets = role_model.predict_targets(
                            pd.DataFrame([row]), players, train_team_game, games
                        ).iloc[0]
                        proj_carries = role_model.predict_carries(
                            pd.DataFrame([row]), players, train_team_game, games
                        ).iloc[0]
                
                # Predict efficiency
                if is_qb:
                    # QB efficiency metrics
                    proj_epa_per_attempt = efficiency_model.predict_epa_per_attempt(
                        pd.DataFrame([row]),
                        train_player_game, players, games
                    ).iloc[0]
                    proj_epa_per_target = 0.0
                    proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
                        pd.DataFrame([row]),
                        train_player_game, players
                    ).iloc[0]
                else:
                    proj_epa_per_target = efficiency_model.predict_epa_per_target(
                        pd.DataFrame([row]),
                        train_player_game, players
                    ).iloc[0]
                    proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
                        pd.DataFrame([row]),
                        train_player_game, players
                    ).iloc[0]
                    proj_epa_per_attempt = 0.0
                
                # Get actuals
                actual = player_game[
                    (player_game["player_id"] == player_id) &
                    (player_game["game_id"] == game_id)
                ]
                
                if len(actual) > 0:
                    actual = actual.iloc[0]
                    if is_qb:
                        actual_targets = 0.0
                        actual_carries = actual.get("carries", 0)
                        actual_pass_attempts = actual.get("pass_attempts", 0)
                        actual_epa_per_target = 0.0
                        actual_epa_per_rush = actual["epa_per_rush"] if actual["carries"] > 0 else 0.0
                        actual_epa_per_attempt = actual.get("epa_per_attempt", 0.0) if actual_pass_attempts > 0 else 0.0
                        actual_epa_total = actual["epa_total"]
                        
                        if isinstance(role_model, BaselineRoleModel):
                            proj_epa_total = proj_carries * proj_epa_per_rush + proj_pass_attempts * proj_epa_per_attempt
                        else:
                            proj_epa_total = proj_carries * proj_epa_per_rush + proj_pass_attempts * proj_epa_per_attempt
                    else:
                        actual_targets = actual["targets"]
                        actual_carries = actual["carries"]
                        actual_pass_attempts = 0.0
                        actual_epa_per_target = actual["epa_per_target"] if actual["targets"] > 0 else 0.0
                        actual_epa_per_rush = actual["epa_per_rush"] if actual["carries"] > 0 else 0.0
                        actual_epa_per_attempt = 0.0
                        actual_epa_total = actual["epa_total"]
                        
                        proj_epa_total = proj_targets * proj_epa_per_target + proj_carries * proj_epa_per_rush
                    
                    all_predictions.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "week": week,
                        "proj_targets": proj_targets,
                        "proj_carries": proj_carries,
                        "proj_pass_attempts": proj_pass_attempts if is_qb else 0.0,
                        "proj_epa_per_target": proj_epa_per_target,
                        "proj_epa_per_rush": proj_epa_per_rush,
                        "proj_epa_per_attempt": proj_epa_per_attempt,
                        "proj_epa_total": proj_epa_total
                    })
                    
                    all_actuals.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "week": week,
                        "actual_targets": actual_targets,
                        "actual_carries": actual_carries,
                        "actual_pass_attempts": actual_pass_attempts,
                        "actual_epa_per_target": actual_epa_per_target,
                        "actual_epa_per_rush": actual_epa_per_rush,
                        "actual_epa_per_attempt": actual_epa_per_attempt,
                        "actual_epa_total": actual_epa_total
                    })
            except Exception as e:
                logger.warning(f"Error predicting for player {player_id} in game {game_id}: {e}")
                continue
    
    # Compute metrics
    pred_df = pd.DataFrame(all_predictions)
    actual_df = pd.DataFrame(all_actuals)
    
    if len(pred_df) == 0:
        logger.warning("No predictions generated")
        return {"error": "No predictions generated"}
    
    # Merge predictions and actuals
    results = pd.merge(
        pred_df,
        actual_df,
        on=["game_id", "player_id", "week"],
        how="inner"
    )
    
    # Compute metrics
    metrics = {
        "role_mae_targets": np.mean(np.abs(results["proj_targets"] - results["actual_targets"])),
        "role_mae_carries": np.mean(np.abs(results["proj_carries"] - results["actual_carries"])),
        "efficiency_mae_epa_per_target": np.mean(np.abs(
            results[results["actual_targets"] > 0]["proj_epa_per_target"] - 
            results[results["actual_targets"] > 0]["actual_epa_per_target"]
        )) if len(results[results["actual_targets"] > 0]) > 0 else np.nan,
        "efficiency_mae_epa_per_rush": np.mean(np.abs(
            results[results["actual_carries"] > 0]["proj_epa_per_rush"] - 
            results[results["actual_carries"] > 0]["actual_epa_per_rush"]
        )) if len(results[results["actual_carries"] > 0]) > 0 else np.nan,
        "total_mae_epa": np.mean(np.abs(results["proj_epa_total"] - results["actual_epa_total"])),
        "n_predictions": len(results)
    }
    
    # Rank correlation for efficiency (only for players with opportunities)
    targets_mask = results["actual_targets"] > 0
    if targets_mask.sum() > 1:
        metrics["epa_per_target_rank_corr"] = results[targets_mask]["proj_epa_per_target"].corr(
            results[targets_mask]["actual_epa_per_target"], method="spearman"
        )
    else:
        metrics["epa_per_target_rank_corr"] = np.nan
    
    carries_mask = results["actual_carries"] > 0
    if carries_mask.sum() > 1:
        metrics["epa_per_rush_rank_corr"] = results[carries_mask]["proj_epa_per_rush"].corr(
            results[carries_mask]["actual_epa_per_rush"], method="spearman"
        )
    else:
        metrics["epa_per_rush_rank_corr"] = np.nan
    
    logger.info(f"Backtest complete. Metrics: {metrics}")
    
    return {
        "metrics": metrics,
        "predictions": results.to_dict("records")
    }


def compare_baseline_vs_ml(
    season: int,
    player_game: pd.DataFrame,
    team_game: pd.DataFrame,
    features: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    ml_model: str = "gbm_pos"
) -> Dict:
    """
    Compare baseline vs ML models in backtest.
    
    Args:
        ml_model: ML model type to compare (default: gbm_pos for position-specific)
    
    Returns:
        Dictionary with metrics for both models
    """
    logger.info(f"Comparing baseline vs {ml_model} for season {season}")
    
    baseline_results = walk_forward_backtest(
        season, player_game, team_game, features, games, players, model_type="baseline"
    )
    
    ml_results = walk_forward_backtest(
        season, player_game, team_game, features, games, players, model_type=ml_model
    )
    
    return {
        "baseline": baseline_results.get("metrics", {}),
        "ml": ml_results.get("metrics", {}),
        "ml_model": ml_model,
        "improvement": {
            k: ml_results.get("metrics", {}).get(k, 0) - baseline_results.get("metrics", {}).get(k, 0)
            for k in baseline_results.get("metrics", {}).keys()
            if k != "n_predictions"
        }
    }
