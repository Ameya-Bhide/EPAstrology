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
        
        # Get training data: all games before this week
        train_games = season_games[season_games["week"] < week]
        train_game_ids = train_games["game_id"].tolist()
        
        if len(train_game_ids) == 0:
            logger.warning(f"No training data for week {week}, skipping")
            continue
        
        # Get features for this week (to predict)
        week_features = features[
            (features["game_id"].isin(week_game_ids)) &
            (features["season"] == season)
        ]
        
        if len(week_features) == 0:
            logger.warning(f"No features for week {week}, skipping")
            continue
        
        # Train models on training data
        train_features = features[
            (features["game_id"].isin(train_game_ids)) &
            (features["season"] == season)
        ]
        train_player_game = player_game[player_game["game_id"].isin(train_game_ids)]
        train_team_game = team_game[team_game["game_id"].isin(train_game_ids)]
        
        # Initialize and fit models
        if model_type == "baseline":
            role_model = BaselineRoleModel()
            role_model.fit(train_player_game, train_team_game, players)
            
            efficiency_model = BaselineEfficiencyModel()
            efficiency_model.fit(train_player_game, players)
        else:
            role_model = MLRoleModel(model_type=model_type)
            role_model.fit(train_features, train_player_game)
            
            efficiency_model = BaselineEfficiencyModel()  # Still use baseline for efficiency
            efficiency_model.fit(train_player_game, players)
        
        # Make predictions for this week
        for _, row in week_features.iterrows():
            player_id = row["player_id"]
            game_id = row["game_id"]
            
            try:
                # Predict role
                if isinstance(role_model, BaselineRoleModel):
                    proj_targets = role_model.predict_targets(
                        pd.DataFrame([row]),
                        train_player_game, train_team_game, players, games
                    ).iloc[0]
                    proj_carries = role_model.predict_carries(
                        pd.DataFrame([row]),
                        train_player_game, train_team_game, players, games
                    ).iloc[0]
                else:
                    proj_targets = role_model.predict_targets(pd.DataFrame([row])).iloc[0]
                    proj_carries = role_model.predict_carries(pd.DataFrame([row])).iloc[0]
                
                # Predict efficiency
                proj_epa_per_target = efficiency_model.predict_epa_per_target(
                    pd.DataFrame([row]),
                    train_player_game, players
                ).iloc[0]
                proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
                    pd.DataFrame([row]),
                    train_player_game, players
                ).iloc[0]
                
                # Get actuals
                actual = player_game[
                    (player_game["player_id"] == player_id) &
                    (player_game["game_id"] == game_id)
                ]
                
                if len(actual) > 0:
                    actual = actual.iloc[0]
                    actual_targets = actual["targets"]
                    actual_carries = actual["carries"]
                    actual_epa_per_target = actual["epa_per_target"] if actual["targets"] > 0 else 0.0
                    actual_epa_per_rush = actual["epa_per_rush"] if actual["carries"] > 0 else 0.0
                    actual_epa_total = actual["epa_total"]
                    
                    proj_epa_total = proj_targets * proj_epa_per_target + proj_carries * proj_epa_per_rush
                    
                    all_predictions.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "week": week,
                        "proj_targets": proj_targets,
                        "proj_carries": proj_carries,
                        "proj_epa_per_target": proj_epa_per_target,
                        "proj_epa_per_rush": proj_epa_per_rush,
                        "proj_epa_total": proj_epa_total
                    })
                    
                    all_actuals.append({
                        "game_id": game_id,
                        "player_id": player_id,
                        "week": week,
                        "actual_targets": actual_targets,
                        "actual_carries": actual_carries,
                        "actual_epa_per_target": actual_epa_per_target,
                        "actual_epa_per_rush": actual_epa_per_rush,
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
    players: pd.DataFrame
) -> Dict:
    """
    Compare baseline vs ML models in backtest.
    
    Returns:
        Dictionary with metrics for both models
    """
    logger.info(f"Comparing baseline vs ML for season {season}")
    
    baseline_results = walk_forward_backtest(
        season, player_game, team_game, features, games, players, model_type="baseline"
    )
    
    ml_results = walk_forward_backtest(
        season, player_game, team_game, features, games, players, model_type="ridge"
    )
    
    return {
        "baseline": baseline_results.get("metrics", {}),
        "ml": ml_results.get("metrics", {}),
        "improvement": {
            k: ml_results.get("metrics", {}).get(k, 0) - baseline_results.get("metrics", {}).get(k, 0)
            for k in baseline_results.get("metrics", {}).keys()
            if k != "n_predictions"
        }
    }
