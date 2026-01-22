"""Diagnostic tools for understanding model performance."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_prediction_errors(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    player_game: pd.DataFrame,
    players: pd.DataFrame
) -> Dict:
    """
    Analyze where predictions are going wrong.
    
    Returns:
        Dictionary with error analysis by position, player type, etc.
    """
    merged = pd.merge(
        predictions,
        actuals,
        on=["game_id", "player_id", "week"],
        how="inner"
    )
    
    # Merge with player info
    merged = pd.merge(merged, players[["player_id", "position"]], on="player_id", how="left")
    
    # Calculate errors
    merged["targets_error"] = merged["proj_targets"] - merged["actual_targets"]
    merged["carries_error"] = merged["proj_carries"] - merged["actual_carries"]
    merged["epa_error"] = merged["proj_epa_total"] - merged["actual_epa_total"]
    
    analysis = {
        "by_position": {},
        "by_opportunity_level": {},
        "extreme_errors": {}
    }
    
    # Analyze by position
    for position in merged["position"].dropna().unique():
        pos_data = merged[merged["position"] == position]
        analysis["by_position"][position] = {
            "n": len(pos_data),
            "targets_mae": np.abs(pos_data["targets_error"]).mean(),
            "carries_mae": np.abs(pos_data["carries_error"]).mean(),
            "epa_mae": np.abs(pos_data["epa_error"]).mean(),
            "targets_bias": pos_data["targets_error"].mean(),  # Positive = overpredicting
            "carries_bias": pos_data["carries_error"].mean(),
        }
    
    # Analyze by opportunity level (high vs low volume players)
    merged["total_opps"] = merged["actual_targets"] + merged["actual_carries"]
    high_vol = merged[merged["total_opps"] >= merged["total_opps"].quantile(0.75)]
    low_vol = merged[merged["total_opps"] <= merged["total_opps"].quantile(0.25)]
    
    analysis["by_opportunity_level"] = {
        "high_volume": {
            "n": len(high_vol),
            "targets_mae": np.abs(high_vol["targets_error"]).mean(),
            "carries_mae": np.abs(high_vol["carries_error"]).mean(),
            "epa_mae": np.abs(high_vol["epa_error"]).mean(),
        },
        "low_volume": {
            "n": len(low_vol),
            "targets_mae": np.abs(low_vol["targets_error"]).mean(),
            "carries_mae": np.abs(low_vol["carries_error"]).mean(),
            "epa_mae": np.abs(low_vol["epa_error"]).mean(),
        }
    }
    
    # Find extreme errors
    merged["abs_epa_error"] = np.abs(merged["epa_error"])
    extreme = merged.nlargest(20, "abs_epa_error")[
        ["game_id", "player_id", "proj_epa_total", "actual_epa_total", "epa_error", "position"]
    ]
    
    analysis["extreme_errors"] = extreme.to_dict("records")
    
    return analysis


def compare_baseline_vs_ml_errors(
    baseline_preds: pd.DataFrame,
    ml_preds: pd.DataFrame,
    actuals: pd.DataFrame
) -> pd.DataFrame:
    """Compare where baseline vs ML differ."""
    baseline_merged = pd.merge(
        baseline_preds,
        actuals,
        on=["game_id", "player_id", "week"],
        how="inner"
    )
    ml_merged = pd.merge(
        ml_preds,
        actuals,
        on=["game_id", "player_id", "week"],
        how="inner"
    )
    
    comparison = pd.merge(
        baseline_merged[["game_id", "player_id", "week", "proj_targets", "proj_carries", "proj_epa_total"]],
        ml_merged[["game_id", "player_id", "week", "proj_targets", "proj_carries", "proj_epa_total"]],
        on=["game_id", "player_id", "week"],
        suffixes=("_baseline", "_ml")
    )
    
    comparison = pd.merge(
        comparison,
        actuals[["game_id", "player_id", "week", "actual_targets", "actual_carries", "actual_epa_total"]],
        on=["game_id", "player_id", "week"]
    )
    
    # Calculate which is better
    comparison["baseline_targets_error"] = np.abs(comparison["proj_targets_baseline"] - comparison["actual_targets"])
    comparison["ml_targets_error"] = np.abs(comparison["proj_targets_ml"] - comparison["actual_targets"])
    comparison["baseline_carries_error"] = np.abs(comparison["proj_carries_baseline"] - comparison["actual_carries"])
    comparison["ml_carries_error"] = np.abs(comparison["proj_carries_ml"] - comparison["actual_carries"])
    
    comparison["ml_better_targets"] = comparison["ml_targets_error"] < comparison["baseline_targets_error"]
    comparison["ml_better_carries"] = comparison["ml_carries_error"] < comparison["baseline_carries_error"]
    
    return comparison
