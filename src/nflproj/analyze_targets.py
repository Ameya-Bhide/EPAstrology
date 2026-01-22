"""Analyze target prediction errors to understand what to improve."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import logging

from .config import PARQUET_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_target_errors(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    features: pd.DataFrame,
    player_game: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame
) -> Dict:
    """Analyze where target predictions are going wrong."""
    
    merged = pd.merge(
        predictions,
        actuals,
        on=["game_id", "player_id", "week"],
        how="inner"
    )
    
    # Add features and player info
    merged = pd.merge(merged, features, on=["game_id", "player_id"], how="left", suffixes=("", "_feat"))
    merged = pd.merge(merged, players[["player_id", "position"]], on="player_id", how="left")
    merged = pd.merge(merged, games[["game_id", "home_team", "away_team"]], on="game_id", how="left")
    
    # Calculate errors
    merged["targets_error"] = merged["proj_targets"] - merged["actual_targets"]
    merged["abs_targets_error"] = np.abs(merged["targets_error"])
    
    analysis = {}
    
    # 1. Error by position
    analysis["by_position"] = {}
    for position in ["WR", "TE", "RB"]:
        pos_data = merged[merged["position"] == position]
        if len(pos_data) > 0:
            analysis["by_position"][position] = {
                "n": len(pos_data),
                "mae": pos_data["abs_targets_error"].mean(),
                "bias": pos_data["targets_error"].mean(),  # Positive = overpredicting
                "rmse": np.sqrt((pos_data["targets_error"] ** 2).mean()),
                "mean_actual": pos_data["actual_targets"].mean(),
                "mean_pred": pos_data["proj_targets"].mean(),
            }
    
    # 2. Error by target volume (high vs low volume players)
    merged["target_volume"] = pd.cut(
        merged["actual_targets"],
        bins=[0, 2, 5, 10, 100],
        labels=["Low (0-2)", "Medium (3-5)", "High (6-10)", "Very High (10+)"]
    )
    analysis["by_volume"] = merged.groupby("target_volume").agg({
        "abs_targets_error": "mean",
        "targets_error": "mean",
        "actual_targets": "mean",
        "proj_targets": "mean"
    }).to_dict("index")
    
    # 3. Error by recent performance
    if "targets_last3" in merged.columns:
        merged["recent_targets"] = pd.cut(
            merged["targets_last3"],
            bins=[0, 5, 10, 20, 100],
            labels=["Low recent", "Medium recent", "High recent", "Very high recent"]
        )
        analysis["by_recent"] = merged.groupby("recent_targets").agg({
            "abs_targets_error": "mean",
            "targets_error": "mean"
        }).to_dict("index")
    
    # 4. Error by target share
    if "target_share_last6" in merged.columns:
        merged["share_level"] = pd.cut(
            merged["target_share_last6"].fillna(0),
            bins=[0, 0.05, 0.15, 0.25, 1.0],
            labels=["Low share", "Medium share", "High share", "Very high share"]
        )
        analysis["by_share"] = merged.groupby("share_level").agg({
            "abs_targets_error": "mean",
            "targets_error": "mean"
        }).to_dict("index")
    
    # 5. Worst predictions (over and under)
    analysis["worst_overpredictions"] = merged.nlargest(20, "targets_error")[
        ["game_id", "player_id", "position", "proj_targets", "actual_targets", "targets_error"]
    ].to_dict("records")
    
    analysis["worst_underpredictions"] = merged.nsmallest(20, "targets_error")[
        ["game_id", "player_id", "position", "proj_targets", "actual_targets", "targets_error"]
    ].to_dict("records")
    
    # 6. Feature importance (correlation with error)
    feature_cols = [c for c in merged.columns if any(x in c for x in ["last", "share", "slope", "rate", "epa"])]
    feature_errors = {}
    for col in feature_cols:
        if merged[col].notna().sum() > 100:  # Need enough data
            corr = merged[col].corr(merged["abs_targets_error"])
            if not np.isnan(corr):
                feature_errors[col] = abs(corr)
    
    analysis["feature_error_correlation"] = dict(sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)[:10])
    
    return analysis
