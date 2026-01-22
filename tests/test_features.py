"""Tests for feature computation (leakage safety)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from nflproj.features import compute_rolling_features


def create_test_data():
    """Create minimal test data."""
    # Create games
    dates = [datetime(2023, 9, 10) + timedelta(days=7*i) for i in range(5)]
    games = pd.DataFrame({
        "game_id": [f"game_{i}" for i in range(5)],
        "season": [2023] * 5,
        "week": list(range(1, 6)),
        "date": dates,
        "home_team": ["KC", "BUF", "KC", "BUF", "KC"],
        "away_team": ["BUF", "KC", "BUF", "KC", "BUF"]
    })
    
    # Create players
    players = pd.DataFrame({
        "player_id": ["player_1", "player_2"],
        "name": ["Player 1", "Player 2"],
        "position": ["WR", "RB"]
    })
    
    # Create player game stats
    player_game = pd.DataFrame({
        "game_id": [f"game_{i}" for i in range(5)] * 2,
        "player_id": ["player_1"] * 5 + ["player_2"] * 5,
        "team": ["KC"] * 5 + ["BUF"] * 5,
        "opponent": ["BUF"] * 5 + ["KC"] * 5,
        "targets": [5, 6, 7, 8, 9] + [0, 0, 0, 0, 0],
        "receptions": [4, 5, 6, 7, 8] + [0, 0, 0, 0, 0],
        "receiving_yards": [50, 60, 70, 80, 90] + [0, 0, 0, 0, 0],
        "carries": [0, 0, 0, 0, 0] + [10, 12, 14, 16, 18],
        "rushing_yards": [0, 0, 0, 0, 0] + [50, 60, 70, 80, 90],
        "epa_total": [2.0, 2.5, 3.0, 3.5, 4.0] + [1.5, 2.0, 2.5, 3.0, 3.5],
        "epa_per_target": [0.4, 0.42, 0.43, 0.44, 0.44] + [0.0, 0.0, 0.0, 0.0, 0.0],
        "epa_per_rush": [0.0, 0.0, 0.0, 0.0, 0.0] + [0.15, 0.17, 0.18, 0.19, 0.19],
        "success_rate": [0.6, 0.65, 0.7, 0.75, 0.8] + [0.5, 0.55, 0.6, 0.65, 0.7],
        "redzone_touches": [1, 1, 2, 2, 2] + [2, 2, 3, 3, 3]
    })
    
    # Create team game stats
    team_game = pd.DataFrame({
        "game_id": [f"game_{i}" for i in range(5)] * 2,
        "team": ["KC"] * 5 + ["BUF"] * 5,
        "team_plays": [60] * 10,
        "dropbacks": [40] * 5 + [35] * 5,
        "rushes": [20] * 5 + [25] * 5,
        "pass_rate": [0.67] * 5 + [0.58] * 5,
        "team_epa_per_play": [0.1] * 10
    })
    
    return games, players, player_game, team_game


def test_no_leakage():
    """Test that features only use data from prior games."""
    games, players, player_game, team_game = create_test_data()
    
    features = compute_rolling_features(player_game, team_game, games, players, windows=[1, 3])
    
    # For game_0 (first game), all rolling features should be 0 or NaN
    game_0_features = features[features["game_id"] == "game_0"]
    
    for player_id in ["player_1", "player_2"]:
        player_features = game_0_features[game_0_features["player_id"] == player_id]
        if len(player_features) > 0:
            # Should have no prior games, so rolling stats should be 0
            assert player_features["targets_last1"].iloc[0] == 0 or pd.isna(player_features["targets_last1"].iloc[0])
            assert player_features["targets_last3"].iloc[0] == 0 or pd.isna(player_features["targets_last3"].iloc[0])


def test_rolling_calculation():
    """Test that rolling features are calculated correctly."""
    games, players, player_game, team_game = create_test_data()
    
    features = compute_rolling_features(player_game, team_game, games, players, windows=[1, 3])
    
    # For player_1 in game_2, targets_last1 should be targets from game_1
    game_2_features = features[
        (features["game_id"] == "game_2") &
        (features["player_id"] == "player_1")
    ]
    
    if len(game_2_features) > 0:
        # game_1 had 6 targets, so targets_last1 should be 6
        assert game_2_features["targets_last1"].iloc[0] == 6
        
        # targets_last3 should be sum of game_0 and game_1 (5 + 6 = 11)
        # But wait, game_0 is the first game, so for game_2, last3 should include game_0 and game_1
        # Actually, game_2 is the 3rd game (index 2), so last3 should include game_0, game_1
        # That's only 2 games, so it should be 5 + 6 = 11
        assert game_2_features["targets_last3"].iloc[0] == 11


def test_features_include_all_columns():
    """Test that features table includes all required columns."""
    games, players, player_game, team_game = create_test_data()
    
    features = compute_rolling_features(player_game, team_game, games, players)
    
    required_cols = [
        "game_id", "player_id", "team", "opponent", "season", "week",
        "targets_last1", "targets_last3", "targets_last6",
        "carries_last1", "carries_last3", "carries_last6"
    ]
    
    for col in required_cols:
        assert col in features.columns, f"Missing column: {col}"
