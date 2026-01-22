"""Feature engineering with leakage-safe rolling windows."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

from .config import PARQUET_DIR, ROLLING_WINDOWS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_rolling_features(
    player_game: pd.DataFrame,
    team_game: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute rolling features for player-game offense.
    
    Ensures no leakage by only using games before the current game.
    
    Schema: game_id, player_id, team, opponent, season, week,
            rolling_role: targets_last1/3/6, carries_last1/3/6, 
                          target_share_last3/6, carry_share_last3/6,
            rolling_eff: epa_per_target_shrunk_last6, epa_per_rush_shrunk_last6,
                         success_rate_last6,
            trend: target_share_slope_3, carry_share_slope_3,
            context: team_pass_rate_last6, team_plays_last6,
            opponent: opp_def_epa_allowed_last6
    """
    logger.info("Computing rolling features")
    
    windows = windows or ROLLING_WINDOWS
    
    # Merge with games to get season/week/date
    player_game = pd.merge(
        player_game,
        games[["game_id", "season", "week", "date"]],
        on="game_id",
        how="left"
    )
    
    # Merge with players to get position
    player_game = pd.merge(
        player_game,
        players[["player_id", "position"]],
        on="player_id",
        how="left"
    )
    
    # Sort by date to ensure chronological order
    player_game = player_game.sort_values(["player_id", "date"]).reset_index(drop=True)
    
    # Initialize feature columns
    features = player_game[["game_id", "player_id", "team", "opponent", "season", "week"]].copy()
    
    # For each window, compute rolling stats using only prior games
    for window in windows:
        # Targets and carries (last N games)
        features[f"targets_last{window}"] = 0
        features[f"carries_last{window}"] = 0
        
        # EPA per target and per rush (last N games, will be shrunk later)
        features[f"epa_per_target_last{window}"] = 0.0
        features[f"epa_per_rush_last{window}"] = 0.0
        
        # Success rate
        features[f"success_rate_last{window}"] = 0.0
        
        # QB features (pass attempts, completion rate, EPA per attempt)
        features[f"pass_attempts_last{window}"] = 0
        features[f"completion_rate_last{window}"] = 0.0
        features[f"epa_per_attempt_last{window}"] = 0.0
    
    # Target share and carry share (last 3 and 6)
    for window in [3, 6]:
        features[f"target_share_last{window}"] = 0.0
        features[f"carry_share_last{window}"] = 0.0
        features[f"pass_attempt_share_last{window}"] = 0.0  # QB's share of team pass attempts
    
    # Trend slopes
    features["target_share_slope_3"] = 0.0
    features["carry_share_slope_3"] = 0.0
    features["pass_attempt_share_slope_3"] = 0.0
    
    # Context features
    features["team_pass_rate_last6"] = 0.0
    features["team_plays_last6"] = 0.0
    
    # Opponent defense (EPA allowed per play, computed from prior games)
    features["opp_def_epa_allowed_last6"] = 0.0
    features["opp_def_pass_epa_allowed_last6"] = 0.0  # Opponent's pass defense EPA allowed
    
    # Compute features for each player-game using only prior games
    for idx, row in player_game.iterrows():
        player_id = row["player_id"]
        game_date = row["date"]
        team = row["team"]
        opponent = row["opponent"]
        season = row["season"]
        week = row["week"]
        
        # Get all prior games for this player (before current game)
        prior_games = player_game[
            (player_game["player_id"] == player_id) &
            (player_game["date"] < game_date)
        ].sort_values("date")
        
        # Get team's prior games
        team_prior = team_game[
            (team_game["team"] == team) &
            (team_game["game_id"].isin(
                games[games["date"] < game_date]["game_id"]
            ))
        ].sort_values("game_id")
        
        # Get opponent's prior games (for defense stats)
        opp_prior = team_game[
            (team_game["team"] == opponent) &
            (team_game["game_id"].isin(
                games[games["date"] < game_date]["game_id"]
            ))
        ].sort_values("game_id")
        
        # Rolling stats for each window
        for window in windows:
            window_games = prior_games.tail(window)
            
            if len(window_games) > 0:
                features.loc[idx, f"targets_last{window}"] = window_games["targets"].sum()
                features.loc[idx, f"carries_last{window}"] = window_games["carries"].sum()
                
                # EPA per opportunity (will be shrunk later)
                total_targets = window_games["targets"].sum()
                total_carries = window_games["carries"].sum()
                
                if total_targets > 0:
                    features.loc[idx, f"epa_per_target_last{window}"] = (
                        (window_games["targets"] * window_games["epa_per_target"]).sum() / total_targets
                    )
                if total_carries > 0:
                    features.loc[idx, f"epa_per_rush_last{window}"] = (
                        (window_games["carries"] * window_games["epa_per_rush"]).sum() / total_carries
                    )
                
                # Success rate
                total_plays = window_games["targets"].sum() + window_games["carries"].sum()
                if total_plays > 0:
                    features.loc[idx, f"success_rate_last{window}"] = (
                        (window_games["targets"] + window_games["carries"]) * window_games["success_rate"]
                    ).sum() / total_plays
                
                # QB features (if player is a QB)
                if "pass_attempts" in window_games.columns:
                    total_pass_attempts = window_games["pass_attempts"].sum()
                    features.loc[idx, f"pass_attempts_last{window}"] = total_pass_attempts
                    
                    if total_pass_attempts > 0:
                        # Completion rate
                        total_completions = window_games["completions"].sum()
                        features.loc[idx, f"completion_rate_last{window}"] = total_completions / total_pass_attempts
                        
                        # EPA per attempt
                        total_epa_passing = window_games["epa_passing"].sum()
                        features.loc[idx, f"epa_per_attempt_last{window}"] = total_epa_passing / total_pass_attempts
        
        # Target share and carry share (last 3 and 6)
        for window in [3, 6]:
            window_games = prior_games.tail(window)
            if len(window_games) > 0:
                player_targets = window_games["targets"].sum()
                player_carries = window_games["carries"].sum()
                
                # Get team totals for same games
                team_window_games = team_prior.tail(window)
                if len(team_window_games) > 0:
                    team_dropbacks = team_window_games["dropbacks"].sum()
                    team_rushes = team_window_games["rushes"].sum()
                    
                    if team_dropbacks > 0:
                        features.loc[idx, f"target_share_last{window}"] = player_targets / team_dropbacks
                    if team_rushes > 0:
                        features.loc[idx, f"carry_share_last{window}"] = player_carries / team_rushes
                    
                    # QB pass attempt share (QB's share of team dropbacks)
                    if "pass_attempts" in window_games.columns:
                        player_pass_attempts = window_games["pass_attempts"].sum()
                        if team_dropbacks > 0:
                            features.loc[idx, f"pass_attempt_share_last{window}"] = player_pass_attempts / team_dropbacks
        
        # Trend slopes (using last 3 games)
        window_games = prior_games.tail(3)
        if len(window_games) >= 2:
            # Compute target share for each of last 3 games
            target_shares = []
            carry_shares = []
            pass_attempt_shares = []
            
            for _, game_row in window_games.iterrows():
                game_id = game_row["game_id"]
                game_date_inner = game_row["date"]
                
                # Get team stats for this game
                team_game_stats = team_game[team_game["game_id"] == game_id]
                if len(team_game_stats) > 0:
                    team_dropbacks = team_game_stats.iloc[0]["dropbacks"]
                    team_rushes = team_game_stats.iloc[0]["rushes"]
                    
                    if team_dropbacks > 0:
                        target_shares.append(game_row["targets"] / team_dropbacks)
                        # QB pass attempt share
                        if "pass_attempts" in game_row and game_row["pass_attempts"] > 0:
                            pass_attempt_shares.append(game_row["pass_attempts"] / team_dropbacks)
                    if team_rushes > 0:
                        carry_shares.append(game_row["carries"] / team_rushes)
            
            # Compute slope if we have at least 2 points
            if len(target_shares) >= 2:
                features.loc[idx, "target_share_slope_3"] = np.polyfit(range(len(target_shares)), target_shares, 1)[0]
            if len(carry_shares) >= 2:
                features.loc[idx, "carry_share_slope_3"] = np.polyfit(range(len(carry_shares)), carry_shares, 1)[0]
            if len(pass_attempt_shares) >= 2:
                features.loc[idx, "pass_attempt_share_slope_3"] = np.polyfit(range(len(pass_attempt_shares)), pass_attempt_shares, 1)[0]
        
        # Context: team pass rate and plays (last 6)
        team_window = team_prior.tail(6)
        if len(team_window) > 0:
            total_plays = team_window["team_plays"].sum()
            total_dropbacks = team_window["dropbacks"].sum()
            if total_plays > 0:
                features.loc[idx, "team_pass_rate_last6"] = total_dropbacks / total_plays
                features.loc[idx, "team_plays_last6"] = total_plays / len(team_window)  # Average per game
        
        # Opponent defense: EPA allowed per play (last 6 games)
        # EPA allowed by opponent = EPA generated by offenses playing against opponent
        # Get games where opponent played (as defense)
        opp_games = games[
            (games["date"] < game_date) &
            ((games["home_team"] == opponent) | (games["away_team"] == opponent))
        ].sort_values("date").tail(6)
        
        if len(opp_games) > 0:
            opp_game_ids = opp_games["game_id"].tolist()
            
            # Get offensive stats from teams that played against this opponent
            # When team A plays opponent B, team A's offensive EPA is what B's defense allowed
            opp_def_stats = player_game[
                (player_game["game_id"].isin(opp_game_ids)) &
                (player_game["opponent"] == opponent)
            ]
            
            if len(opp_def_stats) > 0:
                # Average EPA per play allowed (total EPA / total plays)
                total_epa_allowed = opp_def_stats["epa_total"].sum()
                total_plays_allowed = (opp_def_stats["targets"].sum() + opp_def_stats["carries"].sum())
                
                if total_plays_allowed > 0:
                    features.loc[idx, "opp_def_epa_allowed_last6"] = total_epa_allowed / total_plays_allowed
                
                # Pass defense EPA allowed (for QBs)
                if "epa_passing" in opp_def_stats.columns:
                    total_pass_epa_allowed = opp_def_stats["epa_passing"].sum()
                    total_pass_attempts_allowed = opp_def_stats["pass_attempts"].sum() if "pass_attempts" in opp_def_stats.columns else 0
                    if total_pass_attempts_allowed > 0:
                        features.loc[idx, "opp_def_pass_epa_allowed_last6"] = total_pass_epa_allowed / total_pass_attempts_allowed
    
    # Apply shrinkage to EPA per target and per rush (last 6)
    # This will be done in the models module, but we can prepare the data here
    # For now, we'll keep the raw values and apply shrinkage in models
    
    logger.info("Rolling features computed")
    
    return features


def compute_rolling_features_defense(
    player_game_defense: pd.DataFrame,
    team_game_defense: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    player_game_offense: Optional[pd.DataFrame] = None,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute rolling features for player-game defense.
    
    Similar to offense features but for defensive stats.
    """
    logger.info("Computing rolling features for defense")
    
    windows = windows or ROLLING_WINDOWS
    
    # Merge with games to get season/week/date
    player_game_defense = pd.merge(
        player_game_defense,
        games[["game_id", "season", "week", "date"]],
        on="game_id",
        how="left"
    )
    
    # Merge with players to get position
    player_game_defense = pd.merge(
        player_game_defense,
        players[["player_id", "position"]],
        on="player_id",
        how="left"
    )
    
    # Sort by date
    player_game_defense = player_game_defense.sort_values(["player_id", "date"]).reset_index(drop=True)
    
    # Initialize feature columns
    features = player_game_defense[["game_id", "player_id", "team", "opponent", "season", "week"]].copy()
    
    # For each window, compute rolling stats
    for window in windows:
        features[f"tackles_last{window}"] = 0
        features[f"sacks_last{window}"] = 0
        features[f"interceptions_last{window}"] = 0
        if "qb_hits" in player_game_defense.columns:
            features[f"qb_hits_last{window}"] = 0
        features[f"epa_allowed_last{window}"] = 0.0
    
    # Share features (player's share of team defensive stats)
    for window in [3, 6]:
        features[f"tackle_share_last{window}"] = 0.0
        features[f"sack_share_last{window}"] = 0.0
    
    # Trend slopes
    features["tackle_share_slope_3"] = 0.0
    features["sack_share_slope_3"] = 0.0
    
    # Context features
    features["team_plays_faced_last6"] = 0.0
    features["team_pass_rate_faced_last6"] = 0.0
    
    # Opponent offense (EPA generated by opponent)
    features["opp_off_epa_per_play_last6"] = 0.0
    
    # Compute features for each player-game
    for idx, row in player_game_defense.iterrows():
        player_id = row["player_id"]
        game_date = row["date"]
        team = row["team"]
        opponent = row["opponent"]
        
        # Get prior games for this player
        prior_games = player_game_defense[
            (player_game_defense["player_id"] == player_id) &
            (player_game_defense["date"] < game_date)
        ].sort_values("date")
        
        # Get team's prior defensive games
        team_prior = team_game_defense[
            (team_game_defense["team"] == team) &
            (team_game_defense["game_id"].isin(
                games[games["date"] < game_date]["game_id"]
            ))
        ].sort_values("game_id")
        
        # Rolling stats for each window
        for window in windows:
            window_games = prior_games.tail(window)
            if len(window_games) > 0:
                features.loc[idx, f"tackles_last{window}"] = window_games["tackles"].sum()
                features.loc[idx, f"sacks_last{window}"] = window_games["sacks"].sum()
                features.loc[idx, f"interceptions_last{window}"] = window_games["interceptions"].sum()
                if "qb_hits" in window_games.columns:
                    features.loc[idx, f"qb_hits_last{window}"] = window_games["qb_hits"].sum()
                features.loc[idx, f"epa_allowed_last{window}"] = window_games["epa_allowed"].sum()
        
        # Share features (last 3 and 6) - need to aggregate team tackles/sacks
        for window in [3, 6]:
            window_games = prior_games.tail(window)
            if len(window_games) > 0:
                player_tackles = window_games["tackles"].sum()
                player_sacks = window_games["sacks"].sum()
                
                # Get team totals for same games (aggregate from player_game_defense)
                team_window_games = team_prior.tail(window)
                if len(team_window_games) > 0:
                    # Team tackles/sacks from team_game_defense (if available) or aggregate from players
                    team_tackles = 0
                    team_sacks = 0
                    for _, team_row in team_window_games.iterrows():
                        game_id_team = team_row["game_id"]
                        # Aggregate all player tackles/sacks for this team in this game
                        game_player_stats = player_game_defense[
                            (player_game_defense["game_id"] == game_id_team) &
                            (player_game_defense["team"] == team)
                        ]
                        team_tackles += game_player_stats["tackles"].sum()
                        team_sacks += game_player_stats["sacks"].sum()
                    
                    if team_tackles > 0:
                        features.loc[idx, f"tackle_share_last{window}"] = player_tackles / team_tackles
                    if team_sacks > 0:
                        features.loc[idx, f"sack_share_last{window}"] = player_sacks / team_sacks
        
        # Trend slopes
        window_games = prior_games.tail(3)
        if len(window_games) >= 2:
            tackle_shares = []
            for _, game_row in window_games.iterrows():
                game_id = game_row["game_id"]
                # Aggregate team tackles for this game
                game_player_stats = player_game_defense[
                    (player_game_defense["game_id"] == game_id) &
                    (player_game_defense["team"] == team)
                ]
                team_tackles = game_player_stats["tackles"].sum()
                if team_tackles > 0:
                    tackle_shares.append(game_row["tackles"] / team_tackles)
            if len(tackle_shares) >= 2:
                features.loc[idx, "tackle_share_slope_3"] = np.polyfit(range(len(tackle_shares)), tackle_shares, 1)[0]
        
        # Context: team plays faced and pass rate (last 6)
        team_window = team_prior.tail(6)
        if len(team_window) > 0:
            total_plays = team_window["plays_faced"].sum() if "plays_faced" in team_window.columns else 0
            total_dropbacks = team_window["dropbacks_faced"].sum() if "dropbacks_faced" in team_window.columns else 0
            if total_plays > 0:
                features.loc[idx, "team_pass_rate_faced_last6"] = total_dropbacks / total_plays
                features.loc[idx, "team_plays_faced_last6"] = total_plays / len(team_window)
        
        # Opponent offense: EPA per play (last 6 games) - need player_game_offense
        if player_game_offense is not None:
            opp_games = games[
                (games["date"] < game_date) &
                ((games["home_team"] == opponent) | (games["away_team"] == opponent))
            ].sort_values("date").tail(6)
            
            if len(opp_games) > 0:
                opp_game_ids = opp_games["game_id"].tolist()
                # Get offensive stats from opponent
                opp_off_stats = player_game_offense[
                    (player_game_offense["game_id"].isin(opp_game_ids)) &
                    (player_game_offense["team"] == opponent)
                ]
                if len(opp_off_stats) > 0:
                    total_epa = opp_off_stats["epa_total"].sum() if "epa_total" in opp_off_stats.columns else 0
                    total_plays = (opp_off_stats.get("targets", pd.Series([0])).sum() + 
                                  opp_off_stats.get("carries", pd.Series([0])).sum())
                    if total_plays > 0:
                        features.loc[idx, "opp_off_epa_per_play_last6"] = total_epa / total_plays
    
    logger.info("Rolling features for defense computed")
    return features


def build_features_table(
    player_game_path: Optional[Path] = None,
    team_game_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    players_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Build features table from core tables.
    
    Args:
        player_game_path: Path to player_game_offense Parquet
        team_game_path: Path to team_game_offense Parquet
        games_path: Path to games Parquet
        players_path: Path to players Parquet
        output_path: Path to save features Parquet
    """
    output_dir = output_path.parent if output_path else PARQUET_DIR
    output_dir = Path(output_dir)
    
    # Load tables
    logger.info("Loading core tables")
    player_game = pd.read_parquet(player_game_path or output_dir / "player_game_offense.parquet")
    team_game = pd.read_parquet(team_game_path or output_dir / "team_game_offense.parquet")
    games = pd.read_parquet(games_path or output_dir / "games.parquet")
    players = pd.read_parquet(players_path or output_dir / "players.parquet")
    
    # Compute features
    features = compute_rolling_features(player_game, team_game, games, players)
    
    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"Saved features table to {output_path}")
    
    return features
