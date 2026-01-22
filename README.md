# EPAstrology

NFL offense and defense player projection system focused on role (opportunity) + efficiency modeling, with a CLI interface and extensible data/model pipeline.

## Overview

EPAstrology is a two-stage projection system:
- **Stage A (Role)**: Predicts opportunities (targets, carries, pass attempts)
- **Stage B (Efficiency)**: Predicts efficiency (EPA per target/rush/attempt) with empirical Bayes shrinkage
- **Combined**: Projects total EPA contribution

The system uses walk-forward backtesting to evaluate model performance and supports both baseline and ML models. It includes support for offensive skill positions (WR, RB, TE), quarterbacks (QB), and defensive players.

## Features

- Ingest play-by-play, rosters, and schedules from `nfl_data_py`
- Build queryable datasets with core tables (games, players, player_game_offense, team_game_offense, player_game_defense, team_game_defense)
- Generate leakage-safe rolling features for offense and defense
- Baseline and ML models for role and efficiency prediction
- Position-specific models (WR, RB, TE, QB) for improved accuracy
- Walk-forward backtesting with baseline comparisons
- Comprehensive CLI interface for reports, projections, backtesting, and analysis
- QB support with passing stats (attempts, completions, yards, TDs, INTs, EPA)
- Defense support with defensive stats (tackles, sacks, interceptions, QB hits, fumbles, EPA allowed)
- Team-level offensive and defensive statistics
- Player comparison tools (side-by-side, with/without analysis)
- Extensible design for future enhancements

## Installation

### Requirements

- Python 3.11+
- pip

### Setup

1. Clone the repository:
```bash
cd EPAstrology
```

2. Install the package:
```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Ingest Data

Download NFL data for one or more seasons:

```bash
nflproj ingest --seasons 2020,2021,2022,2023,2024
```

This downloads play-by-play, rosters, and schedules from `nfl_data_py` and saves them as Parquet files in `data/parquet/`. By default, new data is appended to existing data. Use `--overwrite` to replace all data.

### 2. Build Core Tables

Process the ingested data into structured tables:

```bash
nflproj build
```

This creates:
- `games.parquet`: Game metadata (game_id, season, week, date, teams)
- `players.parquet`: Player information (player_id, name, position)
- `player_game_offense.parquet`: Player-level offensive game stats (targets, carries, receiving/rushing yards, EPA, QB passing stats)
- `team_game_offense.parquet`: Team-level offensive game stats (plays, pass rate, team EPA)
- `player_game_defense.parquet`: Player-level defensive game stats (tackles, sacks, interceptions, QB hits, fumbles, EPA allowed)
- `team_game_defense.parquet`: Team-level defensive game stats (plays faced, EPA allowed, sacks, interceptions, fumbles forced)

### 3. Build Features

Generate rolling features for modeling:

```bash
# Build offense features
nflproj features

# Build defense features
nflproj features --defense
```

This creates:
- `features_player_game_offense.parquet` with:
  - Rolling role stats (targets/carries/pass attempts last 1/3/6 games, target/carry/attempt share)
  - Rolling efficiency stats (EPA per opportunity, success rate, completion rate)
  - Trend features (slopes)
  - Context features (team pass rate, plays)
  - Opponent defense features
- `features_player_game_defense.parquet` with:
  - Rolling defensive stats (tackles, sacks, interceptions, QB hits, EPA allowed)
  - Share metrics (tackle share, sack share)
  - Trend features
  - Opponent offensive context

**Important**: Features are computed with no data leakage—only prior games are used.

### 4. Generate Reports

View a player's stats over a date range or season:

```bash
# Offensive player report
nflproj report "Patrick Mahomes" --season 2024
nflproj report "Travis Kelce" --from 2023-09-01 --to 2023-12-31

# With deep dive stats
nflproj report "Patrick Mahomes" --season 2024 --deep-dive

# Defensive player report
nflproj defense-report "Aaron Donald" --season 2023
```

Options:
- `--season`: Filter by season (includes playoffs through February)
- `--from` / `--to`: Filter by date range
- `--team`: Filter by team
- `--position`: Filter by position
- `--deep-dive`: Show detailed advanced stats
- `--data-dir`: Specify data directory

### 5. Make Projections

Project a player's performance for a specific game:

```bash
# By week and season
nflproj project "Travis Kelce" --week 10 --season 2023

# By date
nflproj project "Josh Allen" --date 2023-11-15

# Choose model type
nflproj project "Patrick Mahomes" --week 10 --season 2024 --model gbm_pos
```

Model options: `baseline`, `ridge`, `poisson`, `gbm`, `ridge_pos`, `gbm_pos`

### 6. Compare Players

Compare multiple players side-by-side:

```bash
nflproj compare-players "Travis Kelce,Mark Andrews" --season 2023
nflproj compare-players "Patrick Mahomes,Josh Allen,Lamar Jackson" --season 2024
```

### 7. Compare With/Without

Analyze how a player performs with and without another player:

```bash
nflproj compare-with-without "Patrick Mahomes" "Travis Kelce" --season 2023
nflproj compare-with-without "Josh Allen" "Stefon Diggs" --player1-position QB --season 2023
```

Use `--player1-position` and `--player2-position` to disambiguate when multiple players share the same name.

### 8. Team Statistics

View team-level statistics:

```bash
# Team offense
nflproj team-stats KC --season 2024

# Team defense
nflproj team-defense LA --season 2023
nflproj team-defense KC --from 2023-09-01 --to 2024-12-31
```

### 9. Rank Players

Rank players by position and any stat:

```bash
nflproj rank-players --position QB --sort-by epa_total --limit 10 --season 2024
nflproj rank-players --position WR --sort-by receiving_yards --limit 20 --season 2023
```

### 10. Batch Projections

Project multiple players at once:

```bash
nflproj batch-project "Travis Kelce,Mark Andrews" --week 10 --season 2023
nflproj batch-project "Patrick Mahomes,Josh Allen" --week 10 --season 2024 --output projections.csv
```

### 11. Search Players

Find the correct spelling of a player name:

```bash
nflproj search-players "Kelce"
nflproj search-players "McCaffrey" --position RB
```

### 12. Data Information

Check what data is available:

```bash
nflproj data-info
```

### 13. Run Backtests

Evaluate model performance with walk-forward backtesting:

```bash
# Baseline model
nflproj backtest 2023 --model baseline

# ML model
nflproj backtest 2023 --model gbm_pos

# Compare baseline vs ML
nflproj backtest 2023 --compare
```

Metrics reported:
- Role: MAE for targets/carries/pass attempts
- Efficiency: MAE for EPA per target/rush/attempt, rank correlation
- Total: MAE for total EPA contribution

## Project Structure

```
EPAstrology/
├── src/
│   └── nflproj/
│       ├── __init__.py
│       ├── config.py          # Configuration (paths, defaults)
│       ├── db.py               # DuckDB connection helpers
│       ├── ingest.py           # Data ingestion from nfl_data_py
│       ├── build_tables.py     # Core table construction (offense + defense)
│       ├── features.py          # Feature engineering (offense + defense)
│       ├── models.py            # Baseline and ML models
│       ├── backtest.py          # Walk-forward backtesting
│       └── cli.py               # Typer CLI interface
├── tests/
│   ├── test_shrinkage.py       # Tests for shrinkage functions
│   └── test_features.py        # Tests for feature computation
├── data/
│   ├── parquet/                # Parquet data files
│   └── db/                      # DuckDB database files
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

## Data Schema

### Core Tables

**games**
- `game_id`, `season`, `week`, `date`, `home_team`, `away_team`

**players**
- `player_id`, `name`, `position`

**player_game_offense**
- `game_id`, `player_id`, `team`, `opponent`
- `targets`, `receptions`, `receiving_yards`
- `carries`, `rushing_yards`
- `pass_attempts`, `completions`, `passing_yards`, `passing_tds`, `interceptions` (QB)
- `epa_total`, `epa_per_target`, `epa_per_rush`, `epa_per_attempt` (QB)
- `completion_rate` (QB)
- `success_rate`, `redzone_touches`

**team_game_offense**
- `game_id`, `team`
- `team_plays`, `dropbacks`, `rushes`, `pass_rate`
- `team_epa_per_play`

**player_game_defense**
- `game_id`, `player_id`, `team`, `opponent`
- `tackles`, `solo_tackles`, `assist_tackles`
- `sacks`, `qb_hits`, `interceptions`, `fumbles_forced`
- `epa_allowed`

**team_game_defense**
- `game_id`, `team`
- `plays_faced`, `dropbacks_faced`, `rushes_faced`
- `epa_allowed`, `epa_allowed_per_play`
- `sacks`, `interceptions`, `fumbles_forced`

**features_player_game_offense**
- `game_id`, `player_id`, `team`, `opponent`, `season`, `week`
- Rolling role: `targets_last1/3/6`, `carries_last1/3/6`, `pass_attempts_last1/3/6`, `target_share_last3/6`, `carry_share_last3/6`, `pass_attempt_share_last3/6`
- Rolling efficiency: `epa_per_target_last6`, `epa_per_rush_last6`, `epa_per_attempt_last6`, `completion_rate_last6`
- Trend: `target_share_slope_3`, `carry_share_slope_3`, `pass_attempt_share_slope_3`
- Context: `team_pass_rate_last6`, `team_plays_last6`
- Opponent: `opp_def_epa_allowed_last6`, `opp_def_pass_epa_allowed_last6`

**features_player_game_defense**
- `game_id`, `player_id`, `team`, `opponent`, `season`, `week`
- Rolling defensive: `tackles_last1/3/6`, `sacks_last1/3/6`, `interceptions_last1/3/6`, `qb_hits_last1/3/6`, `epa_allowed_last1/3/6`
- Share metrics: `tackle_share_last3/6`, `sack_share_last3/6`
- Trend: `tackle_share_slope_3`, `sack_share_slope_3`
- Context: `team_plays_faced_last6`, `team_pass_rate_faced_last6`
- Opponent: `opp_off_epa_per_play_last6`

## Models

### Baseline Models

**Role (BaselineRoleModel)**
- Weighted average of last 3 games (60%)
- Season-to-date share × projected team volume (40%)
- Supports targets, carries, and pass attempts (QB)

**Efficiency (BaselineEfficiencyModel)**
- Empirical Bayes shrinkage: `shrunk = (n × player_mean + k × position_mean) / (n + k)`
- Position-specific k values (WR/TE: 30 targets, RB: 40 carries, QB: 50 attempts)
- Supports EPA per target, EPA per rush, EPA per attempt, and completion rate

**Defense (BaselineDefenseModel)**
- Rolling averages for tackles, sacks, interceptions
- Shrinkage for EPA allowed

### ML Models

**Role (MLRoleModel)**
- Position-specific models (WR, RB, TE, QB)
- Options: Ridge, Poisson, or Gradient Boosting
- Features: rolling stats, shares, trends, context, opponent defense
- Tuned hyperparameters for better performance

**Efficiency**
- Currently uses baseline with shrinkage
- Can be extended with ML models

## Backtesting

Walk-forward backtesting ensures realistic evaluation:
1. For each week in a season:
   - Train on all games before that week (including previous seasons)
   - Predict that week
   - Compare predictions to actuals
2. Aggregate metrics across all weeks

This prevents look-ahead bias and simulates real-world usage.

## CLI Commands Reference

### Data Management
- `nflproj ingest [--seasons SEASONS] [--append/--overwrite]`: Download NFL data
- `nflproj build`: Build core tables from ingested data
- `nflproj features [--defense]`: Generate rolling features
- `nflproj data-info`: Show information about available data

### Reports and Analysis
- `nflproj report PLAYER [--season SEASON] [--from DATE] [--to DATE] [--deep-dive]`: Player offensive report
- `nflproj defense-report PLAYER [--season SEASON] [--from DATE] [--to DATE]`: Player defensive report
- `nflproj team-stats TEAM [--season SEASON] [--from DATE] [--to DATE]`: Team offensive statistics
- `nflproj team-defense TEAM [--season SEASON] [--from DATE] [--to DATE]`: Team defensive statistics
- `nflproj compare-players PLAYERS [--season SEASON] [--from DATE] [--to DATE]`: Compare multiple players
- `nflproj compare-with-without PLAYER1 PLAYER2 [--season SEASON] [--player1-position POS] [--player2-position POS]`: Compare player performance with/without another player
- `nflproj rank-players [--position POS] [--sort-by STAT] [--limit N] [--season SEASON]`: Rank players by position and stat

### Projections
- `nflproj project PLAYER [--week WEEK] [--season SEASON] [--date DATE] [--model MODEL]`: Project player performance
- `nflproj batch-project PLAYERS [--week WEEK] [--season SEASON] [--date DATE] [--model MODEL] [--output FILE]`: Project multiple players

### Utilities
- `nflproj search-players QUERY [--position POS]`: Search for players by name
- `nflproj backtest SEASON [--model MODEL] [--compare]`: Run walk-forward backtest

## Configuration

Edit `src/nflproj/config.py` to customize:
- Data directories
- Default seasons
- Shrinkage parameters (k values)
- Rolling window sizes

## Testing

Run tests:
```bash
pytest tests/ -v
```

Or use Make:
```bash
make test
```

## Makefile Commands

```bash
make install      # Install package
make ingest       # Ingest data
make build        # Build tables
make features     # Build features
make test         # Run tests
make clean        # Clean generated files
```

## Data Sources

- **nfl_data_py**: Python package providing NFL play-by-play, rosters, and schedules
- Data includes EPA (Expected Points Added) from nflfastR
- All metrics are public and reproducible

## Limitations

- No proprietary metrics (DVOA, etc.)
- Snaps/routes not included (can be added later)
- Simple baseline models (can be improved with more sophisticated ML)

## Future Enhancements

- [ ] Snaps and routes data
- [ ] More sophisticated ML models (XGBoost, neural networks)
- [ ] Web interface
- [ ] Real-time updates
- [ ] Advanced features (matchup-based, weather, etc.)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
