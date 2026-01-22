# EPAstrology

NFL offense player projection system focused on role (opportunity) + efficiency modeling, with a CLI interface and extensible data/model pipeline.

## Overview

EPAstrology is a two-stage projection system:
- **Stage A (Role)**: Predicts opportunities (targets, carries)
- **Stage B (Efficiency)**: Predicts efficiency (EPA per target/rush) with empirical Bayes shrinkage
- **Combined**: Projects total EPA contribution

The system uses walk-forward backtesting to evaluate model performance and supports both baseline and ML models.

## Features

- ✅ Ingest play-by-play, rosters, and schedules from `nfl_data_py`
- ✅ Build queryable datasets with core tables (games, players, player_game_offense, team_game_offense)
- ✅ Generate leakage-safe rolling features
- ✅ Baseline and ML models for role and efficiency prediction
- ✅ Walk-forward backtesting with baseline comparisons
- ✅ CLI interface for reports, projections, and backtesting
- ✅ Extensible design for future defense modeling

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
nflproj ingest --seasons 2020,2021,2022,2023
```

This downloads play-by-play, rosters, and schedules from `nfl_data_py` and saves them as Parquet files in `data/parquet/`.

### 2. Build Core Tables

Process the ingested data into structured tables:

```bash
nflproj build
```

This creates:
- `games.parquet`: Game metadata (game_id, season, week, date, teams)
- `players.parquet`: Player information (player_id, name, position)
- `player_game_offense.parquet`: Player-level game stats (targets, carries, EPA, etc.)
- `team_game_offense.parquet`: Team-level game stats (plays, pass rate, team EPA)

### 3. Build Features

Generate rolling features for modeling:

```bash
nflproj features
```

This creates `features_player_game_offense.parquet` with:
- Rolling role stats (targets/carries last 1/3/6 games, target/carry share)
- Rolling efficiency stats (EPA per opportunity, success rate)
- Trend features (slopes)
- Context features (team pass rate, plays)
- Opponent defense features

**Important**: Features are computed with no data leakage—only prior games are used.

### 4. Generate Reports

View a player's stats over a date range:

```bash
nflproj report "Patrick Mahomes" --from 2023-09-01 --to 2023-12-31
```

Options:
- `--team`: Filter by team
- `--position`: Filter by position
- `--data-dir`: Specify data directory

### 5. Make Projections

Project a player's performance for a specific game:

```bash
nflproj project "Travis Kelce" --week 10 --season 2023
```

Or by date:
```bash
nflproj project "Josh Allen" --date 2023-11-15
```

### 6. Run Backtests

Evaluate model performance with walk-forward backtesting:

```bash
nflproj backtest 2023 --model baseline
```

Compare baseline vs ML:
```bash
nflproj backtest 2023 --compare
```

Metrics reported:
- Role: MAE for targets/carries
- Efficiency: MAE for EPA per target/rush, rank correlation
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
│       ├── build_tables.py     # Core table construction
│       ├── features.py          # Feature engineering
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
- `epa_total`, `epa_per_target`, `epa_per_rush`
- `success_rate`, `redzone_touches`

**team_game_offense**
- `game_id`, `team`
- `team_plays`, `dropbacks`, `rushes`, `pass_rate`
- `team_epa_per_play`

**features_player_game_offense**
- `game_id`, `player_id`, `team`, `opponent`, `season`, `week`
- Rolling role: `targets_last1/3/6`, `carries_last1/3/6`, `target_share_last3/6`, `carry_share_last3/6`
- Rolling efficiency: `epa_per_target_last6`, `epa_per_rush_last6`, `success_rate_last6`
- Trend: `target_share_slope_3`, `carry_share_slope_3`
- Context: `team_pass_rate_last6`, `team_plays_last6`
- Opponent: `opp_def_epa_allowed_last6`

## Models

### Baseline Models

**Role (BaselineRoleModel)**
- Weighted average of last 3 games (60%)
- Season-to-date share × projected team volume (40%)

**Efficiency (BaselineEfficiencyModel)**
- Empirical Bayes shrinkage: `shrunk = (n × player_mean + k × position_mean) / (n + k)`
- Position-specific k values (WR/TE: 30 targets, RB: 40 carries)

### ML Models

**Role (MLRoleModel)**
- Options: Ridge, Poisson, or Gradient Boosting
- Features: rolling stats, shares, trends, context, opponent defense

**Efficiency**
- Currently uses baseline with shrinkage
- Can be extended with ML models

## Backtesting

Walk-forward backtesting ensures realistic evaluation:
1. For each week in a season:
   - Train on all games before that week
   - Predict that week
   - Compare predictions to actuals
2. Aggregate metrics across all weeks

This prevents look-ahead bias and simulates real-world usage.

## Extending to Defense

The system is designed to be extended to defense by:
1. Creating parallel tables: `player_game_defense`, `team_game_defense`, `features_player_game_defense`
2. Reusing the same pipeline abstractions (ingest → build_tables → features → models)
3. Adding entity_type parameters to distinguish offense/defense

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

- Currently offense-only (defense extension planned)
- No proprietary metrics (DVOA, etc.)
- Snaps/routes not included in MVP (can be added later)
- Simple baseline models (can be improved with more sophisticated ML)

## Future Enhancements

- [ ] Defense modeling
- [ ] Snaps and routes data
- [ ] More sophisticated ML models (XGBoost, neural networks)
- [ ] Web interface
- [ ] Real-time updates
- [ ] Advanced features (matchup-based, weather, etc.)

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
