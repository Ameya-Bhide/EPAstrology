# Quick Start Guide

Follow these steps to try out the NFL projection system:

## Step 1: Install the Package

```bash
# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Step 2: Test Installation

Verify the CLI is available:
```bash
nflproj --help
```

## Step 3: Ingest Data (Start Small)

For a quick test, start with just one recent season:

```bash
# Ingest 2023 season (smaller, faster)
nflproj ingest --seasons 2023
```

This will download:
- Play-by-play data (~100-200 MB)
- Rosters
- Schedules

**Note**: First run may take a few minutes to download data.

## Step 4: Build Core Tables

Process the raw data into structured tables:

```bash
nflproj build
```

This creates:
- `games.parquet`
- `players.parquet`
- `player_game_offense.parquet`
- `team_game_offense.parquet`

## Step 5: Generate Features

Create rolling features for modeling:

```bash
nflproj features
```

This may take a minute or two depending on data size.

## Step 6: Try the CLI Commands

### Generate a Player Report

```bash
# Example: View a player's stats
nflproj report "Patrick Mahomes" --from 2023-09-01 --to 2023-12-31
```

Try other players:
```bash
nflproj report "Travis Kelce" --from 2023-09-01 --to 2023-12-31
nflproj report "Josh Allen" --from 2023-09-01 --to 2023-12-31
```

### Make a Projection

```bash
# Project for a specific week
nflproj project "Patrick Mahomes" --week 10 --season 2023

# Or by date
nflproj project "Travis Kelce" --date 2023-11-15
```

### Run a Backtest

```bash
# Test baseline model on 2023 season
nflproj backtest 2023 --model baseline

# Compare baseline vs ML
nflproj backtest 2023 --compare
```

## Troubleshooting

### If `nflproj` command not found:
- Make sure you ran `pip install -e .`
- Try: `python -m nflproj.cli` instead

### If data download fails:
- Check internet connection
- `nfl_data_py` may be slow on first download
- Try a single season first: `--seasons 2023`

### If build/features fail:
- Make sure you completed previous steps in order
- Check that Parquet files exist in `data/parquet/`

## Next Steps

Once you've tested with 2023, you can:
- Ingest more seasons: `nflproj ingest --seasons 2020,2021,2022,2023`
- Rebuild tables and features with more data
- Experiment with different models in backtests
