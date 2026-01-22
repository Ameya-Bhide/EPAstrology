"""CLI interface using Typer."""

import typer
from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .config import PARQUET_DIR, DB_PATH, DEFAULT_SEASONS
from .ingest import ingest_all
from .build_tables import build_all_tables
from .features import build_features_table
from .models import BaselineRoleModel, BaselineEfficiencyModel, project_player
from .backtest import walk_forward_backtest, compare_baseline_vs_ml
from .db import DB

app = typer.Typer(help="NFL offense player projection system")
console = Console()


@app.command()
def ingest(
    seasons: str = typer.Option(
        ",".join(map(str, DEFAULT_SEASONS)),
        "--seasons",
        help="Comma-separated list of seasons to ingest (e.g., '2020,2021,2022')"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for Parquet files"
    )
):
    """Ingest data from nfl_data_py."""
    console.print("[bold green]Ingesting data...[/bold green]")
    
    season_list = [int(s.strip()) for s in seasons.split(",")]
    output_path = Path(output_dir) if output_dir else None
    
    from .ingest import ingest_all
    data = ingest_all(season_list, output_path)
    
    console.print(f"[green]✓[/green] Ingested data for seasons: {season_list}")
    console.print(f"  - Play-by-play: {len(data['pbp'])} plays")
    console.print(f"  - Rosters: {len(data['rosters'])} player-seasons")
    console.print(f"  - Schedules: {len(data['schedules'])} games")


@app.command()
def build(
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir",
        help="Input directory with ingested Parquet files"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        help="Output directory for built tables"
    )
):
    """Build core tables from ingested data."""
    console.print("[bold green]Building tables...[/bold green]")
    
    input_path = Path(input_dir) if input_dir else PARQUET_DIR
    output_path = Path(output_dir) if output_dir else PARQUET_DIR
    
    tables = build_all_tables(
        pbp_path=input_path / "pbp.parquet",
        rosters_path=input_path / "rosters.parquet",
        schedules_path=input_path / "schedules.parquet",
        output_dir=output_path
    )
    
    console.print("[green]✓[/green] Built tables:")
    for name, df in tables.items():
        console.print(f"  - {name}: {len(df)} rows")


@app.command()
def features(
    input_dir: Optional[str] = typer.Option(
        None,
        "--input-dir",
        help="Input directory with core tables"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        "--output-path",
        help="Output path for features Parquet file"
    )
):
    """Build features table."""
    console.print("[bold green]Building features...[/bold green]")
    
    input_path = Path(input_dir) if input_dir else PARQUET_DIR
    output_path = Path(output_path) if output_path else input_path / "features_player_game_offense.parquet"
    
    features_df = build_features_table(
        player_game_path=input_path / "player_game_offense.parquet",
        team_game_path=input_path / "team_game_offense.parquet",
        games_path=input_path / "games.parquet",
        players_path=input_path / "players.parquet",
        output_path=output_path
    )
    
    console.print(f"[green]✓[/green] Built features: {len(features_df)} rows")


@app.command()
def search_players(
    query: str = typer.Argument(..., help="Player name to search for (partial match)"),
    position: Optional[str] = typer.Option(None, "--position", help="Filter by position"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """Search for players by name to find correct spelling."""
    console.print(f"[bold green]Searching for players matching '{query}'...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    try:
        players = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError:
        console.print("[red]Error:[/red] Players table not found. Run 'build' command first.")
        raise typer.Exit(1)
    
    # Search for players (case-insensitive, partial match)
    matches = players[
        players["name"].str.contains(query, case=False, na=False)
    ]
    
    if position:
        matches = matches[matches["position"] == position]
    
    if len(matches) == 0:
        console.print(f"[yellow]No players found matching '{query}'[/yellow]")
        raise typer.Exit(0)
    
    # Display results in a table
    table = Table(title=f"Players matching '{query}'")
    table.add_column("Player ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Position", style="green")
    
    for _, row in matches.head(50).iterrows():  # Limit to 50 results
        table.add_row(
            row["player_id"],
            row["name"],
            row["position"]
        )
    
    console.print(table)
    
    if len(matches) > 50:
        console.print(f"\n[yellow]Showing first 50 of {len(matches)} results. Use a more specific query to narrow down.[/yellow]")
    else:
        console.print(f"\n[green]Found {len(matches)} player(s)[/green]")
    
    console.print("\n[bold]Tip:[/bold] Copy the exact name from above to use in the 'report' command")


@app.command()
def report(
    player: str = typer.Argument(..., help="Player name to report on"),
    from_date: str = typer.Option(..., "--from", help="Start date (YYYY-MM-DD)"),
    to_date: str = typer.Option(..., "--to", help="End date (YYYY-MM-DD)"),
    team: Optional[str] = typer.Option(None, "--team", help="Filter by team"),
    position: Optional[str] = typer.Option(None, "--position", help="Filter by position"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """Generate a player report for a date range."""
    console.print(f"[bold green]Generating report for {player}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first to create tables.")
        raise typer.Exit(1)
    
    # Parse dates
    try:
        from_dt = pd.to_datetime(from_date)
        to_dt = pd.to_datetime(to_date)
    except ValueError:
        console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
        raise typer.Exit(1)
    
    # Find player
    player_matches = players[players["name"].str.contains(player, case=False, na=False)]
    if len(player_matches) == 0:
        console.print(f"[red]Error:[/red] Player '{player}' not found")
        console.print("\n[yellow]Tip:[/yellow] Use 'nflproj search-players <name>' to find the correct spelling")
        console.print(f"Example: nflproj search-players {player.split()[0] if ' ' in player else player}")
        raise typer.Exit(1)
    
    if len(player_matches) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple players found, using first match")
        console.print(player_matches[["player_id", "name", "position"]].to_string())
    
    player_id = player_matches.iloc[0]["player_id"]
    player_name = player_matches.iloc[0]["name"]
    player_pos = player_matches.iloc[0]["position"]
    
    # Filter by date range
    games_in_range = games[
        (games["date"] >= from_dt) &
        (games["date"] <= to_dt)
    ]
    
    if team:
        games_in_range = games_in_range[
            (games_in_range["home_team"] == team) |
            (games_in_range["away_team"] == team)
        ]
    
    game_ids = games_in_range["game_id"].tolist()
    
    # Get player stats
    player_stats = player_game[
        (player_game["player_id"] == player_id) &
        (player_game["game_id"].isin(game_ids))
    ]
    
    if team:
        player_stats = player_stats[player_stats["team"] == team]
    
    if len(player_stats) == 0:
        console.print(f"[yellow]No games found for {player_name} in date range[/yellow]")
        raise typer.Exit(0)
    
    # Merge with games for date/week
    player_stats = pd.merge(
        player_stats,
        games[["game_id", "date", "week", "season", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    
    # Display report
    table = Table(title=f"Report: {player_name} ({player_pos})")
    table.add_column("Date", style="cyan")
    table.add_column("Week", style="cyan")
    table.add_column("Team", style="magenta")
    table.add_column("Opp", style="magenta")
    table.add_column("Targets", justify="right")
    table.add_column("Carries", justify="right")
    table.add_column("EPA Total", justify="right", style="green")
    table.add_column("EPA/Target", justify="right")
    table.add_column("EPA/Rush", justify="right")
    
    for _, row in player_stats.sort_values("date").iterrows():
        table.add_row(
            row["date"].strftime("%Y-%m-%d"),
            str(row["week"]),
            row["team"],
            row["opponent"],
            str(int(row["targets"])),
            str(int(row["carries"])),
            f"{row['epa_total']:.2f}",
            f"{row['epa_per_target']:.2f}" if row["targets"] > 0 else "-",
            f"{row['epa_per_rush']:.2f}" if row["carries"] > 0 else "-"
        )
    
    console.print(table)
    
    # Summary stats
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Games: {len(player_stats)}")
    console.print(f"  Total Targets: {player_stats['targets'].sum()}")
    console.print(f"  Total Carries: {player_stats['carries'].sum()}")
    console.print(f"  Total EPA: {player_stats['epa_total'].sum():.2f}")
    if player_stats["targets"].sum() > 0:
        console.print(f"  Avg EPA/Target: {(player_stats['targets'] * player_stats['epa_per_target']).sum() / player_stats['targets'].sum():.2f}")
    if player_stats["carries"].sum() > 0:
        console.print(f"  Avg EPA/Rush: {(player_stats['carries'] * player_stats['epa_per_rush']).sum() / player_stats['carries'].sum():.2f}")


@app.command()
def project(
    player: str = typer.Argument(..., help="Player name"),
    week: Optional[int] = typer.Option(None, "--week", help="Week number"),
    season: Optional[int] = typer.Option(None, "--season", help="Season year"),
    date: Optional[str] = typer.Option(None, "--date", help="Game date (YYYY-MM-DD)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """Project a player's performance for a game."""
    console.print(f"[bold green]Projecting for {player}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        team_game = pd.read_parquet(data_path / "team_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
        features = pd.read_parquet(data_path / "features_player_game_offense.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' and 'features' commands first.")
        raise typer.Exit(1)
    
    # Find player
    player_matches = players[players["name"].str.contains(player, case=False, na=False)]
    if len(player_matches) == 0:
        console.print(f"[red]Error:[/red] Player '{player}' not found")
        console.print("\n[yellow]Tip:[/yellow] Use 'nflproj search-players <name>' to find the correct spelling")
        console.print(f"Example: nflproj search-players {player.split()[0] if ' ' in player else player}")
        raise typer.Exit(1)
    
    player_id = player_matches.iloc[0]["player_id"]
    player_name = player_matches.iloc[0]["name"]
    
    # Find game where player actually played
    if date:
        game_date = pd.to_datetime(date)
        # First try to find game where player played on this date
        player_games_on_date = player_game[
            (player_game["player_id"] == player_id) &
            (player_game["game_id"].isin(games[games["date"] == game_date]["game_id"]))
        ]
        if len(player_games_on_date) > 0:
            game_id = player_games_on_date.iloc[0]["game_id"]
        else:
            # Fall back to any game on that date
            game_matches = games[games["date"] == game_date]
            if len(game_matches) == 0:
                console.print(f"[red]Error:[/red] No games found on date {date}")
                raise typer.Exit(1)
            game_id = game_matches.iloc[0]["game_id"]
            console.print(f"[yellow]Warning:[/yellow] Player {player_name} not found in games on {date}, using first game")
    elif week and season:
        # First try to find game where player played in this week
        week_games = games[(games["week"] == week) & (games["season"] == season)]
        if len(week_games) == 0:
            console.print(f"[red]Error:[/red] No games found for week {week}, season {season}")
            raise typer.Exit(1)
        
        player_games_in_week = player_game[
            (player_game["player_id"] == player_id) &
            (player_game["game_id"].isin(week_games["game_id"]))
        ]
        if len(player_games_in_week) > 0:
            game_id = player_games_in_week.iloc[0]["game_id"]
        else:
            # Player didn't play in week 10 - suggest alternative
            console.print(f"[red]Error:[/red] Player {player_name} did not play in week {week}, season {season}")
            # Find what weeks they did play
            player_all_games = player_game[player_game["player_id"] == player_id]
            if len(player_all_games) > 0:
                player_games_with_info = player_all_games.merge(
                    games[["game_id", "week", "season", "date"]], on="game_id"
                )
                player_games_season = player_games_with_info[player_games_with_info["season"] == season]
                if len(player_games_season) > 0:
                    weeks_played = sorted(player_games_season["week"].unique())
                    console.print(f"\n[yellow]Player played in weeks: {', '.join(map(str, weeks_played))}[/yellow]")
                    console.print(f"Try: nflproj project \"{player_name}\" --week {weeks_played[0]} --season {season}")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --date or --week and --season")
        raise typer.Exit(1)
    
    # Initialize models
    role_model = BaselineRoleModel()
    role_model.fit(player_game, team_game, players)
    
    efficiency_model = BaselineEfficiencyModel()
    efficiency_model.fit(player_game, players)
    
    # Check if features exist for this player-game
    player_features = features[
        (features["player_id"] == player_id) &
        (features["game_id"] == game_id)
    ]
    
    if len(player_features) == 0:
        console.print(f"[red]Error:[/red] No features found for {player_name} in game {game_id}")
        console.print("\n[yellow]Possible reasons:[/yellow]")
        console.print("  1. Features not generated - run 'nflproj features' first")
        console.print("  2. Player didn't play in this game")
        console.print("  3. Player is a QB or other position not tracked for targets/carries")
        console.print(f"\n[yellow]Tip:[/yellow] This system tracks offensive skill positions (WR, RB, TE)")
        console.print("   For QBs, use 'report' to see passing stats instead")
        raise typer.Exit(1)
    
    # Make projection
    try:
        proj = project_player(
            player_id, game_id, role_model, efficiency_model,
            features, player_game, team_game, players, games
        )
        
        game_info = games[games["game_id"] == game_id].iloc[0]
        
        console.print(f"\n[bold]Projection for {player_name}[/bold]")
        console.print(f"Game: {game_info['home_team']} vs {game_info['away_team']}")
        console.print(f"Date: {game_info['date'].strftime('%Y-%m-%d')}")
        console.print(f"\n[bold]Role:[/bold]")
        console.print(f"  Projected Targets: {proj['proj_targets']:.1f}")
        console.print(f"  Projected Carries: {proj['proj_carries']:.1f}")
        console.print(f"\n[bold]Efficiency:[/bold]")
        console.print(f"  Projected EPA/Target: {proj['proj_epa_per_target']:.3f}")
        console.print(f"  Projected EPA/Rush: {proj['proj_epa_per_rush']:.3f}")
        console.print(f"\n[bold]Total:[/bold]")
        console.print(f"  Projected EPA Contribution: {proj['proj_epa_total']:.2f}")
        
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing column: {e}")
        console.print("\n[yellow]This might indicate:[/yellow]")
        console.print("  1. Tables need to be rebuilt - run 'nflproj build'")
        console.print("  2. Data structure mismatch")
        import traceback
        console.print(f"\n[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(f"\n[dim]Full error:[/dim]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def backtest(
    season: int = typer.Argument(..., help="Season to backtest"),
    model: str = typer.Option("baseline", "--model", help="Model type: baseline or ml"),
    compare: bool = typer.Option(False, "--compare", help="Compare baseline vs ML"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """Run walk-forward backtest for a season."""
    console.print(f"[bold green]Running backtest for {season}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        team_game = pd.read_parquet(data_path / "team_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
        features = pd.read_parquet(data_path / "features_player_game_offense.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' and 'features' commands first.")
        raise typer.Exit(1)
    
    if compare:
        results = compare_baseline_vs_ml(
            season, player_game, team_game, features, games, players
        )
        
        console.print("\n[bold]Baseline Metrics:[/bold]")
        for k, v in results["baseline"].items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            else:
                console.print(f"  {k}: {v}")
        
        console.print("\n[bold]ML Metrics:[/bold]")
        for k, v in results["ml"].items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            else:
                console.print(f"  {k}: {v}")
        
        console.print("\n[bold]Improvement (ML - Baseline):[/bold]")
        for k, v in results["improvement"].items():
            if isinstance(v, float):
                color = "green" if v < 0 else "red"  # Lower MAE is better
                console.print(f"  {k}: [{color}]{v:.4f}[/{color}]")
    else:
        results = walk_forward_backtest(
            season, player_game, team_game, features, games, players, model_type=model
        )
        
        if "error" in results:
            console.print(f"[red]Error:[/red] {results['error']}")
            raise typer.Exit(1)
        
        console.print("\n[bold]Backtest Results:[/bold]")
        for k, v in results["metrics"].items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            else:
                console.print(f"  {k}: {v}")


if __name__ == "__main__":
    app()
