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

app = typer.Typer(
    help="NFL player projection system for offense and defense",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def data_info(
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Show information about available data.
    
    Examples:
        nflproj data-info
    """
    console.print("[bold]Data Information[/bold]\n")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Check which tables exist
    tables_status = {}
    for table_name, file_name in [
        ("games", "games.parquet"),
        ("players", "players.parquet"),
        ("player_game_offense", "player_game_offense.parquet"),
        ("team_game_offense", "team_game_offense.parquet"),
        ("player_game_defense", "player_game_defense.parquet"),
        ("team_game_defense", "team_game_defense.parquet"),
        ("features_offense", "features_player_game_offense.parquet"),
        ("features_defense", "features_player_game_defense.parquet"),
    ]:
        file_path = data_path / file_name
        tables_status[table_name] = file_path.exists()
    
    # Show status
    console.print("[bold]Table Status:[/bold]")
    for table_name, exists in tables_status.items():
        status = "[green]✓[/green]" if exists else "[red]✗[/red]"
        console.print(f"  {status} {table_name}")
    
    if not tables_status["games"]:
        console.print("\n[yellow]No data found. Run 'nflproj ingest' and 'nflproj build' first.[/yellow]")
        raise typer.Exit(0)
    
    try:
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
        player_game_offense = pd.read_parquet(data_path / "player_game_offense.parquet") if tables_status["player_game_offense"] else pd.DataFrame()
        player_game_defense = pd.read_parquet(data_path / "player_game_defense.parquet") if tables_status["player_game_defense"] else pd.DataFrame()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first.")
        raise typer.Exit(1)
    
    console.print("\n[bold]Available Data:[/bold]")
    console.print(f"  Date Range: {games['date'].min().strftime('%Y-%m-%d')} to {games['date'].max().strftime('%Y-%m-%d')}")
    console.print(f"  Seasons: {', '.join(map(str, sorted(games['season'].unique())))}")
    console.print(f"  Total Games: {len(games):,}")
    console.print(f"  Total Players: {len(players):,}")
    if len(player_game_offense) > 0:
        console.print(f"  Offense Player-Games: {len(player_game_offense):,}")
    if len(player_game_defense) > 0:
        console.print(f"  Defense Player-Games: {len(player_game_defense):,}")
    
    console.print(f"\n[bold]Games by Season:[/bold]")
    season_counts = games.groupby('season').size()
    for season, count in season_counts.items():
        console.print(f"  {season}: {count:,} games")
    
    console.print(f"\n[bold]Players by Position:[/bold]")
    pos_counts = players.groupby('position').size().sort_values(ascending=False)
    for pos, count in pos_counts.head(10).items():
        console.print(f"  {pos}: {count:,} players")
    
    # Show next steps if features missing
    if not tables_status["features_offense"]:
        console.print("\n[yellow]Note:[/yellow] Run 'nflproj features' to build offense features for projections")
    if tables_status["player_game_defense"] and not tables_status["features_defense"]:
        console.print("[yellow]Note:[/yellow] Run 'nflproj features --defense' to build defense features")


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
    ),
    append: bool = typer.Option(True, "--append/--overwrite", help="Append to existing data (default) or overwrite")
):
    """
    Ingest data from nfl_data_py.
    
    Examples:
        nflproj ingest --seasons 2024
        nflproj ingest --seasons 2020,2021,2022,2023,2024
        nflproj ingest --seasons 2024 --overwrite
    """
    console.print("[bold green]Ingesting data...[/bold green]")
    
    season_list = [int(s.strip()) for s in seasons.split(",")]
    output_path = Path(output_dir) if output_dir else None
    
    from .ingest import ingest_all
    try:
        data = ingest_all(season_list, output_path, append=append)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\n[yellow]Note:[/yellow] Future seasons (2025+) may not have complete data available yet.")
        console.print("  NFL seasons typically start in September. Check back later for new data.")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] Ingested data for seasons: {season_list}")
    console.print(f"  - Play-by-play: {len(data['pbp']):,} plays")
    console.print(f"  - Rosters: {len(data['rosters']):,} player-seasons")
    console.print(f"  - Schedules: {len(data['schedules']):,} games")
    
    # Show date range
    if 'game_date' in data['pbp'].columns:
        date_range = f"{data['pbp']['game_date'].min()} to {data['pbp']['game_date'].max()}"
        console.print(f"  - Date Range: {date_range}")
    
    console.print(f"\n[yellow]Next Steps:[/yellow]")
    console.print(f"  1. Run 'nflproj build' to create tables")
    console.print(f"  2. Run 'nflproj features' to build features for projections")
    if any(s > 2024 for s in season_list):
        console.print(f"  3. Run 'nflproj features --defense' to build defense features")


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
        console.print(f"  - {name}: {len(df):,} rows")
    
    # Show summary
    if "player_game_offense" in tables:
        console.print(f"\n[dim]Offense: {len(tables['player_game_offense']):,} player-games[/dim]")
    if "player_game_defense" in tables:
        console.print(f"[dim]Defense: {len(tables['player_game_defense']):,} player-games[/dim]")


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
    ),
    defense: bool = typer.Option(False, "--defense", help="Build defense features instead of offense")
):
    """Build features table."""
    if defense:
        console.print("[bold green]Building defense features...[/bold green]")
    else:
        console.print("[bold green]Building offense features...[/bold green]")
    
    input_path = Path(input_dir) if input_dir else PARQUET_DIR
    output_path = Path(output_path) if output_path else input_path / "features_player_game_offense.parquet"
    
    from .features import compute_rolling_features_defense
    
    if defense:
        # Build defense features
        player_game_defense = pd.read_parquet(input_path / "player_game_defense.parquet")
        team_game_defense = pd.read_parquet(input_path / "team_game_defense.parquet")
        games = pd.read_parquet(input_path / "games.parquet")
        players = pd.read_parquet(input_path / "players.parquet")
        player_game_offense = pd.read_parquet(input_path / "player_game_offense.parquet") if (input_path / "player_game_offense.parquet").exists() else None
        
        features_df = compute_rolling_features_defense(
            player_game_defense, team_game_defense, games, players, player_game_offense
        )
        
        output_path = input_path / "features_player_game_defense.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_path, index=False, compression="snappy")
        console.print(f"[green]✓[/green] Built defense features: {len(features_df)} rows")
    else:
        # Build offense features
        features_df = build_features_table(
            player_game_path=input_path / "player_game_offense.parquet",
            team_game_path=input_path / "team_game_offense.parquet",
            games_path=input_path / "games.parquet",
            players_path=input_path / "players.parquet",
            output_path=output_path
        )
        console.print(f"[green]✓[/green] Built offense features: {len(features_df)} rows")


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
    """
    Search for players by name to find correct spelling.
    
    Examples:
        nflproj search-players "Kelce"
        nflproj search-players "McCaffrey" --position RB
    """
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
        console.print(f"\n[dim]Found {len(matches)} match(es). Use the exact name in other commands.[/dim]")
        if len(matches) == 1:
            console.print(f"[dim]Example: nflproj report \"{matches.iloc[0]['name']}\" --season 2023[/dim]")
    
    console.print("\n[bold]Tip:[/bold] Copy the exact name from above to use in the 'report' command")


@app.command()
def report(
    player: str = typer.Argument(..., help="Player name to report on"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season (includes playoffs through February)"),
    team: Optional[str] = typer.Option(None, "--team", help="Filter by team"),
    position: Optional[str] = typer.Option(None, "--position", help="Filter by position"),
    deep_dive: bool = typer.Option(False, "--deep-dive", help="Show detailed advanced stats"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Generate a player report for a date range or season.
    
    Examples:
        nflproj report "Travis Kelce" --season 2023
        nflproj report "Travis Kelce" --from 2023-09-01 --to 2023-12-31
        nflproj report "Patrick Mahomes" --season 2023 --deep-dive
        nflproj report "Kenneth Walker" --season 2023 --team SEA
        nflproj report "Tom Brady" --season 2020
    """
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
    
    # Parse dates - if season is provided, use that instead
    if season:
        # NFL seasons run from September to February (next year)
        # Season 2024 = Sep 2024 to Feb 2025
        from_dt = pd.to_datetime(f"{season}-09-01")
        to_dt = pd.to_datetime(f"{season + 1}-02-28")  # Include through end of February
        console.print(f"[dim]Using season {season} date range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}[/dim]")
    elif from_date and to_date:
        try:
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
        except ValueError:
            console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --season or both --from and --to")
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
    
    if len(games_in_range) == 0:
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found in date range {date_range_str}[/yellow]")
        console.print(f"Available date range: {games['date'].min().strftime('%Y-%m-%d')} to {games['date'].max().strftime('%Y-%m-%d')}")
        console.print(f"Available seasons: {', '.join(map(str, sorted(games['season'].unique())))}")
        raise typer.Exit(0)
    
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
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found for {player_name} in date range {date_range_str}[/yellow]")
        # Show what seasons/weeks the player did play
        player_all_games = player_game[player_game["player_id"] == player_id]
        if len(player_all_games) > 0:
            player_all_games = pd.merge(player_all_games, games[["game_id", "date", "season", "week"]], on="game_id")
            console.print(f"\n[yellow]Player played in seasons: {', '.join(map(str, sorted(player_all_games['season'].unique())))}[/yellow]")
            console.print(f"Date range: {player_all_games['date'].min().strftime('%Y-%m-%d')} to {player_all_games['date'].max().strftime('%Y-%m-%d')}")
            console.print(f"\n[yellow]Try:[/yellow] nflproj report \"{player_name}\" --from {player_all_games['date'].min().strftime('%Y-%m-%d')} --to {player_all_games['date'].max().strftime('%Y-%m-%d')}")
        raise typer.Exit(0)
    
    # Merge with games for date/week
    player_stats = pd.merge(
        player_stats,
        games[["game_id", "date", "week", "season", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    
    # Check if player is a QB based on position
    is_qb = player_pos == "QB"
    
    # Display report
    if is_qb:
        # QB-focused report
        table = Table(title=f"Report: {player_name} ({player_pos})")
        table.add_column("Date", style="cyan")
        table.add_column("Week", style="cyan")
        table.add_column("Team", style="magenta")
        table.add_column("Opp", style="magenta")
        table.add_column("Att", justify="right")
        table.add_column("Comp", justify="right")
        table.add_column("Comp %", justify="right")
        table.add_column("Pass Yds", justify="right")
        table.add_column("Yds/Att", justify="right")
        table.add_column("TDs", justify="right")
        table.add_column("INTs", justify="right")
        table.add_column("EPA Total", justify="right", style="green")
        table.add_column("EPA/Att", justify="right")
        
        for _, row in player_stats.sort_values("date").iterrows():
            attempts = row.get("pass_attempts", 0)
            completions = row.get("completions", 0)
            pass_yards = row.get("passing_yards", 0)
            comp_pct = (completions / attempts * 100) if attempts > 0 else 0.0
            yds_per_att = (pass_yards / attempts) if attempts > 0 else 0.0
            
            table.add_row(
                row["date"].strftime("%Y-%m-%d"),
                str(row["week"]),
                row["team"],
                row["opponent"],
                str(int(attempts)),
                str(int(completions)),
                f"{comp_pct:.1f}%",
                str(int(pass_yards)),
                f"{yds_per_att:.2f}",
                str(int(row.get("passing_tds", 0))),
                str(int(row.get("interceptions", 0))),
                f"{row['epa_total']:.2f}",
                f"{row.get('epa_per_attempt', 0):.3f}" if attempts > 0 else "-"
            )
        
        # Add total row for QB
        if len(player_stats) > 0:
            total_attempts = player_stats.get("pass_attempts", pd.Series([0])).sum()
            total_completions = player_stats.get("completions", pd.Series([0])).sum()
            total_pass_yards = player_stats.get("passing_yards", pd.Series([0])).sum()
            total_tds = player_stats.get("passing_tds", pd.Series([0])).sum()
            total_ints = player_stats.get("interceptions", pd.Series([0])).sum()
            total_epa = player_stats["epa_total"].sum()
            total_epa_passing = player_stats.get("epa_passing", pd.Series([0])).sum()
            avg_comp_pct = (total_completions / total_attempts * 100) if total_attempts > 0 else 0.0
            avg_yds_per_att = (total_pass_yards / total_attempts) if total_attempts > 0 else 0.0
            avg_epa_per_att = (total_epa_passing / total_attempts) if total_attempts > 0 else 0.0
            
            table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{len(player_stats)}[/bold]",
                "-",
                "-",
                f"[bold]{int(total_attempts)}[/bold]",
                f"[bold]{int(total_completions)}[/bold]",
                f"[bold]{avg_comp_pct:.1f}%[/bold]",
                f"[bold]{int(total_pass_yards)}[/bold]",
                f"[bold]{avg_yds_per_att:.2f}[/bold]",
                f"[bold]{int(total_tds)}[/bold]",
                f"[bold]{int(total_ints)}[/bold]",
                f"[bold]{total_epa:.2f}[/bold]",
                f"[bold]{avg_epa_per_att:.3f}[/bold]" if total_attempts > 0 else "-"
            )
    else:
        # Skill position report
        table = Table(title=f"Report: {player_name} ({player_pos})")
        table.add_column("Date", style="cyan")
        table.add_column("Week", style="cyan")
        table.add_column("Team", style="magenta")
        table.add_column("Opp", style="magenta")
        table.add_column("Targets", justify="right")
        table.add_column("Rec", justify="right")
        table.add_column("Rec Yds", justify="right")
        table.add_column("Yds/Tgt", justify="right")
        table.add_column("Carries", justify="right")
        table.add_column("Rush Yds", justify="right")
        table.add_column("Yds/Carry", justify="right")
        table.add_column("EPA Total", justify="right", style="green")
        table.add_column("EPA/Tgt", justify="right")
        table.add_column("EPA/Rush", justify="right")
        
        for _, row in player_stats.sort_values("date").iterrows():
            targets = row["targets"]
            receptions = row.get("receptions", 0)
            rec_yards = row["receiving_yards"]
            carries = row["carries"]
            rush_yards = row["rushing_yards"]
            
            yds_per_target = (rec_yards / targets) if targets > 0 else 0.0
            yds_per_carry = (rush_yards / carries) if carries > 0 else 0.0
            
            table.add_row(
                row["date"].strftime("%Y-%m-%d"),
                str(row["week"]),
                row["team"],
                row["opponent"],
                str(int(targets)),
                str(int(receptions)) if targets > 0 else "-",
                str(int(rec_yards)) if targets > 0 else "-",
                f"{yds_per_target:.2f}" if targets > 0 else "-",
                str(int(carries)),
                str(int(rush_yards)) if carries > 0 else "-",
                f"{yds_per_carry:.2f}" if carries > 0 else "-",
                f"{row['epa_total']:.2f}",
                f"{row['epa_per_target']:.2f}" if targets > 0 else "-",
                f"{row['epa_per_rush']:.2f}" if carries > 0 else "-"
            )
        
        # Add total row for skill positions
        if len(player_stats) > 0:
            total_targets = player_stats["targets"].sum()
            total_receptions = player_stats.get("receptions", pd.Series([0])).sum()
            total_rec_yards = player_stats["receiving_yards"].sum()
            total_carries = player_stats["carries"].sum()
            total_rush_yards = player_stats["rushing_yards"].sum()
            total_epa = player_stats["epa_total"].sum()
            total_epa_receiving = player_stats.get("epa_receiving", pd.Series([0])).sum()
            total_epa_rushing = player_stats.get("epa_rushing", pd.Series([0])).sum()
            avg_yds_per_target = (total_rec_yards / total_targets) if total_targets > 0 else 0.0
            avg_yds_per_carry = (total_rush_yards / total_carries) if total_carries > 0 else 0.0
            avg_epa_per_target = (total_epa_receiving / total_targets) if total_targets > 0 else 0.0
            avg_epa_per_rush = (total_epa_rushing / total_carries) if total_carries > 0 else 0.0
            
            table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{len(player_stats)}[/bold]",
                "-",
                "-",
                f"[bold]{int(total_targets)}[/bold]",
                f"[bold]{int(total_receptions)}[/bold]",
                f"[bold]{int(total_rec_yards)}[/bold]",
                f"[bold]{avg_yds_per_target:.2f}[/bold]" if total_targets > 0 else "-",
                f"[bold]{int(total_carries)}[/bold]",
                f"[bold]{int(total_rush_yards)}[/bold]",
                f"[bold]{avg_yds_per_carry:.2f}[/bold]" if total_carries > 0 else "-",
                f"[bold]{total_epa:.2f}[/bold]",
                f"[bold]{avg_epa_per_target:.2f}[/bold]" if total_targets > 0 else "-",
                f"[bold]{avg_epa_per_rush:.2f}[/bold]" if total_carries > 0 else "-"
            )
    
    console.print(table)
    
    # Deep dive stats if requested
    if deep_dive:
        console.print("\n[bold]Deep Dive Stats:[/bold]")
        
        if is_qb:
            # QB deep dive
            total_attempts = player_stats.get("pass_attempts", pd.Series([0])).sum()
            total_completions = player_stats.get("completions", pd.Series([0])).sum()
            total_pass_yards = player_stats.get("passing_yards", pd.Series([0])).sum()
            total_tds = player_stats.get("passing_tds", pd.Series([0])).sum()
            total_ints = player_stats.get("interceptions", pd.Series([0])).sum()
            total_carries = player_stats["carries"].sum()
            total_rush_yards = player_stats["rushing_yards"].sum()
            total_epa = player_stats["epa_total"].sum()
            epa_passing = player_stats.get("epa_passing", pd.Series([0])).sum()
            epa_rushing = player_stats.get("epa_rushing", pd.Series([0])).sum()
            
            if total_attempts > 0:
                console.print(f"\n[bold]Passing:[/bold]")
                console.print(f"  Pass Attempts: {total_attempts:.0f}")
                console.print(f"  Completions: {total_completions:.0f}")
                console.print(f"  Completion Rate: {total_completions / total_attempts:.1%}")
                console.print(f"  Yards/Attempt: {total_pass_yards / total_attempts:.2f}")
                console.print(f"  TD Rate: {total_tds / total_attempts:.1%}")
                console.print(f"  INT Rate: {total_ints / total_attempts:.1%}")
                console.print(f"  TD:INT Ratio: {total_tds / total_ints:.2f}" if total_ints > 0 else "  TD:INT Ratio: ∞")
                console.print(f"  EPA/Attempt: {epa_passing / total_attempts:.3f}")
                console.print(f"  Total Passing EPA: {epa_passing:.2f}")
                console.print(f"  Avg Attempts/Game: {total_attempts / len(player_stats):.1f}")
                console.print(f"  Avg Passing Yards/Game: {total_pass_yards / len(player_stats):.1f}")
            
            if total_carries > 0:
                console.print(f"\n[bold]Rushing:[/bold]")
                console.print(f"  Carries: {total_carries:.0f}")
                console.print(f"  Yards/Carry: {total_rush_yards / total_carries:.2f}")
                console.print(f"  EPA/Rush: {epa_rushing / total_carries:.3f}")
                console.print(f"  Total Rushing EPA: {epa_rushing:.2f}")
                console.print(f"  Avg Carries/Game: {total_carries / len(player_stats):.1f}")
                console.print(f"  Avg Rushing Yards/Game: {total_rush_yards / len(player_stats):.1f}")
            
            console.print(f"\n[bold]Overall:[/bold]")
            console.print(f"  Total EPA: {total_epa:.2f}")
            console.print(f"  EPA/Game: {total_epa / len(player_stats):.2f}")
        else:
            # Skill position deep dive
            total_targets = player_stats["targets"].sum()
            total_receptions = player_stats["receptions"].sum()
            total_rec_yards = player_stats["receiving_yards"].sum()
            total_carries = player_stats["carries"].sum()
            total_rush_yards = player_stats["rushing_yards"].sum()
            total_epa = player_stats["epa_total"].sum()
            epa_receiving = player_stats.get("epa_receiving", pd.Series([0])).sum()
            epa_rushing = player_stats.get("epa_rushing", pd.Series([0])).sum()
            
            if total_targets > 0:
                console.print(f"\n[bold]Receiving:[/bold]")
                console.print(f"  Targets: {total_targets:.0f}")
                console.print(f"  Receptions: {total_receptions:.0f}")
                console.print(f"  Catch Rate: {total_receptions / total_targets:.1%}")
                console.print(f"  Yards/Target: {total_rec_yards / total_targets:.2f}")
                console.print(f"  Yards/Reception: {total_rec_yards / total_receptions:.2f}" if total_receptions > 0 else "  Yards/Reception: -")
                console.print(f"  EPA/Target: {epa_receiving / total_targets:.3f}")
                console.print(f"  Total Receiving EPA: {epa_receiving:.2f}")
                console.print(f"  Avg Targets/Game: {total_targets / len(player_stats):.1f}")
                console.print(f"  Avg Receiving Yards/Game: {total_rec_yards / len(player_stats):.1f}")
            
            if total_carries > 0:
                console.print(f"\n[bold]Rushing:[/bold]")
                console.print(f"  Yards/Carry: {total_rush_yards / total_carries:.2f}")
                console.print(f"  EPA/Rush: {epa_rushing / total_carries:.3f}")
                console.print(f"  Total Rushing EPA: {epa_rushing:.2f}")
            
            console.print(f"\n[bold]Overall:[/bold]")
            console.print(f"  Total Touches: {total_targets + total_carries:.0f}")
            console.print(f"  Total Yards: {total_rec_yards + total_rush_yards:.0f}")
            console.print(f"  Total EPA: {total_epa:.2f}")
            console.print(f"  EPA/Game: {total_epa / len(player_stats):.2f}")
            console.print(f"  Success Rate: {player_stats['success_rate'].mean():.1%}")
            console.print(f"  Redzone Touches: {player_stats['redzone_touches'].sum():.0f}")
    
    # Summary stats
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Games: {len(player_stats)}")
    
    if is_qb:
        # QB summary
        total_attempts = player_stats.get("pass_attempts", pd.Series([0])).sum()
        total_completions = player_stats.get("completions", pd.Series([0])).sum()
        total_pass_yards = player_stats.get("passing_yards", pd.Series([0])).sum()
        total_tds = player_stats.get("passing_tds", pd.Series([0])).sum()
        total_ints = player_stats.get("interceptions", pd.Series([0])).sum()
        
        console.print(f"  Total Pass Attempts: {total_attempts:.0f}")
        console.print(f"  Total Completions: {total_completions:.0f}")
        console.print(f"  Total Passing Yards: {total_pass_yards:.0f}")
        console.print(f"  Total Passing TDs: {total_tds:.0f}")
        console.print(f"  Total Interceptions: {total_ints:.0f}")
        console.print(f"  Total EPA: {player_stats['epa_total'].sum():.2f}")
        if total_attempts > 0:
            console.print(f"  Completion Rate: {total_completions / total_attempts:.1%}")
            console.print(f"  Avg EPA/Attempt: {(player_stats.get('pass_attempts', 0) * player_stats.get('epa_per_attempt', 0)).sum() / total_attempts:.3f}")
            console.print(f"  Avg Yards/Attempt: {total_pass_yards / total_attempts:.2f}")
        
        # Also show rushing if QB has rushing stats
        if player_stats["carries"].sum() > 0:
            console.print(f"  Total Carries: {player_stats['carries'].sum():.0f}")
            console.print(f"  Total Rushing Yards: {player_stats['rushing_yards'].sum():.0f}")
    else:
        # Skill position summary
        console.print(f"  Total Targets: {player_stats['targets'].sum()}")
        console.print(f"  Total Receiving Yards: {player_stats['receiving_yards'].sum()}")
        console.print(f"  Total Carries: {player_stats['carries'].sum()}")
        console.print(f"  Total Rushing Yards: {player_stats['rushing_yards'].sum()}")
        console.print(f"  Total EPA: {player_stats['epa_total'].sum():.2f}")
        if player_stats["targets"].sum() > 0:
            console.print(f"  Avg EPA/Target: {(player_stats['targets'] * player_stats['epa_per_target']).sum() / player_stats['targets'].sum():.2f}")
            console.print(f"  Avg Yards/Target: {player_stats['receiving_yards'].sum() / player_stats['targets'].sum():.2f}")
        if player_stats["carries"].sum() > 0:
            console.print(f"  Avg EPA/Rush: {(player_stats['carries'] * player_stats['epa_per_rush']).sum() / player_stats['carries'].sum():.2f}")
            console.print(f"  Avg Yards/Carry: {player_stats['rushing_yards'].sum() / player_stats['carries'].sum():.2f}")


@app.command()
def project(
    player: str = typer.Argument(..., help="Player name"),
    week: Optional[int] = typer.Option(None, "--week", help="Week number"),
    season: Optional[int] = typer.Option(None, "--season", help="Season year"),
    date: Optional[str] = typer.Option(None, "--date", help="Game date (YYYY-MM-DD)"),
    model: str = typer.Option("baseline", "--model", help="Model type: baseline, ridge, poisson, gbm, ridge_pos, or gbm_pos"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Project a player's performance for a game.
    
    Examples:
        nflproj project "Travis Kelce" --week 10 --season 2023
        nflproj project "Travis Kelce" --week 10 --season 2023 --model gbm_pos
        nflproj project "Patrick Mahomes" --date 2023-11-15 --model baseline
    """
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
    
    # Initialize models based on model type
    from .models import MLRoleModel
    
    if model == "baseline":
        role_model = BaselineRoleModel()
        role_model.fit(player_game, team_game, players)
    else:
        # Use ML model
        position_specific = model.endswith("_pos") or model == "gbm_pos" or model == "ridge_pos"
        model_type_clean = model.replace("_pos", "")
        
        # Get hyperparameters
        ml_kwargs = {}
        if model_type_clean == "ridge":
            ml_kwargs = {"alpha": 0.5}
        elif model_type_clean == "gbm":
            ml_kwargs = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_split": 20,
                "min_samples_leaf": 5
            }
        
        role_model = MLRoleModel(model_type=model_type_clean, position_specific=position_specific)
        role_model.kwargs = ml_kwargs
        role_model.fit(features, player_game, players, team_game, games)
    
    efficiency_model = BaselineEfficiencyModel()
    efficiency_model.fit(player_game, players)
    
    # Get player position
    player_info = players[players["player_id"] == player_id]
    player_pos = player_info.iloc[0]["position"] if len(player_info) > 0 else None
    
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
        raise typer.Exit(1)
    
    # Make projection
    try:
        # Check if player is a QB
        is_qb = player_pos == "QB"
        
        if is_qb:
            # QB projection
            if isinstance(role_model, BaselineRoleModel):
                proj_pass_attempts = role_model.predict_pass_attempts(
                    player_features, player_game, team_game, players, games
                ).iloc[0]
            else:
                proj_pass_attempts = role_model.predict_pass_attempts(
                    player_features, players, team_game, games
                ).iloc[0]
            
            proj_epa_per_attempt = efficiency_model.predict_epa_per_attempt(
                player_features, player_game, players, games
            ).iloc[0]
            proj_completion_rate = efficiency_model.predict_completion_rate(
                player_features, player_game, players, games
            ).iloc[0]
            
            proj = {
                "proj_pass_attempts": proj_pass_attempts,
                "proj_epa_per_attempt": proj_epa_per_attempt,
                "proj_completion_rate": proj_completion_rate,
                "proj_epa_total": proj_pass_attempts * proj_epa_per_attempt
            }
            
            # Calculate projected passing yards
            player_historical = player_game[player_game["player_id"] == player_id]
            if len(player_historical) > 0 and "pass_attempts" in player_historical.columns:
                hist_pass_yards = player_historical["passing_yards"].sum()
                hist_attempts = player_historical["pass_attempts"].sum()
                proj_pass_yards = (proj_pass_attempts * (hist_pass_yards / hist_attempts)) if hist_attempts > 0 else 0
                proj_completions = proj_pass_attempts * proj_completion_rate
            else:
                proj_pass_yards = 0
                proj_completions = 0
            
            game_info = games[games["game_id"] == game_id].iloc[0]
            
            console.print(f"\n[bold]Projection for {player_name} (QB)[/bold]")
            console.print(f"Model: {model}")
            console.print(f"Game: {game_info['home_team']} vs {game_info['away_team']}")
            console.print(f"Date: {game_info['date'].strftime('%Y-%m-%d')}")
            console.print(f"\n[bold]Role:[/bold]")
            console.print(f"  Projected Pass Attempts: {proj['proj_pass_attempts']:.1f}")
            console.print(f"  Projected Completions: {proj_completions:.1f}")
            console.print(f"  Projected Passing Yards: {proj_pass_yards:.0f}")
            console.print(f"\n[bold]Efficiency:[/bold]")
            console.print(f"  Projected Completion Rate: {proj['proj_completion_rate']:.1%}")
            console.print(f"  Projected EPA/Attempt: {proj['proj_epa_per_attempt']:.3f}")
            console.print(f"\n[bold]Total:[/bold]")
            console.print(f"  Projected EPA Contribution: {proj['proj_epa_total']:.2f}")
        else:
            # Skill position projection
            if isinstance(role_model, BaselineRoleModel):
                proj = project_player(
                    player_id, game_id, role_model, efficiency_model,
                    features, player_game, team_game, players, games
                )
            else:
                # ML model - use predict methods directly
                proj_targets = role_model.predict_targets(
                    player_features, players, team_game, games
                ).iloc[0]
                proj_carries = role_model.predict_carries(
                    player_features, players, team_game, games
                ).iloc[0]
                
                proj_epa_per_target = efficiency_model.predict_epa_per_target(
                    player_features, player_game, players
                ).iloc[0]
                proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
                    player_features, player_game, players
                ).iloc[0]
                
                proj = {
                    "proj_targets": proj_targets,
                    "proj_carries": proj_carries,
                    "proj_epa_per_target": proj_epa_per_target,
                    "proj_epa_per_rush": proj_epa_per_rush,
                    "proj_epa_total": proj_targets * proj_epa_per_target + proj_carries * proj_epa_per_rush
                }
            
            # Calculate projected yards
            player_historical = player_game[player_game["player_id"] == player_id]
            if len(player_historical) > 0:
                hist_rec_yards = player_historical["receiving_yards"].sum()
                hist_targets = player_historical["targets"].sum()
                hist_rush_yards = player_historical["rushing_yards"].sum()
                hist_carries = player_historical["carries"].sum()
                
                proj_rec_yards = (proj['proj_targets'] * (hist_rec_yards / hist_targets)) if hist_targets > 0 else 0
                proj_rush_yards = (proj['proj_carries'] * (hist_rush_yards / hist_carries)) if hist_carries > 0 else 0
            else:
                proj_rec_yards = 0
                proj_rush_yards = 0
            
            game_info = games[games["game_id"] == game_id].iloc[0]
            
            console.print(f"\n[bold]Projection for {player_name}[/bold]")
            console.print(f"Model: {model}")
            console.print(f"Game: {game_info['home_team']} vs {game_info['away_team']}")
            console.print(f"Date: {game_info['date'].strftime('%Y-%m-%d')}")
            console.print(f"\n[bold]Role:[/bold]")
            console.print(f"  Projected Targets: {proj['proj_targets']:.1f}")
            console.print(f"  Projected Receiving Yards: {proj_rec_yards:.0f}")
            console.print(f"  Projected Carries: {proj['proj_carries']:.1f}")
            console.print(f"  Projected Rushing Yards: {proj_rush_yards:.0f}")
            console.print(f"\n[bold]Efficiency:[/bold]")
            console.print(f"  Projected EPA/Target: {proj['proj_epa_per_target']:.3f}")
            console.print(f"  Projected EPA/Rush: {proj['proj_epa_per_rush']:.3f}")
        console.print(f"\n[bold]Total:[/bold]")
        console.print(f"  Projected EPA Contribution: {proj['proj_epa_total']:.2f}")
        
        # Show confidence indicators
        console.print(f"\n[dim]Note:[/dim] Projections are based on historical data and may vary based on:")
        console.print(f"  - Opponent strength")
        console.print(f"  - Game script (score, time remaining)")
        console.print(f"  - Injuries and lineup changes")
        
    except KeyError as e:
        console.print(f"[red]Error:[/red] Missing column: {e}")
        console.print("\n[yellow]This might indicate:[/yellow]")
        console.print("  1. Tables need to be rebuilt - run 'nflproj build'")
        console.print("  2. Features need to be generated - run 'nflproj features'")
        console.print("  3. Data structure mismatch")
        import traceback
        console.print(f"\n[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Ensure data is ingested: 'nflproj ingest --seasons <year>'")
        console.print("  2. Ensure tables are built: 'nflproj build'")
        console.print("  3. Ensure features are generated: 'nflproj features'")
        console.print("  4. Check player name spelling: 'nflproj search-players <name>'")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        console.print(f"\n[dim]Full error:[/dim]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def compare_players(
    players: str = typer.Argument(..., help="Comma-separated player names (e.g., 'Player1,Player2,Player3')"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season (includes playoffs through February)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Compare multiple players side-by-side over a date range or season.
    
    Examples:
        nflproj compare-players "Travis Kelce,Mark Andrews" --season 2023
        nflproj compare-players "Travis Kelce,Mark Andrews" --from 2023-09-01 --to 2023-12-31
        nflproj compare-players "Justin Jefferson,Tyreek Hill" --season 2023
    """
    console.print(f"[bold green]Comparing players...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players_df = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first.")
        raise typer.Exit(1)
    
    # Parse dates - if season is provided, use that instead
    if season:
        # NFL seasons run from September to February (next year)
        from_dt = pd.to_datetime(f"{season}-09-01")
        to_dt = pd.to_datetime(f"{season + 1}-02-28")
        console.print(f"[dim]Using season {season} date range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}[/dim]")
    elif from_date and to_date:
        try:
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
        except ValueError:
            console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --season or both --from and --to")
        raise typer.Exit(1)
    
    # Parse player names
    player_names = [p.strip() for p in players.split(",")]
    
    # Filter games by date first
    games_in_range = games[
        (games["date"] >= from_dt) &
        (games["date"] <= to_dt)
    ]
    
    if len(games_in_range) == 0:
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found in date range {date_range_str}[/yellow]")
        console.print(f"Available date range: {games['date'].min().strftime('%Y-%m-%d')} to {games['date'].max().strftime('%Y-%m-%d')}")
        console.print(f"Available seasons: {', '.join(map(str, sorted(games['season'].unique())))}")
        raise typer.Exit(0)
    
    # Find players - use same logic as compare-with-without for disambiguation
    player_data = []
    game_ids_for_matching = games_in_range["game_id"].tolist()
    player_game_in_range = player_game[player_game["game_id"].isin(game_ids_for_matching)]
    
    def select_best_match_compare(matches, player_game_data, game_ids_list):
        """Select the best matching player from multiple candidates."""
        if len(matches) == 1:
            return matches.iloc[0]
        
        # Prefer offensive skill positions
        offensive_positions = ["QB", "RB", "WR", "TE"]
        for pos in offensive_positions:
            pos_matches = matches[matches["position"] == pos]
            if len(pos_matches) > 0:
                # Check which one has games in the date range
                for _, match in pos_matches.iterrows():
                    player_id = match["player_id"]
                    has_games = len(player_game_data[
                        (player_game_data["player_id"] == player_id) &
                        (player_game_data["game_id"].isin(game_ids_list))
                    ]) > 0
                    if has_games:
                        return match
                # If none have games, return first match of this position
                return pos_matches.iloc[0]
        
        # If no offensive positions, check which has games
        for _, match in matches.iterrows():
            player_id = match["player_id"]
            has_games = len(player_game_data[
                (player_game_data["player_id"] == player_id) &
                (player_game_data["game_id"].isin(game_ids_list))
            ]) > 0
            if has_games:
                return match
        
        # Fall back to first match
        return matches.iloc[0]
    
    for player_name in player_names:
        matches = players_df[players_df["name"].str.contains(player_name, case=False, na=False)]
        if len(matches) == 0:
            console.print(f"[yellow]Warning:[/yellow] Player '{player_name}' not found, skipping")
            continue
        
        if len(matches) > 1:
            best_match = select_best_match_compare(matches, player_game_in_range, game_ids_for_matching)
            console.print(f"[yellow]Warning:[/yellow] Multiple players found matching '{player_name}', using {best_match['name']} ({best_match['position']})")
            other_matches = [f"{r['name']} ({r['position']})" for _, r in matches.iterrows() if r['player_id'] != best_match['player_id']]
            if other_matches:
                console.print(f"  Other matches: {', '.join(other_matches)}")
            player_data.append(best_match)
        else:
            player_data.append(matches.iloc[0])
    
    if len(player_data) == 0:
        console.print("[red]Error:[/red] No valid players found")
        raise typer.Exit(1)
    
    # games_in_range and game_ids are already defined above
    game_ids = games_in_range["game_id"].tolist()
    
    # Build comparison table
    date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
    table = Table(title=f"Player Comparison ({date_range_str})")
    table.add_column("Player", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Games", justify="right")
    table.add_column("Targets", justify="right")
    table.add_column("Rec Yds", justify="right")
    table.add_column("Carries", justify="right")
    table.add_column("Rush Yds", justify="right")
    table.add_column("Total EPA", justify="right", style="green")
    table.add_column("EPA/Target", justify="right")
    table.add_column("EPA/Rush", justify="right")
    
    for player_info in player_data:
        player_id = player_info["player_id"]
        player_name = player_info["name"]
        position = player_info["position"]
        
        # Get player stats
        player_stats = player_game[
            (player_game["player_id"] == player_id) &
            (player_game["game_id"].isin(game_ids))
        ]
        
        if len(player_stats) == 0:
            table.add_row(player_name, position, "0", "0", "0", "0", "0", "0.00", "-", "-")
            continue
        
        total_targets = player_stats["targets"].sum()
        total_rec_yards = player_stats["receiving_yards"].sum()
        total_carries = player_stats["carries"].sum()
        total_rush_yards = player_stats["rushing_yards"].sum()
        total_epa = player_stats["epa_total"].sum()
        
        avg_epa_target = (player_stats["targets"] * player_stats["epa_per_target"]).sum() / total_targets if total_targets > 0 else 0
        avg_epa_rush = (player_stats["carries"] * player_stats["epa_per_rush"]).sum() / total_carries if total_carries > 0 else 0
        
        table.add_row(
            player_name,
            position,
            str(len(player_stats)),
            str(int(total_targets)),
            str(int(total_rec_yards)),
            str(int(total_carries)),
            str(int(total_rush_yards)),
            f"{total_epa:.2f}",
            f"{avg_epa_target:.3f}" if total_targets > 0 else "-",
            f"{avg_epa_rush:.3f}" if total_carries > 0 else "-"
        )
    
    # Add total row
    if len(player_data) > 0:
        all_player_stats = player_game[
            (player_game["player_id"].isin([p["player_id"] for p in player_data])) &
            (player_game["game_id"].isin(game_ids))
        ]
        if len(all_player_stats) > 0:
            total_games = len(all_player_stats.groupby("game_id"))
            total_targets_all = all_player_stats["targets"].sum()
            total_rec_yards_all = all_player_stats["receiving_yards"].sum()
            total_carries_all = all_player_stats["carries"].sum()
            total_rush_yards_all = all_player_stats["rushing_yards"].sum()
            total_epa_all = all_player_stats["epa_total"].sum()
            avg_epa_target_all = (all_player_stats["targets"] * all_player_stats["epa_per_target"]).sum() / total_targets_all if total_targets_all > 0 else 0
            avg_epa_rush_all = (all_player_stats["carries"] * all_player_stats["epa_per_rush"]).sum() / total_carries_all if total_carries_all > 0 else 0
            
            table.add_row(
                "[bold]TOTAL[/bold]",
                "-",
                f"[bold]{total_games}[/bold]",
                f"[bold]{int(total_targets_all)}[/bold]",
                f"[bold]{int(total_rec_yards_all)}[/bold]",
                f"[bold]{int(total_carries_all)}[/bold]",
                f"[bold]{int(total_rush_yards_all)}[/bold]",
                f"[bold]{total_epa_all:.2f}[/bold]",
                f"[bold]{avg_epa_target_all:.3f}[/bold]" if total_targets_all > 0 else "-",
                f"[bold]{avg_epa_rush_all:.3f}[/bold]" if total_carries_all > 0 else "-"
            )
    
    console.print(table)


@app.command()
def batch_project(
    players: str = typer.Argument(..., help="Comma-separated player names"),
    week: Optional[int] = typer.Option(None, "--week", help="Week number"),
    season: Optional[int] = typer.Option(None, "--season", help="Season year"),
    date: Optional[str] = typer.Option(None, "--date", help="Game date (YYYY-MM-DD)"),
    model: str = typer.Option("baseline", "--model", help="Model type"),
    output: Optional[str] = typer.Option(None, "--output", help="Output file path (CSV or JSON)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Project multiple players at once for a game/week.
    
    Examples:
        nflproj batch-project "Travis Kelce,Mark Andrews" --week 10 --season 2023
        nflproj batch-project "Player1,Player2,Player3" --week 10 --season 2023 --model gbm_pos --output projections.csv
    """
    console.print(f"[bold green]Batch projecting players...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        team_game = pd.read_parquet(data_path / "team_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players_df = pd.read_parquet(data_path / "players.parquet")
        features = pd.read_parquet(data_path / "features_player_game_offense.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' and 'features' commands first.")
        raise typer.Exit(1)
    
    # Initialize models (once for all players)
    from .models import MLRoleModel
    
    if model == "baseline":
        role_model = BaselineRoleModel()
        role_model.fit(player_game, team_game, players_df)
    else:
        position_specific = model.endswith("_pos") or model == "gbm_pos" or model == "ridge_pos"
        model_type_clean = model.replace("_pos", "")
        
        ml_kwargs = {}
        if model_type_clean == "ridge":
            ml_kwargs = {"alpha": 0.5}
        elif model_type_clean == "gbm":
            ml_kwargs = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_split": 20,
                "min_samples_leaf": 5
            }
        
        role_model = MLRoleModel(model_type=model_type_clean, position_specific=position_specific)
        role_model.kwargs = ml_kwargs
        role_model.fit(features, player_game, players_df, team_game, games)
    
    efficiency_model = BaselineEfficiencyModel()
    efficiency_model.fit(player_game, players_df)
    
    # Parse player names
    player_names = [p.strip() for p in players.split(",")]
    
    # Find games
    if date:
        game_date = pd.to_datetime(date)
        week_games = games[games["date"] == game_date]
    elif week and season:
        week_games = games[(games["week"] == week) & (games["season"] == season)]
    else:
        console.print("[red]Error:[/red] Must provide either --date or --week and --season")
        raise typer.Exit(1)
    
    if len(week_games) == 0:
        console.print("[red]Error:[/red] No games found for specified criteria")
        raise typer.Exit(1)
    
    # Project each player
    projections = []
    for player_name in player_names:
        matches = players_df[players_df["name"].str.contains(player_name, case=False, na=False)]
        if len(matches) == 0:
            console.print(f"[yellow]Warning:[/yellow] Player '{player_name}' not found, skipping")
            continue
        
        player_id = matches.iloc[0]["player_id"]
        
        # Find game where player played
        player_games = player_game[
            (player_game["player_id"] == player_id) &
            (player_game["game_id"].isin(week_games["game_id"]))
        ]
        
        if len(player_games) == 0:
            console.print(f"[yellow]Warning:[/yellow] {player_name} did not play in specified game/week, skipping")
            continue
        
        game_id = player_games.iloc[0]["game_id"]
        game_info = games[games["game_id"] == game_id].iloc[0]
        
        # Get features
        player_features = features[
            (features["player_id"] == player_id) &
            (features["game_id"] == game_id)
        ]
        
        if len(player_features) == 0:
            console.print(f"[yellow]Warning:[/yellow] No features for {player_name}, skipping")
            continue
        
        # Make projection
        try:
            if isinstance(role_model, BaselineRoleModel):
                proj_targets = role_model.predict_targets(
                    player_features, player_game, team_game, players_df, games
                ).iloc[0]
                proj_carries = role_model.predict_carries(
                    player_features, player_game, team_game, players_df, games
                ).iloc[0]
            else:
                proj_targets = role_model.predict_targets(
                    player_features, players_df, team_game, games
                ).iloc[0]
                proj_carries = role_model.predict_carries(
                    player_features, players_df, team_game, games
                ).iloc[0]
            
            proj_epa_per_target = efficiency_model.predict_epa_per_target(
                player_features, player_game, players_df
            ).iloc[0]
            proj_epa_per_rush = efficiency_model.predict_epa_per_rush(
                player_features, player_game, players_df
            ).iloc[0]
            
            proj_epa_total = proj_targets * proj_epa_per_target + proj_carries * proj_epa_per_rush
            
            # Project yards based on historical yards per target/carry
            player_historical = player_game[player_game["player_id"] == player_id]
            if len(player_historical) > 0:
                hist_rec_yards = player_historical["receiving_yards"].sum()
                hist_targets = player_historical["targets"].sum()
                hist_rush_yards = player_historical["rushing_yards"].sum()
                hist_carries = player_historical["carries"].sum()
                
                proj_rec_yards = (proj_targets * (hist_rec_yards / hist_targets)) if hist_targets > 0 else 0
                proj_rush_yards = (proj_carries * (hist_rush_yards / hist_carries)) if hist_carries > 0 else 0
            else:
                proj_rec_yards = 0
                proj_rush_yards = 0
            
            projections.append({
                "player": player_name,
                "position": matches.iloc[0]["position"],
                "game": f"{game_info['home_team']} vs {game_info['away_team']}",
                "date": game_info["date"].strftime("%Y-%m-%d"),
                "week": game_info["week"],
                "season": game_info["season"],
                "proj_targets": round(proj_targets, 1),
                "proj_rec_yards": round(proj_rec_yards, 0),
                "proj_carries": round(proj_carries, 1),
                "proj_rush_yards": round(proj_rush_yards, 0),
                "proj_epa_per_target": round(proj_epa_per_target, 3),
                "proj_epa_per_rush": round(proj_epa_per_rush, 3),
                "proj_epa_total": round(proj_epa_total, 2)
            })
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Error projecting {player_name}: {e}")
            continue
    
    if len(projections) == 0:
        console.print("[red]Error:[/red] No valid projections generated")
        raise typer.Exit(1)
    
    # Display results
    table = Table(title=f"Batch Projections (Model: {model})")
    table.add_column("Player", style="cyan")
    table.add_column("Position", style="green")
    table.add_column("Game", style="magenta")
    table.add_column("Targets", justify="right")
    table.add_column("Rec Yds", justify="right")
    table.add_column("Carries", justify="right")
    table.add_column("Rush Yds", justify="right")
    table.add_column("EPA Total", justify="right", style="green")
    
    for proj in projections:
        table.add_row(
            proj["player"],
            proj["position"],
            proj["game"],
            str(proj["proj_targets"]),
            str(int(proj["proj_rec_yards"])),
            str(proj["proj_carries"]),
            str(int(proj["proj_rush_yards"])),
            str(proj["proj_epa_total"])
        )
    
    console.print(table)
    
    # Export if requested
    if output:
        output_path = Path(output)
        proj_df = pd.DataFrame(projections)
        
        if output_path.suffix.lower() == ".json":
            proj_df.to_json(output_path, orient="records", indent=2)
            console.print(f"\n[green]✓[/green] Exported to {output_path}")
        elif output_path.suffix.lower() == ".csv":
            proj_df.to_csv(output_path, index=False)
            console.print(f"\n[green]✓[/green] Exported to {output_path}")
        else:
            console.print(f"[yellow]Warning:[/yellow] Unknown file extension, defaulting to CSV")
            proj_df.to_csv(output_path, index=False)
            console.print(f"\n[green]✓[/green] Exported to {output_path}")


@app.command()
def team_stats(
    team: str = typer.Argument(..., help="Team abbreviation (e.g., KC, SF)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Show team-level offensive statistics.
    
    Examples:
        nflproj team-stats KC --season 2023
        nflproj team-stats SF --from 2023-09-01 --to 2023-12-31
    """
    console.print(f"[bold green]Team stats for {team}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        team_game = pd.read_parquet(data_path / "team_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first.")
        raise typer.Exit(1)
    
    # Filter team games
    team_stats = team_game[team_game["team"] == team.upper()]
    
    if len(team_stats) == 0:
        console.print(f"[red]Error:[/red] No data found for team {team}")
        raise typer.Exit(1)
    
    # Merge with games for filtering
    team_stats = pd.merge(team_stats, games, on="game_id", how="left")
    
    # Determine opponent (home or away)
    team_stats["opponent"] = team_stats.apply(
        lambda x: x["away_team"] if x["team"] == x["home_team"] else x["home_team"],
        axis=1
    )
    
    # Apply filters
    if season:
        team_stats = team_stats[team_stats["season"] == season]
    
    if from_date:
        from_dt = pd.to_datetime(from_date)
        team_stats = team_stats[team_stats["date"] >= from_dt]
    
    if to_date:
        to_dt = pd.to_datetime(to_date)
        team_stats = team_stats[team_stats["date"] <= to_dt]
    
    if len(team_stats) == 0:
        console.print(f"[yellow]No games found for {team} with specified filters[/yellow]")
        raise typer.Exit(0)
    
    # Display summary
    console.print(f"\n[bold]Team: {team.upper()}[/bold]")
    console.print(f"Games: {len(team_stats)}")
    console.print(f"\n[bold]Offensive Stats:[/bold]")
    console.print(f"  Avg Plays/Game: {team_stats['team_plays'].mean():.1f}")
    console.print(f"  Avg Dropbacks/Game: {team_stats['dropbacks'].mean():.1f}")
    console.print(f"  Avg Rushes/Game: {team_stats['rushes'].mean():.1f}")
    console.print(f"  Pass Rate: {team_stats['pass_rate'].mean():.1%}")
    console.print(f"  Avg EPA/Play: {team_stats['team_epa_per_play'].mean():.3f}")
    console.print(f"  Total EPA: {team_stats['team_epa_per_play'].sum():.2f}")
    
    # Show recent games
    table = Table(title=f"Recent Games - {team.upper()}")
    table.add_column("Date", style="cyan")
    table.add_column("Week", style="cyan")
    table.add_column("Opponent", style="magenta")
    table.add_column("Plays", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("EPA/Play", justify="right", style="green")
    
    for _, row in team_stats.sort_values("date", ascending=False).head(10).iterrows():
        table.add_row(
            row["date"].strftime("%Y-%m-%d"),
            str(row["week"]),
            row["opponent"],
            str(int(row["team_plays"])),
            f"{row['pass_rate']:.1%}",
            f"{row['team_epa_per_play']:.3f}"
        )
    
    console.print(table)


@app.command()
def team_defense(
    team: str = typer.Argument(..., help="Team abbreviation (e.g., LA, SF, KC)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Show team-level defensive statistics.
    
    Examples:
        nflproj team-defense LA --season 2023
        nflproj team-defense SF --from 2023-09-01 --to 2023-12-31
        nflproj team-defense KC --season 2024
    """
    console.print(f"[bold green]Team defense stats for {team}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        team_game_defense = pd.read_parquet(data_path / "team_game_defense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first to create defense tables.")
        raise typer.Exit(1)
    
    # Filter team games
    team_def = team_game_defense[team_game_defense["team"] == team.upper()]
    
    if len(team_def) == 0:
        console.print(f"[red]Error:[/red] No defense data found for team {team}")
        console.print("\n[yellow]Note:[/yellow] Defense tables are built when you run 'nflproj build'")
        raise typer.Exit(1)
    
    # Merge with games for filtering
    team_def = pd.merge(team_def, games, on="game_id", how="left")
    
    # Determine opponent (home or away)
    team_def["opponent"] = team_def.apply(
        lambda x: x["away_team"] if x["team"] == x["home_team"] else x["home_team"],
        axis=1
    )
    
    # Apply filters
    if season:
        team_def = team_def[team_def["season"] == season]
    
    if from_date:
        from_dt = pd.to_datetime(from_date)
        team_def = team_def[team_def["date"] >= from_dt]
    
    if to_date:
        to_dt = pd.to_datetime(to_date)
        team_def = team_def[team_def["date"] <= to_dt]
    
    if len(team_def) == 0:
        console.print(f"[yellow]No games found for {team} with specified filters[/yellow]")
        raise typer.Exit(0)
    
    # Display summary
    console.print(f"\n[bold]Team Defense: {team.upper()}[/bold]")
    console.print(f"Games: {len(team_def)}")
    console.print(f"\n[bold]Defensive Stats:[/bold]")
    console.print(f"  Avg Plays Faced/Game: {team_def['plays_faced'].mean():.1f}")
    console.print(f"  Avg Dropbacks Faced/Game: {team_def['dropbacks_faced'].mean():.1f}")
    console.print(f"  Avg Rushes Faced/Game: {team_def['rushes_faced'].mean():.1f}")
    console.print(f"  Avg EPA Allowed/Play: {team_def['epa_allowed_per_play'].mean():.3f}")
    console.print(f"  Total EPA Allowed: {team_def['epa_allowed'].sum():.2f}")
    console.print(f"  Total Sacks: {team_def['sacks'].sum():.0f}")
    console.print(f"  Avg Sacks/Game: {team_def['sacks'].mean():.2f}")
    console.print(f"  Total Interceptions: {team_def['interceptions'].sum():.0f}")
    console.print(f"  Avg INTs/Game: {team_def['interceptions'].mean():.2f}")
    console.print(f"  Total Fumbles Forced: {team_def['fumbles_forced'].sum():.0f}")
    console.print(f"  Avg Fumbles Forced/Game: {team_def['fumbles_forced'].mean():.2f}")
    
    # Show recent games
    table = Table(title=f"Recent Games - {team.upper()} Defense")
    table.add_column("Date", style="cyan")
    table.add_column("Week", style="cyan")
    table.add_column("Opponent", style="magenta")
    table.add_column("Plays Faced", justify="right")
    table.add_column("EPA/Play", justify="right", style="red")
    table.add_column("Sacks", justify="right")
    table.add_column("INTs", justify="right")
    table.add_column("FF", justify="right")
    
    for _, row in team_def.sort_values("date", ascending=False).head(10).iterrows():
        table.add_row(
            row["date"].strftime("%Y-%m-%d"),
            str(row["week"]),
            row["opponent"],
            str(int(row["plays_faced"])),
            f"{row['epa_allowed_per_play']:.3f}",
            str(int(row["sacks"])),
            str(int(row["interceptions"])),
            str(int(row.get("fumbles_forced", 0)))
        )
    
    console.print(table)
    
    # Show best/worst games
    console.print(f"\n[bold]Best Defensive Games (Highest EPA Allowed - Best Defense):[/bold]")
    # For defense, positive EPA is good (it's negated offensive EPA), so highest = best
    best_games = team_def.nlargest(3, "epa_allowed_per_play")
    for _, row in best_games.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if 'date' in row and pd.notna(row['date']) else 'N/A'
        fumbles_forced = int(row.get('fumbles_forced', 0))
        console.print(f"  {date_str} - Week {row['week']} vs {row['opponent']}: {row['epa_allowed_per_play']:.3f} EPA/play, {int(row['sacks'])} sacks, {int(row['interceptions'])} INTs, {fumbles_forced} FF")
    
    console.print(f"\n[bold]Worst Defensive Games (Lowest EPA Allowed - Worst Defense):[/bold]")
    # For defense, lowest EPA = worst defense
    worst_games = team_def.nsmallest(3, "epa_allowed_per_play")
    for _, row in worst_games.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d') if 'date' in row and pd.notna(row['date']) else 'N/A'
        fumbles_forced = int(row.get('fumbles_forced', 0))
        console.print(f"  {date_str} - Week {row['week']} vs {row['opponent']}: {row['epa_allowed_per_play']:.3f} EPA/play, {int(row['sacks'])} sacks, {int(row['interceptions'])} INTs, {fumbles_forced} FF")


@app.command()
def compare_with_without(
    player1: str = typer.Argument(..., help="Primary player to analyze (e.g., 'Patrick Mahomes')"),
    player2: str = typer.Argument(..., help="Secondary player to compare with/without (e.g., 'Tyreek Hill')"),
    player1_position: Optional[str] = typer.Option(None, "--player1-position", help="Position to disambiguate player1 (e.g., 'QB', 'RB', 'WR')"),
    player2_position: Optional[str] = typer.Option(None, "--player2-position", help="Position to disambiguate player2 (e.g., 'QB', 'RB', 'WR')"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season (includes playoffs through February)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Compare a player's performance with and without another player.
    
    Examples:
        nflproj compare-with-without "Patrick Mahomes" "Tyreek Hill" --season 2023
        nflproj compare-with-without "Josh Allen" "Stefon Diggs" --from 2023-09-01 --to 2023-12-31
    """
    console.print(f"[bold green]Comparing {player1} with/without {player2}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first.")
        raise typer.Exit(1)
    
    # Parse dates
    if season:
        from_dt = pd.to_datetime(f"{season}-09-01")
        to_dt = pd.to_datetime(f"{season + 1}-02-28")
        console.print(f"[dim]Using season {season} date range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}[/dim]")
    elif from_date and to_date:
        try:
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
        except ValueError:
            console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --season or both --from and --to")
        raise typer.Exit(1)
    
    # Filter games by date first (needed for player matching)
    games_in_range = games[
        (games["date"] >= from_dt) &
        (games["date"] <= to_dt)
    ]
    
    if len(games_in_range) == 0:
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found in date range {date_range_str}[/yellow]")
        raise typer.Exit(0)
    
    game_ids = games_in_range["game_id"].tolist()
    
    # Get player_game data for the date range (needed for matching)
    player_game_in_range = player_game[player_game["game_id"].isin(game_ids)]
    
    # Find players
    player1_matches = players[players["name"].str.contains(player1, case=False, na=False)]
    player2_matches = players[players["name"].str.contains(player2, case=False, na=False)]
    
    # Filter by position if provided
    if player1_position:
        player1_matches = player1_matches[player1_matches["position"] == player1_position.upper()]
    if player2_position:
        player2_matches = player2_matches[player2_matches["position"] == player2_position.upper()]
    
    if len(player1_matches) == 0:
        console.print(f"[red]Error:[/red] Player '{player1}' not found")
        if player1_position:
            console.print(f"  (with position filter: {player1_position})")
        console.print("\n[yellow]Tip:[/yellow] Use 'nflproj search-players <name>' to find the correct spelling")
        console.print("  Or use --player1-position to disambiguate (e.g., --player1-position QB)")
        raise typer.Exit(1)
    if len(player2_matches) == 0:
        console.print(f"[red]Error:[/red] Player '{player2}' not found")
        if player2_position:
            console.print(f"  (with position filter: {player2_position})")
        console.print("\n[yellow]Tip:[/yellow] Use 'nflproj search-players <name>' to find the correct spelling")
        console.print("  Or use --player2-position to disambiguate (e.g., --player2-position WR)")
        raise typer.Exit(1)
    
    # Helper function to select best match when multiple players have same name
    def select_best_match(matches, player_game_data, game_ids_list):
        """Select the best matching player from multiple candidates."""
        if len(matches) == 1:
            return matches.iloc[0]
        
        # Prefer offensive skill positions
        offensive_positions = ["QB", "RB", "WR", "TE"]
        for pos in offensive_positions:
            pos_matches = matches[matches["position"] == pos]
            if len(pos_matches) > 0:
                # Check which one has games in the date range
                for _, match in pos_matches.iterrows():
                    player_id = match["player_id"]
                    has_games = len(player_game_data[
                        (player_game_data["player_id"] == player_id) &
                        (player_game_data["game_id"].isin(game_ids_list))
                    ]) > 0
                    if has_games:
                        return match
                # If none have games, return first match of this position
                return pos_matches.iloc[0]
        
        # If no offensive positions, check which has games
        for _, match in matches.iterrows():
            player_id = match["player_id"]
            has_games = len(player_game_data[
                (player_game_data["player_id"] == player_id) &
                (player_game_data["game_id"].isin(game_ids_list))
            ]) > 0
            if has_games:
                return match
        
        # Fall back to first match
        return matches.iloc[0]
    
    player1_match = select_best_match(player1_matches, player_game_in_range, game_ids)
    player2_match = select_best_match(player2_matches, player_game_in_range, game_ids)
    
    player1_id = player1_match["player_id"]
    player1_name = player1_match["name"]
    player1_pos = player1_match["position"]
    player2_id = player2_match["player_id"]
    player2_name = player2_match["name"]
    player2_pos = player2_match["position"]
    
    # Warn if multiple matches and we had to choose
    if len(player1_matches) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple players found matching '{player1}', using {player1_name} ({player1_pos})")
        other_matches = [f"{r['name']} ({r['position']})" for _, r in player1_matches.iterrows() if r['player_id'] != player1_id]
        if other_matches:
            console.print(f"  Other matches: {', '.join(other_matches)}")
    if len(player2_matches) > 1:
        console.print(f"[yellow]Warning:[/yellow] Multiple players found matching '{player2}', using {player2_name} ({player2_pos})")
        other_matches = [f"{r['name']} ({r['position']})" for _, r in player2_matches.iterrows() if r['player_id'] != player2_id]
        if other_matches:
            console.print(f"  Other matches: {', '.join(other_matches)}")
    
    # Note: player1_id and player2_id are already selected above with best match logic
    
    # Get Player 1's games in date range
    player1_games = player_game[
        (player_game["player_id"] == player1_id) &
        (player_game["game_id"].isin(game_ids))
    ].copy()
    
    if len(player1_games) == 0:
        console.print(f"[yellow]No games found for {player1_name} in date range[/yellow]")
        raise typer.Exit(0)
    
    # Merge with games to get team info
    player1_games = pd.merge(
        player1_games,
        games[["game_id", "date", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    
    # Get Player 2's games in same date range
    # Only count games where player2 actually played (has targets, carries, or pass attempts)
    player2_all_games = player_game[
        (player_game["player_id"] == player2_id) &
        (player_game["game_id"].isin(game_ids))
    ].copy()
    
    # Filter to games where player2 actually had opportunities
    # Check if columns exist and filter accordingly
    has_targets = "targets" in player2_all_games.columns
    has_carries = "carries" in player2_all_games.columns
    has_pass_attempts = "pass_attempts" in player2_all_games.columns
    
    if has_targets or has_carries or has_pass_attempts:
        conditions = []
        if has_targets:
            conditions.append(player2_all_games["targets"] > 0)
        if has_carries:
            conditions.append(player2_all_games["carries"] > 0)
        if has_pass_attempts:
            conditions.append(player2_all_games["pass_attempts"] > 0)
        
        if conditions:
            # Combine conditions with OR
            played_mask = conditions[0]
            for cond in conditions[1:]:
                played_mask = played_mask | cond
            player2_played = player2_all_games[played_mask]
        else:
            player2_played = player2_all_games
    else:
        # If none of the columns exist, assume all games count
        player2_played = player2_all_games
    
    player2_games = player2_played[["game_id", "team"]].copy()
    player2_games = player2_games.rename(columns={"team": "player2_team"})
    
    # Determine which Player 1 games had Player 2 playing (same team, same game)
    player1_games = pd.merge(
        player1_games,
        player2_games,
        on="game_id",
        how="left",
        indicator=True
    )
    
    # Games WITH Player 2: Player 2 played in same game AND same team
    with_player2 = player1_games[
        (player1_games["_merge"] == "both") &
        (player1_games["team"] == player1_games["player2_team"])
    ].copy()
    
    # Games WITHOUT Player 2: Player 2 didn't play in same game OR different team
    # Also include games where player2 didn't have any stats (didn't play)
    without_player2 = player1_games[
        (player1_games["_merge"] == "left_only") |
        ((player1_games["_merge"] == "both") & (player1_games["team"] != player1_games["player2_team"]))
    ].copy()
    
    if len(with_player2) == 0:
        console.print(f"[yellow]No games found where {player1_name} and {player2_name} played together[/yellow]")
        console.print(f"Found {len(without_player2)} games where {player1_name} played without {player2_name}")
        raise typer.Exit(0)
    
    if len(without_player2) == 0:
        console.print(f"[yellow]No games found where {player1_name} played without {player2_name}[/yellow]")
        console.print(f"Found {len(with_player2)} games where they played together")
        raise typer.Exit(0)
    
    # Aggregate stats
    def aggregate_stats(df, label):
        stats = {
            "label": label,
            "games": len(df),
        }
        
        # Skill position stats
        if "targets" in df.columns:
            stats["targets"] = df["targets"].sum()
            stats["targets_per_game"] = df["targets"].mean()
            stats["receiving_yards"] = df["receiving_yards"].sum()
            stats["receiving_yards_per_game"] = df["receiving_yards"].mean()
            stats["epa_receiving"] = df["epa_receiving"].sum()
            stats["epa_per_target"] = df["epa_receiving"].sum() / df["targets"].sum() if df["targets"].sum() > 0 else 0
        
        if "carries" in df.columns:
            stats["carries"] = df["carries"].sum()
            stats["carries_per_game"] = df["carries"].mean()
            stats["rushing_yards"] = df["rushing_yards"].sum()
            stats["rushing_yards_per_game"] = df["rushing_yards"].mean()
            stats["epa_rushing"] = df["epa_rushing"].sum()
            stats["epa_per_rush"] = df["epa_rushing"].sum() / df["carries"].sum() if df["carries"].sum() > 0 else 0
        
        # QB stats
        if "pass_attempts" in df.columns:
            stats["pass_attempts"] = df["pass_attempts"].sum()
            stats["pass_attempts_per_game"] = df["pass_attempts"].mean()
            stats["completions"] = df["completions"].sum()
            stats["completion_rate"] = df["completions"].sum() / df["pass_attempts"].sum() if df["pass_attempts"].sum() > 0 else 0
            stats["passing_yards"] = df["passing_yards"].sum()
            stats["passing_yards_per_game"] = df["passing_yards"].mean()
            stats["epa_passing"] = df["epa_passing"].sum()
            stats["epa_per_attempt"] = df["epa_passing"].sum() / df["pass_attempts"].sum() if df["pass_attempts"].sum() > 0 else 0
        
        # Total EPA
        stats["epa_total"] = df["epa_total"].sum()
        stats["epa_total_per_game"] = df["epa_total"].mean()
        
        return stats
    
    with_stats = aggregate_stats(with_player2, f"With {player2_name}")
    without_stats = aggregate_stats(without_player2, f"Without {player2_name}")
    
    # Determine which stats to show based on player position
    player1_pos = players[players["player_id"] == player1_id].iloc[0]["position"]
    is_qb = player1_pos == "QB"
    
    # Create comparison table
    from rich.table import Table
    table = Table(title=f"{player1_name} Performance: With vs Without {player2_name}")
    
    if is_qb:
        table.add_column("Stat", style="cyan")
        table.add_column(f"With {player2_name}", style="green")
        table.add_column(f"Without {player2_name}", style="yellow")
        table.add_column("Difference", style="magenta")
        
        table.add_row("Games", str(with_stats["games"]), str(without_stats["games"]), 
                     f"{without_stats['games'] - with_stats['games']:+d}")
        table.add_row("Pass Attempts/Game", f"{with_stats['pass_attempts_per_game']:.1f}",
                     f"{without_stats['pass_attempts_per_game']:.1f}",
                     f"{without_stats['pass_attempts_per_game'] - with_stats['pass_attempts_per_game']:+.1f}")
        table.add_row("Completion Rate", f"{with_stats['completion_rate']:.1%}",
                     f"{without_stats['completion_rate']:.1%}",
                     f"{(without_stats['completion_rate'] - with_stats['completion_rate']):+.1%}")
        table.add_row("Passing Yards/Game", f"{with_stats['passing_yards_per_game']:.1f}",
                     f"{without_stats['passing_yards_per_game']:.1f}",
                     f"{without_stats['passing_yards_per_game'] - with_stats['passing_yards_per_game']:+.1f}")
        table.add_row("EPA/Attempt", f"{with_stats['epa_per_attempt']:.3f}",
                     f"{without_stats['epa_per_attempt']:.3f}",
                     f"{without_stats['epa_per_attempt'] - with_stats['epa_per_attempt']:+.3f}")
        table.add_row("EPA Total/Game", f"{with_stats['epa_total_per_game']:.2f}",
                     f"{without_stats['epa_total_per_game']:.2f}",
                     f"{without_stats['epa_total_per_game'] - with_stats['epa_total_per_game']:+.2f}")
    else:
        table.add_column("Stat", style="cyan")
        table.add_column(f"With {player2_name}", style="green")
        table.add_column(f"Without {player2_name}", style="yellow")
        table.add_column("Difference", style="magenta")
        
        table.add_row("Games", str(with_stats["games"]), str(without_stats["games"]),
                     f"{without_stats['games'] - with_stats['games']:+d}")
        if "targets" in with_stats:
            table.add_row("Targets/Game", f"{with_stats['targets_per_game']:.1f}",
                         f"{without_stats['targets_per_game']:.1f}",
                         f"{without_stats['targets_per_game'] - with_stats['targets_per_game']:+.1f}")
            table.add_row("Receiving Yards/Game", f"{with_stats['receiving_yards_per_game']:.1f}",
                         f"{without_stats['receiving_yards_per_game']:.1f}",
                         f"{without_stats['receiving_yards_per_game'] - with_stats['receiving_yards_per_game']:+.1f}")
            table.add_row("EPA/Target", f"{with_stats['epa_per_target']:.3f}",
                         f"{without_stats['epa_per_target']:.3f}",
                         f"{without_stats['epa_per_target'] - with_stats['epa_per_target']:+.3f}")
        if "carries" in with_stats:
            table.add_row("Carries/Game", f"{with_stats['carries_per_game']:.1f}",
                         f"{without_stats['carries_per_game']:.1f}",
                         f"{without_stats['carries_per_game'] - with_stats['carries_per_game']:+.1f}")
            table.add_row("Rushing Yards/Game", f"{with_stats['rushing_yards_per_game']:.1f}",
                         f"{without_stats['rushing_yards_per_game']:.1f}",
                         f"{without_stats['rushing_yards_per_game'] - with_stats['rushing_yards_per_game']:+.1f}")
            table.add_row("EPA/Rush", f"{with_stats['epa_per_rush']:.3f}",
                         f"{without_stats['epa_per_rush']:.3f}",
                         f"{without_stats['epa_per_rush'] - with_stats['epa_per_rush']:+.3f}")
        table.add_row("EPA Total/Game", f"{with_stats['epa_total_per_game']:.2f}",
                     f"{without_stats['epa_total_per_game']:.2f}",
                     f"{without_stats['epa_total_per_game'] - with_stats['epa_total_per_game']:+.2f}")
    
    console.print(table)


@app.command()
def defense_report(
    player: str = typer.Argument(..., help="Player name to report on"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season (includes playoffs through February)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Generate a defensive player report for a date range or season.
    
    Examples:
        nflproj defense-report "Aaron Donald" --season 2023
        nflproj defense-report "T.J. Watt" --from 2023-09-01 --to 2023-12-31
    """
    console.print(f"[bold green]Generating defense report for {player}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game_defense = pd.read_parquet(data_path / "player_game_defense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first to create tables.")
        raise typer.Exit(1)
    
    # Parse dates
    if season:
        from_dt = pd.to_datetime(f"{season}-09-01")
        to_dt = pd.to_datetime(f"{season + 1}-02-28")
        console.print(f"[dim]Using season {season} date range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}[/dim]")
    elif from_date and to_date:
        try:
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
        except ValueError:
            console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --season or both --from and --to")
        raise typer.Exit(1)
    
    # Find player
    player_matches = players[players["name"].str.contains(player, case=False, na=False)]
    if len(player_matches) == 0:
        console.print(f"[red]Error:[/red] Player '{player}' not found")
        console.print("\n[yellow]Tip:[/yellow] Use 'nflproj search-players <name>' to find the correct spelling")
        raise typer.Exit(1)
    
    player_id = player_matches.iloc[0]["player_id"]
    player_name = player_matches.iloc[0]["name"]
    player_pos = player_matches.iloc[0]["position"]
    
    # Filter by date range
    games_in_range = games[
        (games["date"] >= from_dt) &
        (games["date"] <= to_dt)
    ]
    
    if len(games_in_range) == 0:
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found in date range {date_range_str}[/yellow]")
        raise typer.Exit(0)
    
    game_ids = games_in_range["game_id"].tolist()
    
    # Get player's defensive stats
    player_stats = player_game_defense[
        (player_game_defense["player_id"] == player_id) &
        (player_game_defense["game_id"].isin(game_ids))
    ]
    
    if len(player_stats) == 0:
        date_range_str = f"{from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}" if season else f"{from_date} to {to_date}"
        console.print(f"[yellow]No games found for {player_name} in date range {date_range_str}[/yellow]")
        raise typer.Exit(0)
    
    # Merge with games for display
    player_stats = pd.merge(
        player_stats,
        games[["game_id", "date", "week", "home_team", "away_team"]],
        on="game_id",
        how="left"
    )
    player_stats = player_stats.sort_values("date")
    
    # Create table
    from rich.table import Table
    table = Table(title=f"Defense Report: {player_name} ({player_pos})")
    table.add_column("Date", style="cyan")
    table.add_column("Week", justify="right")
    table.add_column("Team", style="magenta")
    table.add_column("Opp", style="yellow")
    table.add_column("Tackles", justify="right")
    table.add_column("Sacks", justify="right")
    table.add_column("INTs", justify="right")
    if "qb_hits" in player_stats.columns:
        table.add_column("QB Hits", justify="right")
    table.add_column("EPA Allowed", justify="right", style="red")
    
    for _, row in player_stats.iterrows():
        opponent = row["away_team"] if row["team"] == row["home_team"] else row["home_team"]
        row_data = [
            row["date"].strftime("%Y-%m-%d"),
            str(int(row["week"])),
            row["team"],
            opponent,
            str(int(row["tackles"])),
            f"{row['sacks']:.1f}" if row["sacks"] > 0 else "0",
            str(int(row["interceptions"])),
        ]
        if "qb_hits" in player_stats.columns:
            row_data.append(str(int(row.get("qb_hits", 0))))
        row_data.append(f"{row['epa_allowed']:.2f}")
        table.add_row(*row_data)
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Games: {len(player_stats)}")
    console.print(f"  Total Tackles: {player_stats['tackles'].sum():.0f}")
    console.print(f"  Total Sacks: {player_stats['sacks'].sum():.1f}")
    console.print(f"  Total Interceptions: {player_stats['interceptions'].sum():.0f}")
    if "qb_hits" in player_stats.columns:
        console.print(f"  Total QB Hits: {player_stats['qb_hits'].sum():.0f}")
    console.print(f"  Avg Tackles/Game: {player_stats['tackles'].mean():.1f}")
    console.print(f"  Avg Sacks/Game: {player_stats['sacks'].mean():.2f}")
    console.print(f"  Total EPA Allowed: {player_stats['epa_allowed'].sum():.2f}")


@app.command()
def backtest(
    season: int = typer.Argument(..., help="Season to backtest"),
    model: str = typer.Option("baseline", "--model", help="Model type: baseline, ridge, poisson, gbm, ridge_pos, or gbm_pos"),
    compare: bool = typer.Option(False, "--compare", help="Compare baseline vs ML"),
    ml_model: str = typer.Option("gbm_pos", "--ml-model", help="ML model to compare (gbm_pos, ridge_pos, gbm, ridge)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Run walk-forward backtest for a season.
    
    Examples:
        nflproj backtest 2023 --model baseline
        nflproj backtest 2023 --model gbm_pos
        nflproj backtest 2023 --compare
    """
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
            season, player_game, team_game, features, games, players, ml_model=ml_model
        )
        
        console.print(f"\n[bold]Comparing Baseline vs {results.get('ml_model', 'ML')}[/bold]")
        
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


@app.command()
def rank_players(
    position: str = typer.Argument(..., help="Position to rank (QB, RB, WR, TE)"),
    stat: str = typer.Option("epa_total", "--stat", help="Stat to rank by (epa_total, receiving_yards, rushing_yards, passing_yards, targets, carries, pass_attempts, yards_per_carry, yards_per_target, etc.)"),
    from_date: Optional[str] = typer.Option(None, "--from", help="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = typer.Option(None, "--to", help="End date (YYYY-MM-DD)"),
    season: Optional[int] = typer.Option(None, "--season", help="Filter by season (includes playoffs through February)"),
    limit: int = typer.Option(20, "--limit", help="Number of players to show"),
    ascending: bool = typer.Option(False, "--ascending", help="Sort ascending (lowest first)"),
    data_dir: Optional[str] = typer.Option(
        None,
        "--data-dir",
        help="Data directory with Parquet files"
    )
):
    """
    Rank players by position and stat over a date range or season.
    
    Examples:
        nflproj rank-players RB --stat yards_per_carry --season 2023 --limit 10
        nflproj rank-players WR --stat receiving_yards --season 2023
        nflproj rank-players QB --stat epa_total --season 2023
        nflproj rank-players RB --stat epa_total --from 2023-09-01 --to 2023-12-31 --ascending
    """
    console.print(f"[bold green]Ranking {position} by {stat}...[/bold green]")
    
    data_path = Path(data_dir) if data_dir else PARQUET_DIR
    
    # Load tables
    try:
        player_game = pd.read_parquet(data_path / "player_game_offense.parquet")
        games = pd.read_parquet(data_path / "games.parquet")
        players_df = pd.read_parquet(data_path / "players.parquet")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Run 'build' command first.")
        raise typer.Exit(1)
    
    # Parse dates - if season is provided, use that instead
    if season:
        # NFL seasons run from September to February (next year)
        from_dt = pd.to_datetime(f"{season}-09-01")
        to_dt = pd.to_datetime(f"{season + 1}-02-28")
        console.print(f"[dim]Using season {season} date range: {from_dt.strftime('%Y-%m-%d')} to {to_dt.strftime('%Y-%m-%d')}[/dim]")
    elif from_date and to_date:
        try:
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
        except ValueError:
            console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
            raise typer.Exit(1)
    else:
        console.print("[red]Error:[/red] Must provide either --season or both --from and --to")
        raise typer.Exit(1)
    
    # Filter games by date
    games_in_range = games[
        (games["date"] >= from_dt) &
        (games["date"] <= to_dt)
    ]
    
    if len(games_in_range) == 0:
        console.print(f"[yellow]No games found in date range {from_date} to {to_date}[/yellow]")
        console.print(f"Available date range: {games['date'].min().strftime('%Y-%m-%d')} to {games['date'].max().strftime('%Y-%m-%d')}")
        console.print(f"Available seasons: {', '.join(map(str, sorted(games['season'].unique())))}")
        raise typer.Exit(0)
    
    game_ids = games_in_range["game_id"].tolist()
    
    # Filter players by position
    position_players = players_df[players_df["position"] == position.upper()]
    
    if len(position_players) == 0:
        console.print(f"[red]Error:[/red] No players found for position {position}")
        raise typer.Exit(1)
    
    # Get stats for these players in date range
    player_stats = player_game[
        (player_game["player_id"].isin(position_players["player_id"])) &
        (player_game["game_id"].isin(game_ids))
    ]
    
    if len(player_stats) == 0:
        console.print(f"[yellow]No games found for {position} players in date range[/yellow]")
        raise typer.Exit(0)
    
    # Aggregate by player
    agg_dict = {
        "targets": "sum",
        "receptions": "sum",
        "receiving_yards": "sum",
        "carries": "sum",
        "rushing_yards": "sum",
        "epa_total": "sum",
        "epa_receiving": "sum",
        "epa_rushing": "sum",
        "redzone_touches": "sum",
        "success_rate": "mean"
    }
    
    # Add QB stats if available
    if "pass_attempts" in player_stats.columns:
        agg_dict.update({
            "pass_attempts": "sum",
            "completions": "sum",
            "passing_yards": "sum",
            "passing_tds": "sum",
            "interceptions": "sum",
            "epa_passing": "sum"
        })
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in player_stats.columns}
    
    rankings = player_stats.groupby("player_id").agg(agg_dict).reset_index()
    
    # Merge with player names
    rankings = pd.merge(rankings, players_df[["player_id", "name"]], on="player_id", how="left")
    
    # Calculate derived stats
    if "targets" in rankings.columns:
        rankings["yards_per_target"] = rankings["receiving_yards"] / rankings["targets"].replace(0, 1)
        rankings["catch_rate"] = rankings["receptions"] / rankings["targets"].replace(0, 1)
        rankings["epa_per_target"] = rankings["epa_receiving"] / rankings["targets"].replace(0, 1)
    
    if "carries" in rankings.columns:
        rankings["yards_per_carry"] = rankings["rushing_yards"] / rankings["carries"].replace(0, 1)
        rankings["epa_per_rush"] = rankings["epa_rushing"] / rankings["carries"].replace(0, 1)
    
    if "pass_attempts" in rankings.columns:
        rankings["yards_per_attempt"] = rankings["passing_yards"] / rankings["pass_attempts"].replace(0, 1)
        rankings["completion_rate"] = rankings["completions"] / rankings["pass_attempts"].replace(0, 1)
        rankings["epa_per_attempt"] = rankings["epa_passing"] / rankings["pass_attempts"].replace(0, 1)
    
    # Check if stat exists
    if stat not in rankings.columns:
        console.print(f"[red]Error:[/red] Stat '{stat}' not available")
        console.print(f"Available stats: {', '.join(sorted(rankings.columns))}")
        raise typer.Exit(1)
    
    # Sort by stat
    rankings = rankings.sort_values(stat, ascending=ascending)
    rankings = rankings.head(limit)
    
    # Display rankings
    table = Table(title=f"{position.upper()} Rankings by {stat} ({from_date} to {to_date})")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Player", style="magenta")
    
    # Add columns based on position
    if position.upper() == "QB":
        table.add_column("Att", justify="right")
        table.add_column("Comp", justify="right")
        table.add_column("Pass Yds", justify="right")
        table.add_column("TDs", justify="right")
        table.add_column("INTs", justify="right")
        table.add_column("EPA", justify="right", style="green")
        table.add_column(stat.replace("_", " ").title(), justify="right", style="yellow")
    else:
        table.add_column("Targets", justify="right")
        table.add_column("Rec Yds", justify="right")
        table.add_column("Carries", justify="right")
        table.add_column("Rush Yds", justify="right")
        table.add_column("EPA", justify="right", style="green")
        table.add_column(stat.replace("_", " ").title(), justify="right", style="yellow")
    
    for idx, (_, row) in enumerate(rankings.iterrows(), 1):
        if position.upper() == "QB":
            table.add_row(
                str(idx),
                row["name"],
                str(int(row.get("pass_attempts", 0))),
                str(int(row.get("completions", 0))),
                str(int(row.get("passing_yards", 0))),
                str(int(row.get("passing_tds", 0))),
                str(int(row.get("interceptions", 0))),
                f"{row['epa_total']:.2f}",
                f"{row[stat]:.2f}" if isinstance(row[stat], float) else str(row[stat])
            )
        else:
            table.add_row(
                str(idx),
                row["name"],
                str(int(row.get("targets", 0))),
                str(int(row.get("receiving_yards", 0))),
                str(int(row.get("carries", 0))),
                str(int(row.get("rushing_yards", 0))),
                f"{row['epa_total']:.2f}",
                f"{row[stat]:.2f}" if isinstance(row[stat], float) else str(row[stat])
            )
    
    console.print(table)
    
    console.print(f"\n[green]Showing top {len(rankings)} {position} players by {stat}[/green]")


if __name__ == "__main__":
    app()
