"""Tests for shrinkage functions."""

import pytest
import numpy as np
from nflproj.models import apply_shrinkage


def test_shrinkage_basic():
    """Test basic shrinkage calculation."""
    player_mean = 0.8
    position_mean = 0.5
    n_opportunities = 10
    k = 30
    
    shrunk = apply_shrinkage(player_mean, position_mean, n_opportunities, k, "WR", "targets")
    
    expected = (10 * 0.8 + 30 * 0.5) / (10 + 30)
    assert abs(shrunk - expected) < 1e-6


def test_shrinkage_no_opportunities():
    """Test shrinkage when player has no opportunities."""
    player_mean = 0.8
    position_mean = 0.5
    n_opportunities = 0
    k = 30
    
    shrunk = apply_shrinkage(player_mean, position_mean, n_opportunities, k, "WR", "targets")
    
    # Should return position mean
    assert shrunk == position_mean


def test_shrinkage_high_opportunities():
    """Test shrinkage with many opportunities (should approach player mean)."""
    player_mean = 0.8
    position_mean = 0.5
    n_opportunities = 1000
    k = 30
    
    shrunk = apply_shrinkage(player_mean, position_mean, n_opportunities, k, "WR", "targets")
    
    # Should be very close to player mean
    assert abs(shrunk - player_mean) < 0.1


def test_shrinkage_position_specific_k():
    """Test that position-specific k values are used."""
    player_mean = 0.8
    position_mean = 0.5
    n_opportunities = 10
    
    # WR should use k=30 for targets
    shrunk_wr = apply_shrinkage(player_mean, position_mean, n_opportunities, 30, "WR", "targets")
    
    # RB should use k=15 for targets
    shrunk_rb = apply_shrinkage(player_mean, position_mean, n_opportunities, 30, "RB", "targets")
    
    # Should be different
    assert shrunk_wr != shrunk_rb


def test_shrinkage_carries_vs_targets():
    """Test that different k values are used for carries vs targets."""
    player_mean = 0.6
    position_mean = 0.4
    n_opportunities = 10
    
    shrunk_targets = apply_shrinkage(player_mean, position_mean, n_opportunities, 30, "RB", "targets")
    shrunk_carries = apply_shrinkage(player_mean, position_mean, n_opportunities, 40, "RB", "carries")
    
    # Should be different due to different k
    assert shrunk_targets != shrunk_carries
