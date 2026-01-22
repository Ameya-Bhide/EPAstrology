# ML Model Investigation & Tuning Results

## Current Performance Summary

### Baseline Model (Best So Far)
- Targets MAE: **1.59**
- Carries MAE: **1.40**
- Total EPA MAE: **2.49**

### ML Models (After Tuning)

| Model | Targets MAE | Carries MAE | Total EPA MAE | Notes |
|-------|------------|-------------|---------------|-------|
| Ridge (alpha=1.0, old) | 1.74 | 1.58 | 2.54 | Over-regularized |
| Ridge (alpha=0.5, new) | 1.73 | 1.57 | 2.54 | Better, still worse than baseline |
| Poisson | 1.88 | 2.07 | 2.59 | Worse - not good for this data |
| GBM (tuned) | **1.69** | **1.53** | **2.54** | Best ML model, still worse than baseline |

## Key Findings

### 1. **Baseline is Strong**
The baseline model's simple approach (weighted recent average + season share) is hard to beat. This suggests:
- Recent performance is highly predictive
- Simple heuristics work well for role prediction
- ML might be overfitting or missing key signals

### 2. **ML Underperformance Causes**

**Possible Reasons:**
- **Early weeks have little training data**: Walk-forward means week 1-3 have very few examples
- **Feature quality**: Features might not capture the right signals
- **Overfitting**: ML models might be fitting noise in early weeks
- **Baseline uses team context better**: Baseline explicitly uses projected team volume, ML might not weight this correctly

### 3. **What's Working**
- GBM is best ML model (1.69 vs 1.59 baseline for targets)
- Lower regularization (alpha=0.5) helps Ridge
- Poisson regression doesn't help (count data but not Poisson-distributed)

### 4. **What's Not Working**
- All ML models still worse than baseline
- Efficiency prediction identical (both use baseline)
- Rank correlations are low (0.12-0.22) - efficiency is hard to predict

## Recommendations for Improvement

### Immediate Actions

1. **Feature Engineering**
   - Add position-specific features (WR vs RB have different patterns)
   - Add matchup features (opponent defense strength by position)
   - Add game script features (game flow, score differential)
   - Add injury/availability features

2. **Model Architecture**
   - Try ensemble: blend baseline + ML predictions
   - Use baseline as a feature in ML model
   - Separate models by position (WR model, RB model, etc.)
   - Add team-level features more explicitly

3. **Training Strategy**
   - Use more data: train on multiple seasons
   - Weight recent games more heavily
   - Handle early weeks differently (use baseline until enough data)

4. **Efficiency Prediction**
   - Build ML model for efficiency (currently only baseline)
   - Add matchup-specific efficiency features
   - Consider opponent-adjusted efficiency

### Longer-term Improvements

1. **Advanced Features**
   - Snap counts and route participation (if available)
   - Weather/stadium factors
   - Vegas lines (implied game script)
   - Injury reports

2. **Better Models**
   - XGBoost with proper tuning
   - Neural networks for complex interactions
   - Time series models (LSTM, etc.)

3. **Evaluation**
   - Position-specific metrics
   - Fantasy points (if that's the goal)
   - Betting market comparisons

## Next Steps

1. **Try ensemble approach**: `0.7 * baseline + 0.3 * ML`
2. **Position-specific models**: Train separate models for WR, RB, TE
3. **Add more features**: Especially team context and matchups
4. **Multi-season training**: Use 2022+2023 data for more training examples

## Code Changes Made

- ✅ Lowered Ridge alpha from 1.0 to 0.5
- ✅ Improved GBM hyperparameters (learning_rate=0.05, subsample=0.8)
- ✅ Added efficiency features to ML model
- ✅ Created diagnostics module for error analysis
- ✅ Added improved model class with hyperparameter tuning (models_improved.py)

## Conclusion

The baseline model is surprisingly strong. ML models are close but not beating it yet. The gap is small (1.69 vs 1.59 for targets), suggesting we're on the right track but need better features or architecture to surpass the baseline.
