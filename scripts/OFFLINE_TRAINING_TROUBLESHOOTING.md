# Offline Training Troubleshooting

## Why Training Might Be Stuck

When training with large offline datasets (1000+ episodes, 4GB+), the script may appear stuck during:

1. **Loading JSON files** - Loading 100+ large JSON files can take 5-30 minutes
2. **Converting to replay buffer format** - Normalizing millions of transitions
3. **Populating replay buffer** - Adding transitions one by one (slowest part)

## Solutions

### Option 1: Use Smaller Dataset (Recommended for Testing)

Test with a smaller subset first:

```bash
# Test with just 100 episodes
python scripts/collect_offline_data.py --episodes 100 --output-dir ./data/offline_test

# Train with test dataset
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline_test --rho 0.3 --timesteps 1000
```

### Option 2: Filter by Scenario

If you have multiple scenario files, filter to reduce dataset size:

```bash
# Train only with baseline scenario
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --offline-scenario baseline --rho 0.3 --timesteps 1000
```

### Option 3: Monitor Progress

The updated script now shows progress. Look for:
- "Loading: X/Y files (Z%)" - File loading progress
- "Normalizing: X/Y transitions (Z%)" - State normalization progress  
- "Progress: X/Y transitions (Z%)" - Replay buffer population progress

### Option 4: Use Most Recent File Only

Modify the script to load only the most recent file (temporary workaround):

```python
# In offline_rl_training.py, change:
episode_files = list(data_path.glob(pattern))
# To:
episode_files = sorted(list(data_path.glob(pattern)), key=lambda x: x.stat().st_mtime)[-1:]  # Only most recent
```

### Option 5: Increase Memory/Resources

Large datasets require:
- **RAM**: At least 8-16GB for 4GB dataset
- **Disk**: Fast SSD recommended
- **Time**: Be patient - 4GB dataset may take 10-30 minutes to load

## Expected Time Estimates

| Dataset Size | Files | Load Time | Convert Time | Buffer Time | Total |
|-------------|-------|----------|--------------|-------------|-------|
| 100 episodes | 1-5 | 1-5 sec | 1-5 sec | 5-15 sec | ~30 sec |
| 1000 episodes | 10-20 | 1-2 min | 1-2 min | 2-5 min | ~5-10 min |
| 5000 episodes | 50-100 | 5-10 min | 5-10 min | 10-30 min | ~20-50 min |
| 10,000+ episodes | 100+ | 10-30 min | 10-30 min | 30-60 min | ~50-120 min |

## Quick Fix: Use Smaller Dataset

```bash
# Create a smaller test dataset (100 episodes)
python scripts/collect_offline_data.py --episodes 100 --output-dir ./data/offline_small

# Train with smaller dataset
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline_small --rho 0.3 --timesteps 1000
```

## Check if Script is Actually Running

1. **Check CPU usage**: Task Manager should show Python using CPU
2. **Check memory**: Memory usage should increase gradually
3. **Check disk**: Disk read activity should be visible
4. **Look for progress messages**: Script now prints progress every 10% or 10,000 transitions

## If Truly Stuck

1. **Kill and restart** with smaller dataset
2. **Check for errors** in terminal output
3. **Verify data files** are not corrupted
4. **Use subset** of data files

## Optimized Workflow

For large datasets, recommended workflow:

```bash
# Step 1: Collect in manageable batches
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline

# Step 2: Test training with first batch
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --offline-scenario baseline --rho 0.3 --timesteps 1000

# Step 3: If successful, collect more data
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --seed-start 1000

# Step 4: Train with full dataset (expect longer load time)
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --rho 0.3 --timesteps 10000
```

