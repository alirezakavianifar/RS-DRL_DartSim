# Guide: Collecting 5000 Episodes of Offline Data

This guide explains how to efficiently collect 5000 episodes of DARTSim simulation data for offline RL training.

## Quick Start

### Basic Command

```bash
# Collect 5000 episodes (baseline scenario)
python scripts/collect_offline_data.py --episodes 5000 --output-dir ./data/offline --scenario baseline
```

## Time Estimates

- **Per Episode**: ~5-30 seconds (depending on simulation length)
- **5000 Episodes**: ~7-42 hours (estimated)
- **With failures/retries**: May take longer

## Recommended Approaches

### Option 1: Single Long-Run (Recommended for consistency)

Run the collection in one command, but with intermediate checkpoints:

```bash
# Activate virtual environment
cd e:\projects\dart
.\venv\Scripts\Activate.ps1

# Start collection (will save checkpoints every 10 episodes)
python scripts/collect_offline_data.py --episodes 5000 --output-dir ./data/offline --scenario baseline --seed-start 42
```

**Advantages:**
- Simple, single command
- Automatic checkpointing (saves every 10 episodes)
- Consistent seed sequence

**Considerations:**
- Long runtime (may take 7-42 hours)
- If interrupted, can resume from last checkpoint
- Keep terminal/computer running

### Option 2: Batch Collection (Recommended for reliability)

Split into smaller batches and combine:

```bash
# Batch 1: Episodes 1-1000
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 42

# Batch 2: Episodes 1001-2000
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 1042

# Batch 3: Episodes 2001-3000
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 2042

# Batch 4: Episodes 3001-4000
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 3042

# Batch 5: Episodes 4001-5000
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 4042
```

**Advantages:**
- Can run batches separately (e.g., overnight)
- Easier to monitor progress
- Less risk of data loss if interrupted
- Can verify each batch before continuing

**Note:** All batches will save to the same directory. The script saves files with timestamps, so they won't overwrite.

### Option 3: Parallel Collection (Advanced)

If you have multiple Docker containers or want to collect different scenarios:

```powershell
# Terminal 1: Baseline scenario
python scripts/collect_offline_data.py --episodes 2500 --scenario baseline --seed-start 42

# Terminal 2: Medium scenario  
python scripts/collect_offline_data.py --episodes 2500 --scenario medium --seed-start 10042
```

## Disk Space Requirements

- **Per episode**: ~10-50 KB (depending on trajectory length)
- **5000 episodes**: ~50-250 MB
- **With all files (episodes, trajectories, results, metadata)**: ~200-1000 MB

**Recommendation:** Ensure at least 2-5 GB free space for safety.

## Monitoring Progress

### Check Progress During Collection

The script prints progress for each episode:
```
Episode 1/5000 (seed=42)... OK (steps=45, targets=2)
Episode 2/5000 (seed=43)... OK (steps=38, targets=1)
...
Episode 10/5000 (seed=51)... OK (steps=52, targets=3)
  Saved intermediate checkpoint (10 episodes)
```

### Check Saved Files

```powershell
# List collected files
ls data/offline/episodes_*.json

# Count episodes collected
(ls data/offline/episodes_*.json).Count
```

### Verify Data Quality

After collection, verify the dataset:

```bash
python scripts/use_offline_data_example.py --data-dir ./data/offline
```

## Running in Background (Windows)

To run collection in background without blocking terminal:

### Option A: PowerShell Background Job

```powershell
# Start background job
$job = Start-Job -ScriptBlock {
    cd e:\projects\dart
    .\venv\Scripts\Activate.ps1
    python scripts/collect_offline_data.py --episodes 5000 --output-dir ./data/offline --scenario baseline
}

# Check job status
Get-Job $job

# View output
Receive-Job $job

# Wait for completion
Wait-Job $job
```

### Option B: Run in Separate Terminal

1. Open a new PowerShell window
2. Run the collection command
3. Minimize the window
4. Check periodically

## Complete Example: Collect 5000 Episodes

```powershell
# 1. Navigate to project
cd e:\projects\dart

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Verify Docker container is running
docker ps --filter "name=dartsim"

# 4. Start collection (recommended: use batch approach)
# Batch 1
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 42

# Batch 2 (after batch 1 completes)
python scripts/collect_offline_data.py --episodes 1000 --output-dir ./data/offline --scenario baseline --seed-start 1042

# Continue with batches 3, 4, 5...

# 5. Verify collection
python scripts/use_offline_data_example.py --data-dir ./data/offline

# 6. Use for offline RL training
python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --episodes 5000 --rho 0.3
```

## Tips for Large-Scale Collection

### 1. Start Small, Scale Up

```bash
# Test with 10 episodes first
python scripts/collect_offline_data.py --episodes 10

# Then scale to 100
python scripts/collect_offline_data.py --episodes 100

# Finally scale to 5000 (in batches)
```

### 2. Monitor System Resources

- **CPU**: DARTSim simulations are CPU-intensive
- **Memory**: Docker container uses memory
- **Disk**: Ensure sufficient space

### 3. Handle Interruptions

The script saves checkpoints every 10 episodes. If interrupted:
- Check the last saved file in `data/offline/`
- Resume collection with a new seed_start value
- The script will create new files (won't overwrite)

### 4. Combine Multiple Scenarios

For diverse dataset:

```bash
# Collect from all scenarios
python scripts/collect_offline_data.py --episodes 1667 --scenario baseline --seed-start 42
python scripts/collect_offline_data.py --episodes 1667 --scenario medium --seed-start 10000
python scripts/collect_offline_data.py --episodes 1666 --scenario hard --seed-start 20000
# Total: ~5000 episodes
```

### 5. Verify Before Training

After collection, always verify:

```bash
# Check dataset statistics
python scripts/use_offline_data_example.py --data-dir ./data/offline

# Verify file integrity
python scripts/offline_rl_training.py --data-dir ./data/offline
```

## Troubleshooting

### Issue: Collection is very slow

**Solutions:**
- Check Docker container performance: `docker stats dartsim`
- Reduce timeout if simulations are hanging
- Consider collecting fewer episodes per batch

### Issue: Many failed episodes

**Solutions:**
- Check Docker logs: `docker logs dartsim`
- Verify container is healthy: `docker ps`
- Try a different scenario (e.g., baseline instead of hard)

### Issue: Out of disk space

**Solutions:**
- Clean up old data files
- Use a different output directory with more space
- Collect in smaller batches and archive after each

### Issue: Collection interrupted

**Solutions:**
- Check last checkpoint file
- Resume with new seed_start (current_episode + seed_start)
- All files are timestamped, so no data loss

## Expected Output

After successful collection, you'll have:

```
data/offline/
├── episodes_baseline_5000_YYYYMMDD_HHMMSS.json    # RL transitions (main file)
├── trajectories_baseline_5000_YYYYMMDD_HHMMSS.json # Raw trajectories
├── results_baseline_5000_YYYYMMDD_HHMMSS.json     # Mission results
└── metadata_baseline_5000_YYYYMMDD_HHMMSS.json    # Dataset metadata
```

## Next Steps

After collecting 5000 episodes:

1. **Analyze the dataset:**
   ```bash
   python scripts/use_offline_data_example.py --data-dir ./data/offline
   ```

2. **Train offline RL:**
   ```bash
   python scripts/train_rs_drl.py --offline --offline-data-dir ./data/offline --episodes 5000 --rho 0.3 --timesteps 100000
   ```

3. **Compare with online training:**
   ```bash
   python scripts/train_rs_drl.py --method rs_drl --timesteps 100000 --rho 0.3
   ```

## Summary

**Recommended workflow for 5000 episodes:**

1. Use **batch approach** (5 batches of 1000 episodes each)
2. Run each batch separately (can pause between batches)
3. Verify each batch before continuing
4. Monitor disk space and system resources
5. After collection, verify dataset quality
6. Use collected data for offline RL training

**Quick command:**
```bash
python scripts/collect_offline_data.py --episodes 5000 --output-dir ./data/offline --scenario baseline --seed-start 42
```

Good luck with your data collection!

