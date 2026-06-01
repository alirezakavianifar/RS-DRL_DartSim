# Generator-Based Sequential Loading Explanation

## How It Currently Works

The generator approach **does** load episodes sequentially, but there's one bottleneck:

### Current Flow:
```
1. Load File 1 (entire file loaded into memory) → json.load()
2. Yield episodes one by one from File 1
3. Delete File 1 data from memory
4. Load File 2 (entire file loaded into memory) → json.load()
5. Yield episodes one by one from File 2
6. Delete File 2 data from memory
...
```

### The Issue:
- `json.load(f)` loads the **entire file** into memory first
- Then we yield episodes one by one
- This means for a 25MB file, we temporarily hold 25MB in memory

### The Real Problem Right Now:
The error happens **before** we even get to load data:
```python
import torch  # ← ERROR HERE (PyTorch DLL loading fails)
```

This is a Windows paging file issue, not related to our data loading.

## True Sequential Loading (if needed)

If we want to load episodes truly one-by-one without loading entire files, we'd need to:
1. Parse JSON incrementally (use `ijson` library)
2. Stream parse the file
3. Yield each episode as it's parsed

But this is more complex and may not be necessary if files aren't too large.

## Current Generator Benefits

Even with the current approach:
- ✅ Only **one file** in memory at a time (not all files)
- ✅ Episodes are yielded incrementally
- ✅ After yielding, memory can be freed
- ✅ Replay buffer is populated incrementally

## Memory Usage Comparison

### Old Approach (All-at-once):
```
Load File 1 → 25MB in memory
Load File 2 → 25MB in memory (total: 50MB)
Load File 3 → 25MB in memory (total: 75MB)
...
Load File 179 → 25MB in memory (total: 4GB+)
```

### New Generator Approach:
```
Load File 1 → 25MB → Yield episodes → Delete → 0MB
Load File 2 → 25MB → Yield episodes → Delete → 0MB
...
```

**Peak memory**: Only one file (~25MB) + batch buffer (~10k transitions)

## The PyTorch Import Issue

The current error is:
```
OSError: [WinError 1455] The paging file is too small
```

This happens when PyTorch tries to load its DLLs, **before** we load any data.

**Solution**: Increase Windows paging file (virtual memory) as described earlier.



