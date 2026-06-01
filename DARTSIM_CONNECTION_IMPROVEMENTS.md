# DARTSim Connection Improvements

## Summary

Made improvements to both DARTSim C++ source code and Python client to address connection issues.

## Changes Made

### 1. DARTSim C++ Source (`dartsim-master/src/dartsim/AdaptInterface.cpp`)

#### Improved `serviceClient()`:
- Added TCP_NODELAY and SO_KEEPALIVE socket options
- Added comprehensive error handling with try-catch blocks
- Better connection cleanup on errors

#### Enhanced `readCmd()`:
- Added socket state validation before reading
- Better error handling and logging
- Improved exception handling

#### Improved `sendBytes()`:
- Added socket state validation
- Better error messages
- Ensures socket is open before sending

#### Enhanced `handleClientCmd()`:
- Wrapped command handlers in try-catch
- Ensures responses are always sent (even on errors)
- Handles empty commands gracefully
- Sends error responses if handlers throw

### 2. Python Client (`src/dartsim_env.py`)

#### Improved `_send_command()`:
- Changed from one-connection-per-command to persistent connection pattern
- Uses larger read chunks (4096 bytes) instead of byte-by-byte
- Better timeout handling
- Improved error recovery

#### Enhanced `_connect()`:
- Added TCP_NODELAY and SO_KEEPALIVE socket options
- Added small delay after connect to let DARTSim fully accept connection

## Current Status

The changes compile successfully, but there are still connection issues:
- DARTSim accepts connections but closes them immediately
- Responses are empty or connections close before reading

## Next Steps

1. **Investigate DARTSim connection behavior**: The issue appears to be that DARTSim's `readCmd()` might be failing silently, causing the service loop to exit
2. **Check DARTSim logs**: Need to see what DARTSim is actually doing when connections are accepted
3. **Consider reverting readCmd() changes**: The try-catch that returns nullptr might be causing the connection to close prematurely
4. **Test with original DARTSim**: Compare behavior with unmodified DARTSim to isolate the issue

## To Rebuild DARTSim

```powershell
# Rebuild in Docker
docker exec a65c5bd74a66 bash -c "cd /headless/dartsim/build && make -j$(nproc)"

# Or use the script
.\rebuild_dartsim_docker.ps1
```

## Files Modified

1. `dartsim-master/src/dartsim/AdaptInterface.cpp` - DARTSim server improvements
2. `src/dartsim_env.py` - Python client improvements
3. `rebuild_dartsim_docker.ps1` - Rebuild script
4. `rebuild_dartsim.sh` - Rebuild script (Linux)

