"""
Live DARTSim Gymnasium Environment
====================================
Connects to a running DARTSim instance via its TCP adaptation-manager
interface (default port 5418).  The Docker container must be started with
port 5418 exposed and DARTSim running inside it.

Quick start
-----------
1. Pull and start the container::

       docker run -d -p 5901:5901 -p 6901:6901 -p 5418:5418 \\
           --name dartsim gabrielmoreno/dartsim:1.0

2. (Optionally) launch DARTSim immediately::

       docker exec -d dartsim bash -c "cd /headless/dartsim && ./run.sh"

3. Then use this class::

       env = LiveDARTSimEnv()
       obs, info = env.reset()   # starts DARTSim if not already running

Observation vector (17 floats)
-------------------------------
Index  Feature
-----  -------
0      position_x  (absolute)
1      position_y  (absolute)
2      altitude_level  (1-based)
3      formation  (0=LOOSE, 1=TIGHT)
4      ecm  (0/1)
5      direction_x
6      direction_y
7-11   threats_ahead[0..4]  (forward threat sensor, 5 cells)
12-16  targets_ahead[0..4]  (forward target sensor, 5 cells)

Action space — Discrete(8)
--------------------------
0  IncAlt    1  DecAlt    2  IncAlt2   3  DecAlt2
4  GoTight   5  GoLoose   6  EcmOn     7  EcmOff
"""

import json
import socket
import subprocess
import time
import logging
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENSOR_HORIZON = 5  # number of look-ahead cells for threat/target sensors

ACTIONS: List[str] = [
    "IncAlt",   # 0
    "DecAlt",   # 1
    "IncAlt2",  # 2
    "DecAlt2",  # 3
    "GoTight",  # 4
    "GoLoose",  # 5
    "EcmOn",    # 6
    "EcmOff",   # 7
]


# ---------------------------------------------------------------------------
# Low-level DARTSim TCP client
# ---------------------------------------------------------------------------

class _DARTSimTCPClient:
    """
    Thin wrapper around the DARTSim TCP adaptation-manager protocol.

    Commands are newline-terminated ASCII strings; responses (also
    newline-terminated) are JSON values.
    """

    def __init__(self, host: str = "localhost", port: int = 5418, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if self._sock is not None:
            self.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        s.connect((self.host, self.port))
        self._sock = s
        self._buf = b""
        logger.debug("Connected to DARTSim at %s:%d", self.host, self.port)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._buf = b""

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    # ------------------------------------------------------------------
    # Raw protocol
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> None:
        assert self._sock is not None, "Not connected"
        self._sock.sendall((cmd + "\n").encode())

    def _recv_line(self) -> str:
        """Receive bytes until a newline is found; return decoded line."""
        while b"\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("DARTSim closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode().strip()

    def send_command(self, cmd: str) -> str:
        """Send *cmd* and return the raw response string."""
        logger.debug("→ %s", cmd)
        self._send(cmd)
        resp = self._recv_line()
        logger.debug("← %s", resp)
        return resp

    def send_command_json(self, cmd: str):
        """Send *cmd* and parse the JSON response."""
        raw = self.send_command(cmd)
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Protocol helpers
    # ------------------------------------------------------------------

    def get_parameters(self) -> Dict:
        return self.send_command_json("getParameters")

    def get_state(self) -> Dict:
        return self.send_command_json("getState")

    def read_forward_threat_sensor(self, cells: int = _SENSOR_HORIZON) -> List[bool]:
        resp = self.send_command_json(f"readForwardThreatSensor {cells}")
        return [bool(v) for v in resp]

    def read_forward_target_sensor(self, cells: int = _SENSOR_HORIZON) -> List[bool]:
        resp = self.send_command_json(f"readForwardTargetSensor {cells}")
        return [bool(v) for v in resp]

    def step(self, tactics: List[str], decision_time_ms: float = 0.0) -> bool:
        """
        Execute *tactics* and advance the simulation by one period.

        NOTE: DARTSim's step command returns whether a *target was detected*
        in this step (not whether the simulation is finished).  Call
        ``finished()`` separately to check episode termination.

        Returns ``True`` if a target was detected this step.
        """
        tactic_json = json.dumps(tactics)
        resp = self.send_command(f"step {tactic_json} {decision_time_ms}")
        return resp.strip().lower() == "true"  # target-detected flag

    def finished(self) -> bool:
        """Return ``True`` when the simulation episode is over."""
        resp = self.send_command("finished")
        return resp.strip().lower() == "true"

    def get_results(self) -> Dict:
        return self.send_command_json("getResults")


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class LiveDARTSimEnv(gym.Env):
    """
    Gymnasium environment backed by a live DARTSim TCP server.

    Parameters
    ----------
    host : str
        Hostname where DARTSim is listening (default ``"localhost"``).
    port : int
        TCP port (default ``5418``).
    container_name : str or None
        Docker container name.  When set, ``reset()`` will automatically
        kill and restart DARTSim inside the container at the start of each
        episode.  Set to ``None`` to manage the DARTSim process yourself.
    sim_args : str
        Extra command-line arguments forwarded to ``./run.sh`` inside the
        container (e.g. ``"--seed=42 --map-size=40"``).
    connect_timeout : float
        Seconds to wait for DARTSim to start listening (default 30).
    step_timeout : float
        Per-command socket timeout in seconds (default 10).
    sensor_horizon : int
        Number of look-ahead cells for the forward sensors (default 5).
    reward_weights : dict or None
        Override default terminal-reward coefficients.
        Keys: ``mission_success``, ``targets_detected``, ``survival``,
        ``destruction``, ``step_penalty``.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    OBS_DIM = 17  # must match _build_obs

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5418,
        container_name: Optional[str] = "dartsim",
        sim_args: str = "",
        connect_timeout: float = 30.0,
        step_timeout: float = 10.0,
        sensor_horizon: int = _SENSOR_HORIZON,
        reward_weights: Optional[Dict] = None,
    ):
        super().__init__()

        self._host = host
        self._port = port
        self._container = container_name
        self._sim_args = sim_args
        self._connect_timeout = connect_timeout
        self._step_timeout = step_timeout
        self._sensor_horizon = sensor_horizon

        self._reward_w = {
            "mission_success": 0.4,
            "targets_detected": 0.3,
            "survival": 0.2,
            "destruction": -0.5,
            "step_penalty": -0.01,
        }
        if reward_weights:
            self._reward_w.update(reward_weights)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._client = _DARTSimTCPClient(host=host, port=port, timeout=step_timeout)
        self._params: Dict = {}
        self._ep_steps: int = 0
        self._last_obs: np.ndarray = np.zeros(self.OBS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def _kill_dartsim(self) -> None:
        """Kill any running DARTSim process inside the Docker container and
        wait until it is fully gone before returning."""
        if not self._container:
            return
        try:
            # SIGKILL (-9) for immediate termination
            subprocess.run(
                ["docker", "exec", self._container, "bash", "-c",
                 "pkill -9 -x dartsim 2>/dev/null; pkill -9 -f 'dartsim' 2>/dev/null; true"],
                capture_output=True, timeout=10
            )
            # Poll until the process is gone (up to 5 s)
            for _ in range(10):
                time.sleep(0.3)
                r = subprocess.run(
                    ["docker", "exec", self._container, "bash", "-c",
                     "pgrep -x dartsim 2>/dev/null && echo alive || echo gone"],
                    capture_output=True, text=True, timeout=5
                )
                if "gone" in r.stdout:
                    break
            else:
                logger.warning("DARTSim process did not exit within 3 s")
        except Exception as e:
            logger.warning("Could not kill DARTSim: %s", e)

    def _ensure_container_running(self) -> None:
        """Make sure the Docker container is in the 'running' state."""
        if not self._container:
            return
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", self._container],
            capture_output=True, text=True, timeout=10
        )
        status = r.stdout.strip()
        if r.returncode != 0:
            raise RuntimeError(
                f"Container '{self._container}' does not exist. "
                "Run utils/start_dartsim_live.ps1 first."
            )
        if status != "running":
            logger.info("Container '%s' is '%s' — starting it...", self._container, status)
            subprocess.run(
                ["docker", "start", self._container],
                capture_output=True, timeout=20
            )
            time.sleep(1.0)

    def _start_dartsim(self) -> None:
        """Start DARTSim in TCP-server mode inside the Docker container."""
        if not self._container:
            return
        # Ensure the container itself is running before trying exec
        self._ensure_container_running()
        args = self._sim_args.strip()
        # Use the compiled binary directly — NOT run.sh (which runs the
        # library/simple-cpp example that doesn't open a TCP port).
        #
        # IMPORTANT: use `exec` (not `nohup ... &`) so bash is replaced by
        # the binary process.  With `docker exec -d`, dockerd then owns the
        # process directly, avoiding SIGHUP when the exec-session shell exits.
        cmd = (
            f"cd /headless/dartsim && "
            f"exec ./build/src/dartsim/dartsim {args} "
            f"> /tmp/dartsim.log 2>&1"
        )
        r = subprocess.run(
            ["docker", "exec", "-d", self._container, "bash", "-c", cmd],
            capture_output=True, timeout=10
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Failed to start DARTSim in container '{self._container}': "
                f"{r.stderr.decode().strip()}"
            )

    def _wait_for_dartsim(self) -> None:
        """
        Poll until DARTSim is accepting connections AND responds correctly to
        a getParameters probe.  The same socket is kept open for the episode.

        DARTSim sometimes accepts the TCP connection before its simulation is
        fully initialized, then closes it immediately.  We therefore connect
        and send a getParameters command; only if that succeeds do we consider
        DARTSim ready.
        """
        deadline = time.time() + self._connect_timeout
        last_err = None
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            try:
                self._client.connect()
                # Verify DARTSim is fully up by probing the protocol
                raw = self._client.send_command("getParameters")
                # If we get a valid JSON back, the session is healthy
                json.loads(raw)
                logger.debug("DARTSim ready after %d attempt(s)", attempts)
                # Store the response for use in reset()
                self._client._pending_params = raw
                return
            except Exception as e:
                last_err = e
                self._client.close()
                time.sleep(0.4)
        raise TimeoutError(
            f"DARTSim did not become ready on {self._host}:{self._port} "
            f"within {self._connect_timeout}s — last error: {last_err}"
        )

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_obs(self, state: Dict, threats: List[bool], targets: List[bool]) -> np.ndarray:
        """
        Build the 17-dimensional observation vector:
          [pos_x, pos_y, altitude, formation, ecm, dir_x, dir_y,
           threats[0..4], targets[0..4]]
        """
        cfg = state.get("config", {})
        pos = state.get("position", {})
        formation = 1.0 if cfg.get("formation", "LOOSE") == "TIGHT" else 0.0
        ecm = 1.0 if cfg.get("ecm", False) else 0.0

        # Pad/truncate sensor arrays to exactly sensor_horizon values
        def _pad(lst: List[bool], n: int) -> List[float]:
            lst = [float(v) for v in lst]
            if len(lst) < n:
                lst += [0.0] * (n - len(lst))
            return lst[:n]

        obs = np.array(
            [
                float(pos.get("x", 0)),
                float(pos.get("y", 0)),
                float(cfg.get("altitudeLevel", 1)),
                formation,
                ecm,
                float(state.get("directionX", 1)),
                float(state.get("directionY", 0)),
            ]
            + _pad(threats, self._sensor_horizon)
            + _pad(targets, self._sensor_horizon),
            dtype=np.float32,
        )
        return obs

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Disconnect from any previous episode
        self._client.close()

        # Optionally embed seed into sim_args
        extra = ""
        if seed is not None:
            extra = f"--seed={seed}"

        # Restart DARTSim for a fresh episode
        if self._container:
            self._kill_dartsim()
            stored = self._sim_args
            if extra:
                self._sim_args = f"{stored} {extra}".strip()
            self._start_dartsim()
            self._sim_args = stored  # restore

        # Wait for DARTSim to be ready and connect
        # _wait_for_dartsim also sends getParameters as a handshake probe
        self._wait_for_dartsim()

        # Use the params already fetched by the readiness probe
        try:
            pending = getattr(self._client, "_pending_params", None)
            self._params = json.loads(pending) if pending else {}
        except Exception:
            self._params = {}

        # Read initial state
        state = self._client.get_state()
        threats = self._client.read_forward_threat_sensor(self._sensor_horizon)
        targets = self._client.read_forward_target_sensor(self._sensor_horizon)

        self._ep_steps = 0
        self._last_obs = self._build_obs(state, threats, targets)

        return self._last_obs.copy(), {"params": self._params}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        tactic = ACTIONS[int(action)]
        try:
            return self._step_inner(tactic)
        except (ConnectionError, OSError, BrokenPipeError) as exc:
            # DARTSim closed the connection unexpectedly — treat as truncation
            # so SB3 auto-resets rather than crashing the whole training run.
            logger.warning("DARTSim connection lost during step (%s) — truncating episode", exc)
            self._client.close()
            return self._last_obs.copy(), 0.0, False, True, {"tactic": tactic, "error": str(exc)}

    def _safe_tactic(self, tactic: str) -> str:
        """
        Shield altitude tactics that would drive altitudeLevel below 1 or
        above altitudeLevels.  DARTSim has no bounds-checking in executeTactic;
        an out-of-range altitude causes a buffer overwrite in the screen array
        (``screen[pos][altitudeLevel - 1]``) and crashes the process.

        When the requested tactic would be unsafe we substitute ``GoTight``
        (formation change — no altitude effect) so the simulation step still
        executes and the agent receives a real reward signal.
        """
        altitude = int(round(float(self._last_obs[2])))
        alt_max = int(self._params.get("altitudeLevels", 4))

        if tactic == "DecAlt" and altitude <= 1:
            return "GoTight"
        if tactic == "DecAlt2" and altitude <= 2:
            return "GoTight"
        if tactic == "IncAlt" and altitude >= alt_max:
            return "GoLoose"
        if tactic == "IncAlt2" and altitude >= alt_max - 1:
            return "GoLoose"
        return tactic

    def _step_inner(self, tactic: str) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Shield out-of-bounds altitude tactics before sending to DARTSim
        safe_tactic = self._safe_tactic(tactic)
        # Send tactic; return value is "target detected this step" (not done flag)
        self._client.step([safe_tactic], 0.0)
        self._ep_steps += 1

        # Check termination via the dedicated finished() command
        terminated = self._client.finished()
        truncated = False

        if not terminated:
            # Get next observation
            state = self._client.get_state()
            threats = self._client.read_forward_threat_sensor(self._sensor_horizon)
            targets = self._client.read_forward_target_sensor(self._sensor_horizon)
            obs = self._build_obs(state, threats, targets)
            reward = self._reward_w["step_penalty"]
            info: Dict = {"tactic": tactic}
            if safe_tactic != tactic:
                info["shielded_to"] = safe_tactic
        else:
            # Episode over — fetch results
            obs = self._last_obs.copy()
            results = self._client.get_results()
            reward = self._compute_terminal_reward(results)
            info = {"tactic": tactic, "results": results}
            if safe_tactic != tactic:
                info["shielded_to"] = safe_tactic
            self._client.close()

        self._last_obs = obs
        return obs.copy(), reward, terminated, truncated, info

    def _compute_terminal_reward(self, results: Dict) -> float:
        r = 0.0
        if results.get("missionSuccess", False):
            r += self._reward_w["mission_success"]
        n_targets = results.get("targetsDetected", 0)
        if n_targets > 0:
            r += self._reward_w["targets_detected"] * min(n_targets / 10.0, 1.0)
        if results.get("destroyed", False):
            r += self._reward_w["destruction"]
        else:
            r += self._reward_w["survival"]
        return float(r)

    def render(self) -> None:
        pass  # no-op — use VNC on port 6901 for visual rendering

    def close(self) -> None:
        self._client.close()
        if self._container:
            self._kill_dartsim()
