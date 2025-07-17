# Filepath: marllib/envs/base_env/disaster.py

import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import newaxis as na
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union

policy_mapping_dict = {
    "test1": {
        "team_prefix": "agent_",
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    }
}


class DisasterEnv(MultiAgentEnv):
    """
    MARLlib-compatible UAV environment for multi-agent reinforcement learning.
    Features:
    - K SE-UAVs collecting data from M DSs
    - 1 SD-UAV for data relay
    - Automatic DS connection based on nearest valid DS
    - 3D continuous action space (direction, speed, hover time)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()

        # Extract config parameters
        self.env_config = env_config
        self.seed_val = env_config.get("seed", None)
        self.debug = env_config.get("debug", False)
        self.local_only = env_config.get("local_only", False)

        # Initialize core environment parameters
        self._init_core_params()

        # Set random seed
        if self.seed_val is not None:
            self.seed(self.seed_val)

        # Initialize environment state
        self._init_environment()

        # Define action and observation spaces
        self._init_spaces()

        # Initialize rendering attributes
        self._render_inited = False
        self.fig = None
        self.ax = None
        self.canvas = None
        self._trajs = [[] for _ in range(self.K)]

        # Initialize other attributes that may be set later
        self.conn = None
        self._snr_cached = None

    def _init_core_params(self):
        """Initialize core environment parameters"""
        # System parameters
        self.M = self.env_config.get("num_ds", 60)  # Number of DSs
        self.K = self.env_config.get("num_agents", 4)  # Number of SE-UAVs
        self.agents = [f"agent_{i}" for i in range(self.K)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.K))))

        self.image_num = 5  # Max images per DS per slot
        self.map_size = 3000.0  # Map size in meters
        center_xy = self.map_size / 2
        self.sd_pos = np.array([center_xy, center_xy, 50.0])  # SD-UAV position
        self.ds_z = 0.0  # DS height

        # Time parameters
        self.T = self.env_config.get("episode_limit", 1000)  # Total time slots
        self.delta_T = 3.0  # Time slot duration (s)

        # SE-UAV parameters
        self.v_se_max = 20.0  # Max speed (m/s)
        self.z_se = np.full(self.K, 50.0)  # SE-UAV heights

        # Communication parameters
        self._init_comm_params()

        # Reward parameters
        self.base_coeff = 2
        self.boundary_penalty = 10000.0
        self.penalty_no_conn_coeff = 0.1
        self.penalty_no_collect_coeff = 0.2
        self.trans_reward_coeff = 0.1

        # MCS table
        self.mcs_table = [
            (2, 1, 1 / 2), (5, 2, 1 / 2), (9, 2, 3 / 4), (11, 4, 1 / 2),
            (15, 4, 3 / 4), (18, 6, 2 / 3), (20, 6, 3 / 4), (25, 6, 5 / 6),
            (29, 8, 3 / 4), (31, 8, 5 / 6)
        ]

    def _init_comm_params(self):
        """Initialize communication parameters"""
        # SE-SD link parameters
        self.W_se = 23.00
        self.gamma_se_sd = 2.0
        self.f_se_sd = 9e8
        self.B_se_sd = 1e6
        self.F_se_sd = 3.0
        self.G_tx_se_sd = 7.0
        self.G_rx_se_sd = 11.0
        self.sigma_sh = 0.5
        self.N_SD_U_se_sd = 24
        self.T_DFT_se_sd = 32e-6
        self.T_GI_se_sd = 8e-6

        # SE-DS link parameters
        self.W_ds = 20.0
        self.gamma_se_ds_los = 2.0
        self.gamma_se_ds_nlos = 3.0
        self.f_se_ds = 2.4e9
        self.B_se_ds = 20e6
        self.F_se_ds = 3.0
        self.G_tx_se_ds = 3.0
        self.G_rx_se_ds = 9.0
        self.C_los = 1.0
        self.C_nlos = 20.0
        self.N_SD_U_se_ds = 52
        self.T_DFT_se_ds = 3.2e-6
        self.T_GI_se_ds = 0.8e-6
        self.a = 4.88
        self.b = 0.43

        # Common parameters
        self.N_ss = 1
        self.cmp_ratio = 1 / 192.0
        self.S_img = 3000
        self.RS_max = 100
        self.QoE_td = 0.4
        self.op_fixed = 100
        self.snr_td = 0
        self.tao_min = 1.0
        self.tao_max = 3.0

    def _init_environment(self):
        """Initialize environment state"""
        # Generate cluster centers
        self._generate_cluster_centers()

        # Generate DS positions
        ds_xy = self._gen_ds_positions_by_cluster()
        z_arr = np.full((self.M, 1), float(self.ds_z), dtype=np.float32)
        self.ds_pos = np.hstack([ds_xy, z_arr])

        # Assign DS to clusters
        d2 = np.linalg.norm(
            self.ds_pos[:, None, :2] - self.cluster_centers[None, :, :],
            axis=2
        )
        self.ds_to_cluster = np.argmin(d2, axis=1)

        # Initialize state variables
        self.P_se = np.zeros((self.K, 3), dtype=np.float32)
        self.tx_count = np.zeros(self.M, dtype=int)
        self.Q_in = [deque() for _ in range(self.K)]
        self.Q_out = [deque() for _ in range(self.K)]
        self.RS_avail = np.full(self.K, self.RS_max)
        self.finish_slots = [[] for _ in range(self.K)]
        self.t = 0
        self.agent_tx_count = np.zeros(self.K, dtype=int)

    def _generate_cluster_centers(self):
        """Generate K cluster centers in sectors"""
        angles = np.linspace(0, 2 * np.pi, self.K + 1)
        min_dist = 1200.0
        self.cluster_centers = np.zeros((self.K, 2), dtype=np.float32)

        for k in range(self.K):
            low, high = angles[k], angles[k + 1]
            for _ in range(10):
                theta = self.rng.uniform(low, high)
                r = self.rng.uniform(600.0, 800.0)
                x = self.sd_pos[0] + r * np.cos(theta)
                y = self.sd_pos[1] + r * np.sin(theta)
                if k == 0 or np.min(np.linalg.norm(self.cluster_centers[:k] - np.array([x, y]), axis=1)) >= min_dist:
                    self.cluster_centers[k] = [x, y]
                    break
            else:
                mid_theta = 0.5 * (low + high)
                self.cluster_centers[k] = [
                    self.sd_pos[0] + 800.0 * np.cos(mid_theta),
                    self.sd_pos[1] + 800.0 * np.sin(mid_theta)
                ]

    def _gen_ds_positions_by_cluster(self) -> np.ndarray:
        """Generate DS positions around cluster centers"""
        M, K = self.M, self.K
        centers = self.cluster_centers
        base, rem = divmod(M, K)
        counts = [base + (1 if i < rem else 0) for i in range(K)]

        sigma = 120.0
        all_pts = []
        for i, c in enumerate(centers):
            n = counts[i]
            pts = self.rng.normal(loc=c, scale=sigma, size=(n, 2))
            pts[:, 0] = np.clip(pts[:, 0], 0, self.map_size)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.map_size)
            all_pts.append(pts)
        all_pts = np.vstack(all_pts)
        return all_pts[:M]

    def _init_spaces(self):
        """Initialize action and observation spaces"""
        # Action space: 3D continuous (direction, speed, hover_time)
        act_dim = 3
        single_obs_dim = 3 * self.K + 3 + 3 * self.M if not self.local_only else (3 + 3 + 3 * self.M)
        core_box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(single_obs_dim,),
            dtype=np.float32
        )

        # Wrap into a Dict so RLlib and MARLlib build_model 能够找到 obs 子空间
        self.observation_space = spaces.Dict({
            "obs": core_box
        })

        # 同理共享观测
        self.share_observation_space = spaces.Dict({
            "obs": core_box
        })

        self.action_space = spaces.Box(
            low=np.zeros(act_dim),
            high=np.ones(act_dim),
            dtype=np.float32
        )

        # 标记已经是 preferred 格式
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True

        # RLlib 多 agent 要用到
        self._agent_ids = list(range(self.K))

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.t = 0
        self.agent_tx_count[:] = 0
        self.tx_count[:] = 0

        # Place SE-UAVs at cluster centers
        for k in range(self.K):
            cx, cy = self.cluster_centers[k]
            self.P_se[k] = np.array([cx, cy, self.z_se[k]])

        # Reinitialize queues and resources
        self.Q_in = [deque() for _ in range(self.K)]
        self.Q_out = [deque() for _ in range(self.K)]
        self.RS_avail[:] = self.RS_max
        self.finish_slots = [[] for _ in range(self.K)]

        # Reset trajectories
        for k in range(self.K):
            x0, y0 = self.P_se[k, :2]
            self._trajs[k] = [(x0, y0)]

        # Update SNR cache
        self._update_snr_cache()

        # Get observations
        obs = self._get_obs()

        # Return observations in MARLlib format
        return {agent: {"obs": obs.copy()} for agent in self.agents}

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one environment step"""
        # Convert action dict to list
        actions = [action_dict.get(agent, np.zeros(3)) for agent in self.agents]

        # Process actions
        active = self.t % self.K
        current_time = self.t * self.delta_T
        req_images = self.rng.randint(2, self.image_num + 1, size=self.M)

        # Extract action components
        phis = np.array([a[0] * 2 * np.pi for a in actions])
        speeds = np.array([a[1] * self.v_se_max for a in actions])
        hov_times = np.array([a[2] * self.delta_T for a in actions])
        fly_times = np.clip(self.delta_T - hov_times, 0.0, None)

        # Update positions
        dx = speeds * fly_times * np.sin(phis)
        dy = speeds * fly_times * np.cos(phis)
        self.P_se[:, 0] += dx
        self.P_se[:, 1] += dy

        # Update trajectories
        for k in range(self.K):
            x, y = self.P_se[k, :2]
            self._trajs[k].append((x, y))

        # Check boundaries
        out_x = (self.P_se[:, 0] < 0) | (self.P_se[:, 0] > self.map_size)
        out_y = (self.P_se[:, 1] < 0) | (self.P_se[:, 1] > self.map_size)

        if (out_x | out_y).any():
            obs = self._get_obs()
            dones = {agent: True for agent in self.agents}
            dones["__all__"] = True
            return (
                {agent: {"obs": obs.copy()} for agent in self.agents},
                {agent: -self.boundary_penalty for agent in self.agents},
                dones,  # 使用修正后的 dones
                {agent: {'boundary_violation': True} for agent in self.agents}
            )

        # Update SNR and connections
        self._update_snr_cache()
        conn = self._auto_connect_ds()
        self.conn = conn.copy()

        # Release encoding resources
        self._release_encoding_resources()

        # Collection phase
        D_collect = self._collection_phase(conn, hov_times, req_images)

        # Encoding phase
        self._encoding_phase()

        # Compute rewards
        rewards = self._compute_rewards(conn, D_collect)

        # Update time
        self.t += 1
        done = self.t >= self.T

        # Get observations
        obs = self._get_obs()

        # ✅ 构造正确的 dones 字典
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Return in MARLlib format
        return (
            {agent: {"obs": obs.copy()} for agent in self.agents},
            {agent: rewards[i] for i, agent in enumerate(self.agents)},
            dones,  # 使用修正后的 dones
            {agent: {} for agent in self.agents}
        )

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([
            self.P_se.flatten(),  # shape: (3K,)
            self.sd_pos,  # shape: (3,)
            self.ds_pos.flatten()  # shape: (3M,)
        ]).astype(np.float32)
        return obs

    def _update_snr_cache(self):
        """Update SNR matrix cache"""
        self._snr_cached = self._snr_matrix_se_ds()

    def _snr_matrix_se_ds(self) -> np.ndarray:
        """Compute SNR matrix for all SE-DS pairs"""
        se_xyz = self.P_se[:, None, :]
        ds_xyz = self.ds_pos[None, :, :]
        L = np.linalg.norm(se_xyz - ds_xyz, axis=2)

        theta = np.arcsin(self.z_se[:, na] / np.clip(L, 1.0, None))
        P_los = 1.0 / (1.0 + self.a * np.exp(-self.b * (np.degrees(theta) - self.a)))

        PL_los, PL_nlos = self._path_loss_mat(L)
        PL = P_los * PL_los + (1.0 - P_los) * PL_nlos

        noise_db = -174.0 + 10.0 * np.log10(self.B_se_ds) + self.F_se_ds
        snr_mat = self.W_ds + self.G_tx_se_ds + self.G_rx_se_ds - PL - noise_db
        return snr_mat

    def _path_loss_mat(self, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute path loss matrices"""
        L_eff = np.clip(L, 1.0, None)
        fspl = 20.0 * np.log10(4.0 * np.pi * L_eff * self.f_se_ds / 3e8)
        shadow = self.rng.normal(0.0, self.sigma_sh, size=L.shape)
        PL_los = fspl + 10.0 * self.gamma_se_ds_los * np.log10(L_eff) + self.C_los + shadow
        PL_nlos = fspl + 10.0 * self.gamma_se_ds_nlos * np.log10(L_eff) + self.C_nlos + shadow
        return PL_los, PL_nlos

    def _auto_connect_ds(self) -> np.ndarray:
        """Automatically connect to nearest valid DS"""
        cluster_ok = (self.ds_to_cluster == np.arange(self.K)[:, na])
        snr_ok = (self._snr_cached >= self.snr_td)
        valid_mask = cluster_ok & snr_ok

        conn = np.zeros((self.K, self.M), dtype=int)
        se_xyz = self.P_se[:, None, :]
        ds_xyz = self.ds_pos[None, :, :]
        distances = np.linalg.norm(se_xyz - ds_xyz, axis=2)

        for k in range(self.K):
            valid_ds_mask = valid_mask[k]
            if valid_ds_mask.any():
                valid_indices = np.where(valid_ds_mask)[0]
                valid_distances = distances[k, valid_indices]
                nearest_idx = valid_indices[np.argmin(valid_distances)]
                conn[k, nearest_idx] = 1

        return conn

    def _release_encoding_resources(self):
        """Release encoding resources"""
        for k in range(self.K):
            ready = [pair for pair in self.finish_slots[k] if pair[0] == self.t]
            self.RS_avail[k] += len(ready)
            for (_, ds_id) in ready:
                self.Q_out[k].append(ds_id)
            self.finish_slots[k] = [pair for pair in self.finish_slots[k] if pair[0] > self.t]

    def _collection_phase(self, conn: np.ndarray, hov_times: np.ndarray, req_images: np.ndarray) -> np.ndarray:
        """Collection phase: SE-UAVs collect from DSs"""
        D_collect = np.zeros(self.K, dtype=int)

        for k in range(self.K):
            ds_idx = np.where(conn[k] == 1)[0]
            if ds_idx.size == 0:
                continue

            m = ds_idx[0]
            ds_xyz = self.ds_pos[m]
            L_se_ds = np.linalg.norm(self.P_se[k] - ds_xyz)
            snr = self._snr_cached[k, m]

            bps, cr = self._lookup_mcs(snr)
            rate = bps * cr * self.N_ss * self.N_SD_U_se_ds / (self.T_DFT_se_ds + self.T_GI_se_ds)
            can_recv = int((rate * hov_times[k]) // self.S_img)
            actual = min(int(req_images[m]), can_recv)
            D_collect[k] = actual

            for _ in range(actual):
                self.Q_in[k].append(m)

        return D_collect

    def _encoding_phase(self):
        """Encoding phase"""
        for k in range(self.K):
            pend = len(self.Q_in[k])
            can = min(int(self.RS_avail[k]), pend)
            if can > 0:
                self.RS_avail[k] -= can
                for _ in range(can):
                    ds_id = self.Q_in[k].popleft()
                    tao = self.rng.uniform(self.tao_min, self.tao_max)
                    tau = int(np.ceil(tao / self.delta_T))
                    slot_rel = self.t + tau
                    self.finish_slots[k].append((slot_rel, ds_id))

    def _transmission_phase(self, active: int, hov_times: np.ndarray) -> int:
        """Transmission phase: active SE-UAV transmits to SD-UAV"""
        k = active
        L_se_sd = float(np.linalg.norm(self.P_se[k] - self.sd_pos))
        snr = self._snr_se_sd(k, L_se_sd)

        D_trans = 0
        if snr >= self.snr_td:
            bps, cr = self._lookup_mcs(snr)
            rate = bps * cr * self.N_ss * self.N_SD_U_se_sd / (self.T_DFT_se_sd + self.T_GI_se_sd)
            if snr < 2.0:
                rate = 0.15e6
            can_tx = int((rate * hov_times[k]) // (self.S_img * self.cmp_ratio))
            pending = len(self.Q_out[k])
            actual = min(can_tx, pending)
            D_trans = actual

            for _ in range(actual):
                ds_id = self.Q_out[k].popleft()
                self.tx_count[ds_id] += 1

        return D_trans

    def _snr_se_sd(self, k: int, L_se_sd: float) -> float:
        """Compute SE-SD SNR"""
        PL = self._path_loss(L_se_sd, self.f_se_sd, self.gamma_se_sd, 0.0)
        noise_db = -174.0 + 10.0 * np.log10(self.B_se_sd) + self.F_se_sd
        return self.W_se + self.G_tx_se_sd + self.G_rx_se_sd - PL - noise_db

    def _path_loss(self, L: float, freq: float, gamma: float, C: float = 0.0) -> float:
        """Compute path loss"""
        L_eff = max(L, 1.0)
        fspl = 20.0 * np.log10(4.0 * np.pi * L_eff * freq / 3e8)
        shadow = self.rng.normal(0.0, self.sigma_sh)
        return fspl + 10.0 * gamma * np.log10(L_eff) + C + shadow

    def _lookup_mcs(self, snr: float) -> Tuple[int, float]:
        """Lookup MCS from SNR"""
        best = (1, 0.5)
        for thr, bits, rate in self.mcs_table:
            if snr >= thr:
                best = (bits, rate)
        return best

    def _compute_rewards(self, conn: np.ndarray, D_collect: np.ndarray) -> List[float]:
        """Compute rewards for all agents"""
        # Compute Jain fairness per agent
        per_agent_jain = np.zeros(self.K, dtype=float)
        for k in range(self.K):
            cluster_ds = np.where(self.ds_to_cluster == k)[0]
            counts = self.tx_count[cluster_ds]
            j = self._compute_jain(counts)
            per_agent_jain[k] = j if j is not None else 0.0

        # Compute individual rewards
        rewards = []
        for k in range(self.K):
            base_k = self.base_coeff * per_agent_jain[k]
            tr = self.trans_reward_coeff * len(self.Q_out[k])
            nc = 0 if conn[k].any() else self.penalty_no_conn_coeff
            ncol = 0 if D_collect[k] > 0 else self.penalty_no_collect_coeff
            r = base_k + tr - nc - ncol
            rewards.append(r)

        return rewards

    @staticmethod
    def _compute_jain(x: Union[List, np.ndarray]) -> Optional[float]:
        """Compute Jain fairness index"""
        x_arr = np.array(x, dtype=float)
        total = x_arr.sum()
        if total == 0:
            return None
        denom = float(np.sum(x_arr ** 2))
        n = len(x_arr)
        return (total ** 2) / (n * denom)

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
        if not self._render_inited:
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.canvas = FigureCanvas(self.fig)
            self._render_inited = True

        self.ax.clear()

        # Draw DSs
        ds_xy = self.ds_pos[:, :2]
        conn_mask = self.conn.any(axis=0) if self.conn is not None else np.zeros(self.M, dtype=bool)
        self.ax.scatter(ds_xy[~conn_mask, 0], ds_xy[~conn_mask, 1],
                        c='blue', s=20, label='DS (unconnected)')
        self.ax.scatter(ds_xy[conn_mask, 0], ds_xy[conn_mask, 1],
                        c='orange', s=30, marker='s', label='DS (connected)')

        # Draw UAVs
        se_xy = self.P_se[:, :2]
        sd_xy = self.sd_pos[:2]
        self.ax.scatter(se_xy[:, 0], se_xy[:, 1], c='red', s=50, label='SE-UAV')
        self.ax.scatter(sd_xy[0], sd_xy[1], c='green', s=80, marker='^', label='SD-UAV')

        # Draw trajectories
        for k in range(self.K):
            if len(self._trajs[k]) > 1:
                xs, ys = zip(*self._trajs[k])
                self.ax.plot(xs, ys, color='green', linewidth=1, alpha=0.5)

        # Setup plot
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(f"Timestep {self.t}/{self.T}")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)

        if mode == "human":
            plt.pause(0.01)
            return None
        elif mode == "rgb_array":
            self.canvas.draw()
            buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            w, h = int(width), int(height)
            return buf.reshape(h, w, 3)
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self):
        """Close environment and cleanup"""
        plt.ioff()
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.canvas = None
        self._render_inited = False

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed"""
        self.rng = np.random.RandomState(seed)
        return [seed] if seed is not None else []

    def get_env_info(self) -> Dict[str, Any]:
        """Get environment information for MARLlib"""
        map_name = self.env_config.get("map_name", "default")
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "space_share": self.share_observation_space,
            "num_agents": self.K,
            "episode_limit": self.T,
            "policy_mapping_info": {
                map_name: {
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": True,
                }
            }
        }
        return env_info