import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., 6) 6D rotation representation -> (..., 3, 3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def apply_pose(points: torch.Tensor, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    points: (B,K,N,3), R: (B,K,3,3), t: (B,K,3), s: (B,K,1)
    """
    B, K, N, _ = points.shape
    p = points * s.view(B, K, 1, 1)
    p = torch.matmul(p, R.transpose(-1, -2)) + t.view(B, K, 1, 3)
    return p


# -----------------------------
# Differentiable renderer: soft point splatting
# -----------------------------
class SoftPointRenderer(nn.Module):
    """
    Simple differentiable renderer:
    projects 3D points onto image plane and splats them as Gaussians.
    Output is an "occupancy-like" image used for scene-level objectives / features.
    """
    def __init__(self, H=96, W=96, sigma_px=1.5, z_clip=(0.2, 10.0)):
        super().__init__()
        self.H, self.W = H, W
        self.sigma_px = sigma_px
        self.zmin, self.zmax = z_clip

        ys = torch.linspace(0, H - 1, H)
        xs = torch.linspace(0, W - 1, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", gx[None, None, :, :])  # (1,1,H,W)
        self.register_buffer("grid_y", gy[None, None, :, :])  # (1,1,H,W)

    def forward(self, points_world: torch.Tensor) -> torch.Tensor:
        """
        points_world: (B,K,N,3)
        output: (B,1,H,W)
        """
        B, K, N, _ = points_world.shape
        fx = fy = 60.0
        cx = (self.W - 1) / 2.0
        cy = (self.H - 1) / 2.0

        x = points_world[..., 0]
        y = points_world[..., 1]
        z = points_world[..., 2].clamp(min=1e-3)

        valid = (z > self.zmin) & (z < self.zmax)
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        u = u.view(B, K, N, 1, 1)
        v = v.view(B, K, N, 1, 1)
        valid = valid.view(B, K, N, 1, 1).float()

        dx = (self.grid_x - u) / self.sigma_px
        dy = (self.grid_y - v) / self.sigma_px
        g = torch.exp(-0.5 * (dx * dx + dy * dy)) * valid  # (B,K,N,H,W)

        img = g.sum(dim=2).sum(dim=1, keepdim=True)        # (B,1,H,W)
        img = 1.0 - torch.exp(-img)
        return img


# -----------------------------
# Scene coherence losses (prompt-agnostic)
# -----------------------------
class SceneCoherenceCriterion(nn.Module):
    """
    Reusable constraints for coherence:
      - collision avoidance
      - contact/support (no floating)
      - scale consistency
      - pose plausibility (upright)
    """
    def __init__(self, ground_y=0.0, radius=0.04):
        super().__init__()
        self.ground_y = ground_y
        self.radius = radius

    @staticmethod
    def collision_loss(points_world: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Soft collision penalty via point-to-point distances across objects.
        points_world: (B,K,N,3)
        """
        B, K, N, _ = points_world.shape
        loss = 0.0
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                d = torch.cdist(points_world[:, i], points_world[:, j])  # (B,N,N)
                pen = F.relu(2 * radius - d)
                loss = loss + (pen * pen).mean()
                pairs += 1
        return loss / max(pairs, 1)

    @staticmethod
    def support_no_floating_loss(points_world: torch.Tensor, ground_y: float, margin: float = 0.02) -> torch.Tensor:
        """
        Penalize if object's lowest point is above ground plane.
        """
        y = points_world[..., 1]  # (B,K,N)
        min_y = y.min(dim=-1).values  # (B,K)
        float_pen = F.relu(min_y - (ground_y + margin))
        return (float_pen * float_pen).mean()

    @staticmethod
    def upright_pose_loss(R: torch.Tensor) -> torch.Tensor:
        """
        Encourage local up-axis (col 1) to align with world up (0,1,0).
        """
        up = R[..., :, 1]  # (B,K,3)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=R.device).view(1, 1, 3)
        cos = (up * world_up).sum(dim=-1).clamp(-1, 1)  # (B,K)
        return (1.0 - cos).mean()

    @staticmethod
    def scale_consistency_loss(s: torch.Tensor, s_min=0.7, s_max=1.4) -> torch.Tensor:
        lo = F.relu(s_min - s)
        hi = F.relu(s - s_max)
        return (lo * lo + hi * hi).mean()

    def get_running_state_loss(self, scene_struct: dict, target=None) -> torch.Tensor:
        pts = scene_struct["points_world"]
        R = scene_struct["R"]
        s = scene_struct["s"]

        L = 0.0
        L = L + 5.0 * self.collision_loss(pts, self.radius)
        L = L + 2.0 * self.support_no_floating_loss(pts, self.ground_y)
        L = L + 0.5 * self.upright_pose_loss(R)
        L = L + 0.2 * self.scale_consistency_loss(s)
        return L

    def get_terminal_state_loss(self, scene_struct: dict, target=None) -> torch.Tensor:
        return self.get_running_state_loss(scene_struct, target)


# -----------------------------
# Pose controller: Theta_t = pi_theta(feat(X_t), t)
# -----------------------------
class PoseController(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_rot6d = nn.Linear(hidden, 6)
        self.head_t = nn.Linear(hidden, 3)
        self.head_log_s = nn.Linear(hidden, 1)

    def forward(self, feat_xyz: torch.Tensor, t: torch.Tensor):
        """
        feat_xyz: (B,K,3), t: (B,)
        """
        B, K, _ = feat_xyz.shape
        t_in = t.view(B, 1, 1).expand(B, K, 1)
        h = self.net(torch.cat([feat_xyz, t_in], dim=-1).reshape(B * K, 4))
        rot6d = self.head_rot6d(h).view(B, K, 6)
        trans = self.head_t(h).view(B, K, 3)
        scale = torch.exp(self.head_log_s(h)).view(B, K, 1)
        R = rot6d_to_matrix(rot6d)
        return R, trans, scale


# -----------------------------
# Aggregator: compose + render (nonlinear)
# -----------------------------
class SceneAggregator(nn.Module):
    def __init__(self, renderer: nn.Module, pose_controller: PoseController):
        super().__init__()
        self.renderer = renderer
        self.pose_controller = pose_controller

    def forward(self, agent_point_clouds, t_batch: torch.Tensor):
        """
        agent_point_clouds: list length K, each tensor (B,N,3)
        t_batch: (B,)
        """
        pts_local = torch.stack(agent_point_clouds, dim=1)  # (B,K,N,3)
        feat = pts_local.mean(dim=2)                        # (B,K,3) (minimal feature)
        R, trans, scale = self.pose_controller(feat, t_batch)

        pts_world = apply_pose(pts_local, R, trans, scale)
        img = self.renderer(pts_world)
        return {"img": img, "points_world": pts_world, "R": R, "t": trans, "s": scale}


# -----------------------------
# Point control net u_i: controls diffusion drift in point space
# -----------------------------
class PointControlNet(nn.Module):
    def __init__(self, in_ch=3 + 1 + 1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor, y_feat: torch.Tensor, g_feat: torch.Tensor):
        """
        x: (B,N,3)
        y_feat: (B,N,1)
        g_feat: (B,N,1)
        """
        inp = torch.cat([x, y_feat, g_feat], dim=-1)
        return self.net(inp)


# -----------------------------
# Point-E score model wrapper (FROZEN)
# -----------------------------
class PointEScoreModel(nn.Module):
    """
    Wrap a pretrained Point-E diffusion model as eps-predictor (score proxy).
    You MUST adapt this to Point-E API you are using.
    """
    def __init__(self, point_e_model: nn.Module):
        super().__init__()
        self.model = point_e_model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)


# -----------------------------
# Placeholder SDE + Tweedie (replace with your exact schedule)
# -----------------------------
class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def diffusion_coeff(self, t):
        return torch.sqrt(self.beta(t))

    def f(self, x, t):
        return -0.5 * self.beta(t)[:, None, None] * x

    def marginal_prob_std(self, t):
        return torch.sqrt(1.0 - torch.exp(-t))


def get_tweedie_estimate_vpsde(sde: VPSDE, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor):
    sigma = sde.marginal_prob_std(t).view(-1, 1, 1)
    return x_t - sigma * eps_hat


# -----------------------------
# SOC training step (Option B)
# -----------------------------
def train_control_bptt_point_e_optionB(
    aggregator: SceneAggregator,
    batch_size: int,
    control_agents: dict,            # key -> PointControlNet
    device: str,
    eps: float,
    num_steps: int,
    lambda_reg: float,
    running_state_cost_scaling: float,
    optimality_criterion: SceneCoherenceCriterion,
    score_models: dict,              # key -> PointEScoreModel (frozen)
    sde: VPSDE,
    optimizer: torch.optim.Optimizer,  # must include aggregator.pose_controller params too
    num_points: int,
    debug: bool = False,
    use_x0hat_for_pose_and_cost: bool = True,  # recommended
):
    # Modes
    for m in score_models.values():
        m.eval()
    optimality_criterion.eval()
    for m in control_agents.values():
        m.train()
    aggregator.train()  # trains pose_controller too

    optimizer.zero_grad(set_to_none=True)

    agent_keys = sorted(control_agents.keys())
    K = len(agent_keys)
    assert K > 0

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)

    # init states: point clouds X_t
    t0 = torch.full((batch_size,), time_steps[0], device=device)
    init_std = sde.marginal_prob_std(t0).view(-1, 1, 1)
    X = {k: torch.randn(batch_size, num_points, 3, device=device) * init_std for k in agent_keys}

    cumulative_control_loss = torch.tensor(0.0, device=device)
    cumulative_running_loss = torch.tensor(0.0, device=device)

    info = {}

    for t_idx in range(len(time_steps) - 1):
        t_cur = time_steps[t_idx]
        t_nxt = time_steps[t_idx + 1]
        dt = t_cur - t_nxt
        t_batch = torch.full((batch_size,), t_cur, device=device)

        g = sde.diffusion_coeff(t_batch)          # (B,)
        g_sq = (g**2).view(-1, 1, 1)              # (B,1,1)
        g_noise = g.view(-1, 1, 1)

        # scores + Tweedie
        eps_hat = {}
        x0_hat = {}
        for k in agent_keys:
            eps_hat[k] = score_models[k](X[k], t_batch)  # (B,N,3)
            if running_state_cost_scaling > 0 or use_x0hat_for_pose_and_cost:
                x0_hat[k] = get_tweedie_estimate_vpsde(sde, X[k], t_batch, eps_hat[k])

        # choose which clouds define the scene cost (recommended: x0_hat)
        clouds_for_scene = [x0_hat[k] for k in agent_keys] if use_x0hat_for_pose_and_cost else [X[k] for k in agent_keys]

        # running state cost
        if running_state_cost_scaling > 0:
            scene_struct_hat = aggregator(clouds_for_scene, t_batch)
            cumulative_running_loss += optimality_criterion.get_running_state_loss(scene_struct_hat) * dt

        # current scene for control features (minimal)
        scene_struct_xt = aggregator([X[k] for k in agent_keys], t_batch)
        y_scalar = scene_struct_xt["img"].mean(dim=[1, 2, 3], keepdim=True)  # (B,1,1,1)
        y_feat = y_scalar.view(batch_size, 1, 1).expand(batch_size, num_points, 1)

        # controls u_i
        U = {}
        for k in agent_keys:
            g_feat = torch.zeros(batch_size, num_points, 1, device=device)  # plug guidance grads if desired
            U[k] = control_agents[k](X[k], y_feat, g_feat)                  # (B,N,3)
            cumulative_control_loss += U[k].pow(2).mean() * dt / K

        # Euler–Maruyama reverse update (placeholder)
        noise_scale = torch.sqrt(dt) * g_noise
        for k in agent_keys:
            drift_rev = -sde.f(X[k], t_batch) + g_sq * eps_hat[k]
            mean = X[k] + (drift_rev + g_sq * U[k]) * dt
            X[k] = mean + noise_scale * torch.randn_like(X[k])

    # terminal cost (recommended: use x0_hat at final time; here use current X)
    t_final = torch.full((batch_size,), time_steps[-1], device=device)
    scene_final = aggregator([X[k] for k in agent_keys], t_final)
    terminal_loss = optimality_criterion.get_terminal_state_loss(scene_final)

    total_loss = lambda_reg * cumulative_control_loss + terminal_loss + running_state_cost_scaling * cumulative_running_loss
    total_loss.backward()

    for k in agent_keys:
        torch.nn.utils.clip_grad_norm_(control_agents[k].parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(aggregator.pose_controller.parameters(), 1.0)

    optimizer.step()

    if debug:
        info["terminal_loss"] = terminal_loss.item()
        info["cumulative_control_loss"] = cumulative_control_loss.item()
        info["cumulative_running_loss"] = cumulative_running_loss.item() if running_state_cost_scaling > 0 else 0.0

    return total_loss.item(), info


# -----------------------------
# Minimal runnable demo with dummy "Point-E" models
# (Replace DummyPointE with real Point-E loading.)
# -----------------------------
class DummyPointE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, x, t):
        # x: (B,N,3), t: (B,)
        tt = t.view(-1, 1, 1).expand(x.shape[0], x.shape[1], 1)
        return self.net(torch.cat([x, tt], dim=-1))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    batch_size = 4
    num_points = 1024
    prompts = ["a chair", "a ball"]  # conceptual; real Point-E would condition on these
    agent_keys = [f"agent_{i}" for i in range(len(prompts))]

    # Frozen score models (replace with real Point-E models)
    score_models = {}
    for k in agent_keys:
        m = DummyPointE().to(device)
        set_requires_grad(m, False)
        m.eval()
        score_models[k] = PointEScoreModel(m).to(device)

    # Trainable controls
    control_agents = {k: PointControlNet().to(device) for k in agent_keys}

    # Aggregator (trainable pose controller) + renderer
    renderer = SoftPointRenderer().to(device)
    pose_controller = PoseController().to(device)
    aggregator = SceneAggregator(renderer=renderer, pose_controller=pose_controller).to(device)

    # Coherence loss
    criterion = SceneCoherenceCriterion(ground_y=0.0, radius=0.04).to(device)

    # SDE
    sde = VPSDE()

    # Optimizer MUST include pose controller params
    optimizer = torch.optim.Adam(
        [p for m in control_agents.values() for p in m.parameters()] +
        list(aggregator.pose_controller.parameters()),
        lr=1e-3
    )

    # One training step
    loss, info = train_control_bptt_point_e_optionB(
        aggregator=aggregator,
        batch_size=batch_size,
        control_agents=control_agents,
        device=device,
        eps=0.02,
        num_steps=32,
        lambda_reg=0.1,
        running_state_cost_scaling=1.0,
        optimality_criterion=criterion,
        score_models=score_models,
        sde=sde,
        optimizer=optimizer,
        num_points=num_points,
        debug=True,
        use_x0hat_for_pose_and_cost=True,
    )
    print("loss:", loss, "info:", info)


if __name__ == "__main__":
    main()