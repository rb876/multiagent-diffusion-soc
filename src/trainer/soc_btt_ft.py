import torch
from torch.nn import functional as F

def train_control_btt(
    score_model,
    classifier,
    control_agents,
    aggregator,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    optimizer,
    target_digit,
    num_steps,
    batch_size,
    device,
    eps,
    lambda_reg,
    running_class_reg,
):
    """Single-step training of multiple control policies in BTT setting."""

    score_model.eval()
    classifier.eval()

    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for training.")

    for key in agent_keys:
        control_agents[key].train()

    optimizer.zero_grad()

    time_steps = torch.linspace(1.0, eps, num_steps, device=device)
    if len(time_steps) < 2:
        raise ValueError("num_steps must be at least 2 to compute a diffusion step.")
    step_size = time_steps[0] - time_steps[1]

    initial_time = torch.full((batch_size,), time_steps[0], device=device)
    initial_std = marginal_prob_std_fn(initial_time)[:, None, None, None]

    agent_states = {
        key: torch.randn(batch_size, 1, 28, 28, device=device) * initial_std
        for key in agent_keys
    }

    cumulative_control_loss = torch.tensor(0.0, device=device)
    cumulative_class_loss = torch.tensor(0.0, device=device)

    for t_idx in range(len(time_steps)):
        batch_time_step = torch.full((batch_size,), time_steps[t_idx], device=device)

        g = diffusion_coeff_fn(batch_time_step)
        g_sq = (g**2)[:, None, None, None]
        g_noise = g[:, None, None, None]

        Y_t = aggregator([agent_states[key] for key in agent_keys])

        controls = {}
        scores = {}
        x0_hats = {}
        for key in agent_keys:
            control_input = torch.cat([agent_states[key], Y_t], dim=1)
            controls[key] = control_agents[key](control_input, batch_time_step)
            scores[key] = score_model(agent_states[key], batch_time_step)

        current_std = marginal_prob_std_fn(batch_time_step)[:, None, None, None]
        for key in agent_keys:
            x0_hats[key] = agent_states[key] + (current_std**2) * scores[key]

        Y_0_hat = aggregator([x0_hats[key] for key in agent_keys])

        target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
        logits_running = classifier(Y_0_hat)
        running_loss = F.cross_entropy(logits_running, target_labels)
        cumulative_class_loss += running_loss * step_size

        noise_scale = torch.sqrt(step_size) * g_noise
        for key in agent_keys:
            drift = g_sq * scores[key]
            mean_state = agent_states[key] + (drift + controls[key]) * step_size
            agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])
            cumulative_control_loss += torch.mean(controls[key] ** 2) * step_size

    Y_final = aggregator([agent_states[key] for key in agent_keys])
    logits = classifier(Y_final)
    target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
    class_loss = F.cross_entropy(logits, target_labels)

    total_loss = (
        lambda_reg * cumulative_control_loss
        + class_loss
        + running_class_reg * cumulative_class_loss
    )

    total_loss.backward()
    for key in agent_keys:
        torch.nn.utils.clip_grad_norm_(control_agents[key].parameters(), 1.0)

    optimizer.step()
    return total_loss.item()


def fictitious_train_control_btt(
    score_model,
    classifier,
    control_agents,
    aggregator,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    target_digit,
    num_steps,
    inner_iters,
    batch_size=64,
    device="cuda",
    eps=1e-3,
    lambda_reg=0.1,
    running_class_reg=1.0,
    learning_rate=1e-4,
):
    """Fictitious-play style training of an arbitrary number of control policies."""

    agent_keys = sorted(control_agents.keys())
    if not agent_keys:
        raise ValueError("No control agents provided for fictitious training.")

    optimizers = {
        key: torch.optim.Adam(control_agents[key].parameters(), lr=learning_rate)
        for key in agent_keys
    }

    def _train_single_control_policy(player_key):
        for key in agent_keys:
            if key == player_key:
                control_agents[key].train()
            else:
                control_agents[key].eval()

        optimizer = optimizers[player_key]
        optimizer.zero_grad()

        time_steps = torch.linspace(1.0, eps, num_steps, device=device)
        if len(time_steps) < 2:
            raise ValueError("num_steps must be at least 2 to compute a diffusion step.")
        step_size = time_steps[0] - time_steps[1]

        initial_time = torch.full((batch_size,), time_steps[0], device=device)
        initial_std = marginal_prob_std_fn(initial_time)[:, None, None, None]

        agent_states = {
            key: torch.randn(batch_size, 1, 28, 28, device=device) * initial_std
            for key in agent_keys
        }

        cumulative_control_loss = torch.tensor(0.0, device=device)
        cumulative_class_loss = torch.tensor(0.0, device=device)

        for _, t_val in enumerate(time_steps):
            batch_time_step = torch.full((batch_size,), t_val, device=device)

            g = diffusion_coeff_fn(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            g_noise = g[:, None, None, None]

            Y_t = aggregator([agent_states[key] for key in agent_keys])

            controls = {}
            for key in agent_keys:
                control_input = torch.cat([agent_states[key], Y_t], dim=1)
                if key == player_key:
                    controls[key] = control_agents[key](control_input, batch_time_step)
                else:
                    with torch.no_grad():
                        controls[key] = control_agents[key](control_input, batch_time_step)

            scores = {}
            for key in agent_keys:
                scores[key] = score_model(agent_states[key], batch_time_step)

            current_std = marginal_prob_std_fn(batch_time_step)[:, None, None, None]
            x0_hats = {
                key: agent_states[key] + (current_std**2) * scores[key]
                for key in agent_keys
            }

            Y_0_hat = aggregator([x0_hats[key] for key in agent_keys])

            target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
            logits_running = classifier(Y_0_hat)
            running_loss = F.cross_entropy(logits_running, target_labels)
            cumulative_class_loss += running_loss * step_size

            noise_scale = torch.sqrt(step_size) * g_noise
            for key in agent_keys:
                drift = g_sq * scores[key]
                mean_state = agent_states[key] + (drift + controls[key]) * step_size
                if key == player_key:
                    agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])
                else:
                    with torch.no_grad():
                        agent_states[key] = mean_state + noise_scale * torch.randn_like(agent_states[key])

            cumulative_control_loss += torch.mean(controls[player_key] ** 2) * step_size

        Y_final = aggregator([agent_states[key] for key in agent_keys])
        target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
        logits = classifier(Y_final)
        class_loss = F.cross_entropy(logits, target_labels)

        total_loss = (
            lambda_reg * cumulative_control_loss
            + class_loss
            + running_class_reg * cumulative_class_loss
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(control_agents[player_key].parameters(), 1.0)
        optimizer.step()

        return total_loss.item()

    loss_dict = {}
    for key in agent_keys:
        loss_value = None
        for _ in range(inner_iters):
            loss_value = _train_single_control_policy(player_key=key)
        loss_dict[key] = loss_value

    return loss_dict
