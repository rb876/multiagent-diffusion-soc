import torch
from torch.nn import functional as F

def train_control_btt(
    score_model,
    classifier,
    control_agents,
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

    # Set models to correct modes
    score_model.eval()
    classifier.eval()
    for control_net in control_agents.values():
        control_net.train()
    
    # Set gradients to zero
    optimizer.zero_grad()

    # Precompute time steps and step size
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    init_x1 = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std_fn(time_steps[0])[:, None, None, None]
    init_x2 = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std_fn(time_steps[0])[:, None, None, None]
    
    x1 = init_x1
    x2 = init_x2
    
    # Set up overlap masks logic
    mid = 14
    start_overlap = mid - (4 // 2)
    end_overlap   = mid + (4 // 2)

    mask_top = torch.zeros((1, 1, 28, 28), device=device)
    mask_top[:, :, :end_overlap, :] = 1.0
    
    mask_bot = torch.zeros((1, 1, 28, 28), device=device)
    mask_bot[:, :, start_overlap:, :] = 1.0

    cumulative_control_loss = 0
    cumulative_class_loss = 0
    for time_step in range(len(time_steps)):

        batch_time_step = torch.ones(batch_size, device=device) * time_steps[time_step]
        g = diffusion_coeff_fn(batch_time_step)
        g_sq = (g**2)[:, None, None, None]
        g_noise = g[:, None, None, None]
        
        Y_t = (x1 * mask_top) + (x2 * mask_bot)
        
        u1 = control_agents[0](torch.cat([x1, Y_t], dim=1), batch_time_step)
        u2 = control_agents[1](torch.cat([x2, Y_t], dim=1), batch_time_step)

        # Classification loss computation on Tweedy estimates
        current_std = marginal_prob_std_fn(batch_time_step)[:, None, None, None]        
        s1 = score_model(x1, batch_time_step)
        s2 = score_model(x2, batch_time_step)                
        x1_0_hat = x1 + (current_std ** 2) * s1
        x2_0_hat = x2 + (current_std ** 2) * s2
        
        # Stitch back together
        Y_0_hat = (x1_0_hat * mask_top) + (x2_0_hat * mask_bot)
    
        # Get classifier predictions
        logits_running = classifier(Y_0_hat)
        target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
        
        # Add to cumulative loss (scaled by step_size)
        running_loss = F.cross_entropy(logits_running, target_labels)
        cumulative_class_loss += running_loss * step_size

        drift1 = g_sq * s1 
        mean_x1 = x1 + (drift1 + u1) * step_size 
        x1 = mean_x1 + torch.sqrt(step_size) * g_noise * torch.randn_like(x1)
        
        drift2 = g_sq * s2 
        mean_x2 = x2 + (drift2 + u2) * step_size
        x2 = mean_x2 + torch.sqrt(step_size) * g_noise * torch.randn_like(x2)
                
        cumulative_control_loss += torch.mean(u1**2) * step_size
        cumulative_control_loss += torch.mean(u2**2) * step_size

    Y_final = (x1 * mask_top) + (x2 * mask_bot)
    logits = classifier(Y_final)
    target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
    class_loss = F.cross_entropy(logits, target_labels)

    total_loss = (lambda_reg * cumulative_control_loss) + \
                 class_loss + \
                 (running_class_reg * cumulative_class_loss)
                 
    # Backpropagate the total loss
    total_loss.backward()

    for control_net in control_agents.values():
        torch.nn.utils.clip_grad_norm_(control_net.parameters(), 1.0)

    # Update control network parameters
    optimizer.step()
    return total_loss.item()


def fictitious_train_control_btt(
    score_model,
    classifier,
    control_net_1,
    control_net_2,
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
    """Fictitious Play training of two control policies in BTT setting."""

    def _train_single_control_policy(
        score_model,
        classifier,
        control_net_1,
        control_net_2,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        optimizer,
        target_digit,
        num_steps,
        batch_size,
        device="cuda",
        eps=1e-3,
        lambda_reg=0.1,
        running_class_reg=1.0,
        player_idx=0,
    ):
        """Train one control policy given the other is fixed."""

        score_model.eval()
        classifier.eval()
        optimizer.zero_grad()

        t = torch.ones(batch_size, device=device)
        time_steps = torch.linspace(1., eps, num_steps, device=device)
        step_size = time_steps[0] - time_steps[1]
        
        def get_std(time_tensor):
            return marginal_prob_std_fn(time_tensor)[:, None, None, None]

        init_x1 = torch.randn(batch_size, 1, 28, 28, device=device) * get_std(t)
        init_x2 = torch.randn(batch_size, 1, 28, 28, device=device) * get_std(t)
        
        x1 = init_x1
        x2 = init_x2
        
        # Overlap logic
        mid = 14
        start_overlap = mid - (4 // 2) # 12
        end_overlap   = mid + (4 // 2) # 16

        mask_top = torch.zeros((1, 1, 28, 28), device=device)
        mask_top[:, :, :end_overlap, :] = 1.0
        
        mask_bot = torch.zeros((1, 1, 28, 28), device=device)
        mask_bot[:, :, start_overlap:, :] = 1.0

        cumulative_control_loss = 0
        cumulative_class_loss = 0

        for time_step in range(len(time_steps)):
            batch_time_step = torch.ones(batch_size, device=device) * time_steps[time_step]
            
            g = diffusion_coeff_fn(batch_time_step)
            g_sq = (g**2)[:, None, None, None]
            g_noise = g[:, None, None, None]
            
            Y_t = (x1 * mask_top) + (x2 * mask_bot)
            
            if player_idx == 0:
                u1 = control_net_1(torch.cat([x1, Y_t], dim=1), batch_time_step)
                with torch.no_grad():
                    u2 = control_net_2(torch.cat([x2, Y_t], dim=1), batch_time_step)
            elif player_idx == 1:
                with torch.no_grad():
                    u1 = control_net_1(torch.cat([x1, Y_t], dim=1), batch_time_step)
                u2 = control_net_2(torch.cat([x2, Y_t], dim=1), batch_time_step)
            else:
                raise ValueError("player_idx must be 0 or 1 - Python indexing")

            current_std = get_std(batch_time_step)
            
            s1 = score_model(x1, batch_time_step)
            s2 = score_model(x2, batch_time_step)
            
            x1_0_hat = x1 + (current_std ** 2) * s1
            x2_0_hat = x2 + (current_std ** 2) * s2
            
            Y_0_hat = (x1_0_hat * mask_top) + (x2_0_hat * mask_bot)
            
            logits_running = classifier(Y_0_hat)
            target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
            
            running_loss = F.cross_entropy(logits_running, target_labels)
            cumulative_class_loss += running_loss * step_size

            drift1 = g_sq * s1 
            mean_x1 = x1 + (drift1 + u1) * step_size
            x1 = mean_x1 + torch.sqrt(step_size) * g_noise * torch.randn_like(x1)
            
            drift2 = g_sq * s2 
            mean_x2 = x2 + (drift2 + u2) * step_size
            x2 = mean_x2 + torch.sqrt(step_size) * g_noise * torch.randn_like(x2)
            if player_idx == 0:
                cumulative_control_loss += torch.mean(u1**2) * step_size
            elif player_idx == 1:
                cumulative_control_loss += torch.mean(u2**2) * step_size
            else: 
                raise ValueError("player_idx must be 0 or 1 - Python indexing")

        Y_final = (x1 * mask_top) + (x2 * mask_bot)
        logits = classifier(Y_final)
        target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
        class_loss = F.cross_entropy(logits, target_labels)

        total_loss = (lambda_reg * cumulative_control_loss) + \
                    class_loss + \
                    (running_class_reg * cumulative_class_loss)
                    
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(control_net_1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(control_net_2.parameters(), 1.0)
        optimizer.step()
        
        return total_loss
    
    # ----- set up optimizers for both control nets -----
    opt1 = torch.optim.Adam(control_net_1.parameters(), lr=learning_rate)
    opt2 = torch.optim.Adam(control_net_2.parameters(), lr=learning_rate)

    # ----- update player 0 given player 1 fixed -----
    for _ in range(inner_iters):
        loss1 = _train_single_control_policy(
            score_model,
            classifier,
            control_net_1,
            control_net_2,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            opt1,
            target_digit=target_digit,
            num_steps=num_steps,
            batch_size=batch_size,
            device=device,
            eps=eps,
            lambda_reg=lambda_reg,
            running_class_reg=running_class_reg,
            player_idx=0,
        )
    # ----- update player 1 given player 0 fixed -----
    for _ in range(inner_iters):
        loss2 = _train_single_control_policy(
            score_model,
            classifier,
            control_net_1,
            control_net_2,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            opt2,
            target_digit=target_digit,
            num_steps=num_steps,
            batch_size=batch_size,
            device=device,
            eps=eps,
            lambda_reg=lambda_reg,
            running_class_reg=running_class_reg,
            player_idx=1,
        )

    return loss1.item(), loss2.item()
