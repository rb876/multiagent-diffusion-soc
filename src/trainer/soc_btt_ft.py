import torch
from torch.nn import functional as F

def train_control_btt(
    score_model,
    classifier,
    control_net_1,
    control_net_2,
    marginal_prob_std,
    diffusion_coeff,
    optimizer,
    target_digit=0,
    num_steps=100,
    batch_size=64,
    device='cuda',
    eps=1e-3,
    lambda_reg=0.1,
    running_class_reg=1.0,
):
    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    def get_std(time_tensor):
        return marginal_prob_std(time_tensor)[:, None, None, None]

    init_x1 = torch.randn(batch_size, 1, 28, 28, device=device) * get_std(t)
    init_x2 = torch.randn(batch_size, 1, 28, 28, device=device) * get_std(t)
    
    x1 = init_x1
    x2 = init_x2
    
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
        
        g = diffusion_coeff(batch_time_step)
        g_sq = (g**2)[:, None, None, None]
        g_noise = g[:, None, None, None]
        
        Y_t = (x1 * mask_top) + (x2 * mask_bot)
        
        u1 = control_net_1(torch.cat([x1, Y_t], dim=1), batch_time_step)
        u2 = control_net_2(torch.cat([x2, Y_t], dim=1), batch_time_step)

        # Classification loss computation on Tweedy estimates
        current_std = get_std(batch_time_step)        

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
                 
    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(control_net_1.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(control_net_2.parameters(), 1.0)

    optimizer.step()
    
    return total_loss.item()