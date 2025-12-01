from torch.nn import functional as F
import torch

from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torch

# PARAMETERS
NUM_STEPS = 500
BATCH_SIZE = 16
ITERS = 1000
LAMBDA_REG = 0.01

def train_control_policy(
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
    lambda_reg=0.1,      # Cost of using control
    running_class_reg=1.0 # New: Weight for the running classifier loss
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
    
    # Overlap logic
    mid = 14
    start_overlap = mid - (4 // 2) # 12
    end_overlap   = mid + (4 // 2) # 16

    mask_top = torch.zeros((1, 1, 28, 28), device=device)
    mask_top[:, :, :end_overlap, :] = 1.0
    
    mask_bot = torch.zeros((1, 1, 28, 28), device=device)
    mask_bot[:, :, start_overlap:, :] = 1.0

    cumulative_control_loss = 0
    cumulative_class_loss = 0 # Track running classification loss

    for time_step in range(len(time_steps)):
        batch_time_step = torch.ones(batch_size, device=device) * time_steps[time_step]
        
        g = diffusion_coeff(batch_time_step)
        g_sq = (g**2)[:, None, None, None]
        g_noise = g[:, None, None, None]
        
        # 1. Current Noisy Aggregate (for Control Input)
        Y_t = (x1 * mask_top) + (x2 * mask_bot)
        
        # 2. Get Controls
        u1 = control_net_1(torch.cat([x1, Y_t], dim=1), batch_time_step)
        u2 = control_net_2(torch.cat([x2, Y_t], dim=1), batch_time_step)

        # --- Running Classifier Guidance (Tweedie's Formula) ---
        current_std = get_std(batch_time_step)
        
        # Get scores (re-used for drift later, but needed raw here)
        s1 = score_model(x1, batch_time_step)
        s2 = score_model(x2, batch_time_step)
        
        # Tweedie's Estimate: x0 = xt + sigma^2 * score
        x1_0_hat = x1 + (current_std ** 2) * s1
        x2_0_hat = x2 + (current_std ** 2) * s2
        
        # Stitch the ESTIMATED CLEAN images
        Y_0_hat = (x1_0_hat * mask_top) + (x2_0_hat * mask_bot)
        
        # Blur the estimate slightly to remove artifacts before classifying
        # Y_0_hat_blur = TF.gaussian_blur(Y_0_hat, kernel_size=3)
        
        logits_running = classifier(Y_0_hat)
        target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
        
        # Add to cumulative loss (scaled by step_size)
        running_loss = F.cross_entropy(logits_running, target_labels)
        cumulative_class_loss += running_loss * step_size

        # 3. Physics Updates
        # Note: drift = g^2 * score
        drift1 = g_sq * s1 
        mean_x1 = x1 + (drift1 + u1) * step_size 
        x1 = mean_x1 + torch.sqrt(step_size) * g_noise * torch.randn_like(x1)
        
        drift2 = g_sq * s2 
        mean_x2 = x2 + (drift2 + u2) * step_size
        x2 = mean_x2 + torch.sqrt(step_size) * g_noise * torch.randn_like(x2)
                
        cumulative_control_loss += torch.mean(u1**2) * step_size
        cumulative_control_loss += torch.mean(u2**2) * step_size

    # Final Classification (Terminal Cost)
    Y_final = (x1 * mask_top) + (x2 * mask_bot)
    # blur_kernel = 3
    # Y_blurred = TF.gaussian_blur(Y_final, kernel_size=blur_kernel)
    logits = classifier(Y_final)
    target_labels = torch.full((batch_size,), target_digit, device=device, dtype=torch.long)
    class_loss = F.cross_entropy(logits, target_labels)

    # Total Loss = Control Cost + Terminal Class + Running Class
    total_loss = (lambda_reg * cumulative_control_loss) + \
                 class_loss + \
                 (running_class_reg * cumulative_class_loss)
                 
    optimizer.zero_grad()
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(control_net_1.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(control_net_2.parameters(), 1.0)
    optimizer.step()
    
    return total_loss.item()

optimizer = torch.optim.Adam(list(control_net_1.parameters()) + list(control_net_2.parameters()), lr=1e-4)

from tqdm.auto import tqdm

pbar = tqdm(range(ITERS), desc="Training control policy")
for epoch in pbar:
    loss = train_control_policy(
        score_model,
        classifier,
        control_net_1,
        control_net_2,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        optimizer,
        target_digit=6,
        num_steps=25,
        batch_size=BATCH_SIZE,
        device=device,
        eps=1e-3,
        lambda_reg=LAMBDA_REG,
    )
    pbar.set_postfix(loss=f"{loss:.4f}")
    if epoch % 100 == 0:
        generate_and_plot_samples(
			score_model,
			control_net_1,
			control_net_2,
			classifier,
			marginal_prob_std_fn,
			diffusion_coeff_fn,
			sample_batch_size=64,
			num_steps=500,
			device='cuda'
		)


# PARAMETERS
NUM_STEPS = 500
BATCH_SIZE = 16
OUTER_ITERS = 300
INNER_ITERS = 5
LAMBDA_REG = 0.01

def train_control_policy(
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
    lambda_reg=0.1,      # Cost of using control
    running_class_reg=1.0, # New: Weight for the running classifier loss
    player_idx=0,
):   
    """Train one control policy given the other is fixed."""
    optimizer.zero_grad()

    t = torch.ones(batch_size, device=device)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    def get_std(time_tensor):
        return marginal_prob_std(time_tensor)[:, None, None, None]

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
        
        g = diffusion_coeff(batch_time_step)
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


opt1 = torch.optim.Adam(control_net_1.parameters(), lr=1e-4)
opt2 = torch.optim.Adam(control_net_2.parameters(), lr=1e-4)

print("Starting training...")
from tqdm.auto import tqdm

pbar = tqdm(range(OUTER_ITERS), desc="Training control policy")
for epoch in pbar:
    # ----- update player 0 given player 1 fixed -----
    for i in range(INNER_ITERS):
        loss1 = train_control_policy(
            score_model,
            classifier,
            control_net_1,
            control_net_2,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            opt1,
            target_digit=0,
            num_steps=25,
            batch_size=BATCH_SIZE,
            device=device,
            eps=1e-3,
            lambda_reg=LAMBDA_REG,
            player_idx=0,
        )
    # ----- update player 1 given player 0 fixed -----
    for i in range(INNER_ITERS):
        loss2 = train_control_policy(
            score_model,
            classifier,
            control_net_1,
            control_net_2,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            opt2,
            target_digit=0,
            num_steps=25,
            batch_size=BATCH_SIZE,
            device=device,
            eps=1e-3,
            lambda_reg=LAMBDA_REG,
            player_idx=1,
        )
    pbar.set_postfix(loss=f"{loss1.item():.4f}, {loss2.item():.4f}")
    if epoch % 100 == 0:
        generate_and_plot_samples(
            score_model,
            control_net_1,
            control_net_2,
            classifier,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            sample_batch_size=64,
            num_steps=500,
            device='cuda'
        )

print("\n--- Training Complete ---")
