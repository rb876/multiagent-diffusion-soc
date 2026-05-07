# Multi-Agent Diffusion SOC

This repository trains and evaluates multi-agent stochastic optimal control policies for guided MNIST diffusion sampling. Experiments are configured with Hydra under `configs/`, with runnable workflow entrypoints under `workflows/`.

## Layout

- `workflows/learning_agent_joint.py`: joint control-agent fine-tuning entrypoint.
- `workflows/learning_agent_control_wise.py`: control-wise fine-tuning entrypoint.
- `configs/exps/`: experiment configs.
- `checkpoints/`: expected classifier and score-model checkpoints.
- `scripts/`: convenience scripts for MNIST experiment sweeps.
- `viz/`: notebooks and helpers for run lookup and evaluation.

## Requirements

Use the project virtual environment if available:

```bash
source venv/bin/activate
```

This repository uses `venv/` as its default virtual environment directory, not `.venv/`.

The default configs expect these checkpoint files to exist:

```text
checkpoints/cnet.pt
checkpoints/vp/latest.ckpt
```

## Run Individual Experiments

Run from the repository root. Set `CUDA_VISIBLE_DEVICES` to choose the GPU.

Joint BPTT training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m workflows.learning_agent_joint --config-path ../configs --config-name exps/bptt_learning_agents_fine_tuning
```

Control-wise BPTT training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m workflows.learning_agent_control_wise --config-path ../configs --config-name exps/control_wise_bptt_learning_agents_fine_tuning
```

Equivalent command using the active `python`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m workflows.learning_agent_control_wise --config-path ../configs --config-name exps/control_wise_bptt_learning_agents_fine_tuning
```

Useful Hydra overrides:

```bash
exps.soc.optimality_target=3
exps.soc.batch_size=8
exps.soc.num_control_agents=3
exps.soc.learning_rate=1e-4
exps.sde.name=VP
```

Example with overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python -m workflows.learning_agent_joint --config-path ../configs --config-name exps/bptt_learning_agents_fine_tuning exps.soc.optimality_target=3 exps.soc.batch_size=8 exps.soc.num_control_agents=3
```

## Convenience Scripts

Run both joint and control-wise MNIST BPTT workflows:

```bash
./scripts/run_mnist.sh --agents 2
```

Run the evaluation sweep:

```bash
./scripts/run_eval_on_mnist.sh --agents 3 --digits-list "9 3 0"
```

Run the experiment and ablation suite:

```bash
./scripts/run_mnist_exps_and_ablations.sh
```

## Run on a Slurm Cluster

Submit the MNIST eval sweep as a Slurm job:

```bash
./scripts/submit_eval_on_mnist_slurm.sh \
  --partition gpu \
  --time 08:00:00 \
  --mem 32G \
  --cpus-per-task 4 \
  -- --agents 3 --digits-list "9 3 0"
```

Notes:

- Logs are written to `logs/slurm/`.
- Add `--account ...` or `--qos ...` if your cluster requires them.
- The job wrapper activates `venv/` automatically when present.
- If your environment lives somewhere else, pass `--venv /path/to/venv` when submitting.

## Notes

The workflow modules were renamed for clarity. Use:

- `workflows.learning_agent_joint`
- `workflows.learning_agent_control_wise`

Older names such as `workflows.learning_agents_bptt` and `workflows.learning_agents_soc_control_wise` are not the current entrypoints.
