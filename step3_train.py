
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
import time
import json
from tqdm import tqdm
import pandas as pd
from scipy import stats

from step0_model  import create_hybrid_mlp_model
from utils_SIR import create_dataloaders, compute_metrics, get_device, EarlyStopping

n_timepoints=50

# REPRODUCIBILITY


def set_seed(seed):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

# BALANCED LOSS


def compute_balanced_loss(predictions, targets, device, weight_mode='balanced'):
    """
    Multi-component loss that prevents the Infected compartment from
    collapsing to zero (the 'mode collapse' problem near R₀ = 1).

    Components:
        loss_S, loss_I, loss_R  – normalised MSE per compartment
        loss_I_peak             – penalty for mis-predicting epidemic peak
        loss_I_mean             – penalty that prevents I from collapsing
        conservation_error      – soft S+I+R = N penalty

    Returns:
        total_loss, loss_S, loss_I, loss_R
    """
    S_pred = predictions[:, :, 0]
    I_pred = predictions[:, :, 1]
    R_pred = predictions[:, :, 2]

    S_true = targets[:, :, 0]
    I_true = targets[:, :, 1]
    R_true = targets[:, :, 2]

    N = 10000

   

   
    loss_S = ((S_pred - S_true)).pow(2).mean()
    loss_I = ((I_pred - I_true)).pow(2).mean()
    #loss_R = ((R_pred - R_true)).pow(2).mean()
    
    I_pred_peak = I_pred.max(dim=1)[0]
    I_true_peak = I_true.max(dim=1)[0]
    loss_I_peak = (I_pred_peak - I_true_peak).pow(2).mean()

    I_pred_mean = I_pred.mean(dim=1)
    I_true_mean = I_true.mean(dim=1)
    loss_I_mean = (I_pred_mean - I_true_mean).pow(2).mean()
   # conservation_error = ((S_pred + I_pred + R_pred - N) / N).pow(2).mean()

    total_loss = (
    15.0 * loss_S +
    15.0 * loss_I +
    #1.0 * loss_R +
    8.0 * loss_I_peak +
    3.0 * loss_I_mean 
   # 0.1 * conservation_error
)
    loss_R=0.0
    return total_loss, loss_S, loss_I, loss_R #, loss_R


# TRAIN / VALIDATE ONE EPOCH

def train_epoch_balanced(model, train_loader, optimizer, device, n_timesteps,
                         weight_mode='modest'):
    """One training epoch — no graph_stats, no dummy tensors."""
    model.train()

    total_loss = total_loss_S = total_loss_I = total_loss_R = 0.0
    all_predictions, all_targets = [], []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ── Forward pass (3-param model, no graph_stats) ─────────────────────
        predictions = model(batch, n_timesteps=n_timesteps)
        targets     = batch.y

        loss, loss_S, loss_I, loss_R = compute_balanced_loss(
            predictions, targets, device, weight_mode
        )
        loss_R=0.0 # R=N-S-I by conservation,
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss   += loss.item()
        total_loss_S += loss_S.item()
        total_loss_I += loss_I.item()
        total_loss_R += loss_R

        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets     = torch.cat(all_targets,     dim=0)
    metrics     = compute_metrics(predictions, targets)

    n_batches       = len(train_loader)
    metrics['loss_S'] = total_loss_S / n_batches
    metrics['loss_I'] = total_loss_I / n_batches
    metrics['loss_R'] = total_loss_R / n_batches

    return total_loss / n_batches, metrics


def validate_balanced(model, val_loader, device, n_timesteps, weight_mode='modest'):
    """One validation pass — no graph_stats."""
    model.eval()

    total_loss = 0.0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass 
            predictions = model(batch, n_timesteps=n_timesteps)
            targets     = batch.y

            loss, *_ = compute_balanced_loss(predictions, targets, device, weight_mode)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets     = torch.cat(all_targets,     dim=0)
    metrics     = compute_metrics(predictions, targets)

    return total_loss / len(val_loader), metrics


# SINGLE REPLICATE TRAINING


def train_single_replicate(
    replicate_id,
    seed,
    config,
    dataloaders,
    output_dir,
    weight_mode='modest',
    verbose=True,
):
    """
    Train one replicate with a given seed.

    Args:
        replicate_id : 1-indexed integer
        seed         : random seed for reproducibility
        config       : model + training hyperparameters
        dataloaders  : dict from create_dataloaders()
        output_dir   : directory where the .pt checkpoint is saved
        weight_mode  : loss weighting strategy
        verbose      : print progress every 10 epochs

    Returns:
        results : summary dict
        history : training curve arrays
    """
    set_seed(seed)

    device     = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_filename   = f'best_balanced_mlp_model_{replicate_id}.pt'
    model_path       = output_dir / model_filename
    history_filename = f'training_history_{replicate_id}.npy'
    history_path     = output_dir / history_filename

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    n_timesteps  = dataloaders['metadata']['n_timepoints']

    if verbose:
       
        print(f"REPLICATE {replicate_id}  (seed={seed})")
        print(f"Parameters: tau (τ), gamma (γ), rho (ρ)")
       
        print(f"  Saving to: {model_filename}")

    # Build model 
    model = create_hybrid_mlp_model(config).to(device)

    if verbose:
        comp = model.get_component_params()
        print(f"  Total parameters : {comp['total']:,}")
        print(f"    param_encoder  : {comp['param_encoder']:,}")
        print(f"    fusion         : {comp['fusion']:,}")
        print(f"    temporal_decoder: {comp['temporal_decoder']:,}")

    # Optimiser & scheduler ─
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    early_stopping = EarlyStopping(patience=config['patience'])

    # History buffers 
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae' : [], 'val_mae' : [],
        'train_mae_s': [], 'val_mae_s': [],
        'train_mae_i': [], 'val_mae_i': [],
        'train_mae_r': [], 'val_mae_r': [],
        'train_r2'  : [], 'val_r2'  : [],
    }

    best_val_r2  = -float('inf')
    best_val_mae = float('inf')
    best_epoch   = 0
    start_time   = time.time()

    # Training loop 
    for epoch in range(config['epochs']):
        train_loss, train_metrics = train_epoch_balanced(
            model, train_loader, optimizer, device, n_timesteps, weight_mode
        )
        val_loss, val_metrics = validate_balanced(
            model, val_loader, device, n_timesteps, weight_mode
        )
        scheduler.step()

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_mae_s'].append(train_metrics['MAE_S'])
        history['val_mae_s'].append(val_metrics['MAE_S'])
        history['train_mae_i'].append(train_metrics['MAE_I'])
        history['val_mae_i'].append(val_metrics['MAE_I'])
        history['train_mae_r'].append(train_metrics['MAE_R'])
        history['val_mae_r'].append(val_metrics['MAE_R'])
        history['train_r2'].append(train_metrics['R2'])
        history['val_r2'].append(val_metrics['R2'])

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == config['epochs'] - 1):
            print(f"  Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"Val R²={val_metrics['R2']:.4f}, "
                  f"MAE_I={val_metrics['MAE_I']:.2f}")

        # Save best checkpoint
        if val_metrics['R2'] > best_val_r2:
            best_val_r2  = val_metrics['R2']
            best_val_mae = val_metrics['MAE']
            best_epoch   = epoch + 1

            torch.save({
                'epoch'              : epoch + 1,
                'model_state_dict'   : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss'           : val_loss,
                'val_metrics'        : val_metrics,
                'config'             : config,
                'weight_mode'        : weight_mode,
                'seed'               : seed,
                'replicate_id'       : replicate_id,
                'training_time_minutes': (time.time() - start_time) / 60,
                'model_filename'     : model_filename,
                'param_names'        : ['tau', 'gamma', 'rho'],  # document params
            }, model_path)

        if early_stopping(val_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    if verbose:
        print(f"  Best epoch : {best_epoch}  |  Val R²: {best_val_r2:.4f}  |  "
              f"MAE: {best_val_mae:.2f}")
        print(f"  Time       : {training_time:.1f}s ({training_time/60:.2f} min)")
        print(f"  Saved      : {model_filename}")

    np.save(history_path, history)

    return {
        'replicate_id'          : replicate_id,
        'seed'                  : seed,
        'best_epoch'            : best_epoch,
        'best_val_r2'           : float(best_val_r2),
        'best_val_mae'          : float(best_val_mae),
        'best_val_mae_i'        : float(val_metrics['MAE_I']),
        'best_val_mae_s'        : float(val_metrics['MAE_S']),
        'best_val_mae_r'        : float(val_metrics['MAE_R']),
        'training_time_minutes' : training_time / 60,
        'model_filename'        : model_filename,
        'output_dir'            : str(output_dir),
    }, history


# MULTIPLE REPLICATES


def train_multiple_replicates(
    n_replicates, seeds, config, dataloaders, output_dir, weight_mode='modest'
):
    """
    Train n_replicates models, each with a different random seed.

    All checkpoints are saved in output_dir as:
        best_balanced_mlp_model_1.pt
        best_balanced_mlp_model_2.pt
        ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if seeds is None:
        seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021][:n_replicates]

    if len(seeds) < n_replicates:
        raise ValueError(f"Need {n_replicates} seeds, got {len(seeds)}")
    seeds = seeds[:n_replicates]

    
    print(f"TRAINING {n_replicates} REPLICATES  ·  3-Parameter SIR (τ, γ, ρ)")

    print(f"\n  All models → {output_dir}")
    print(f"  Seeds      : {seeds}")
    print(f"  Loss mode  : {weight_mode}")

    all_results, all_histories = [], []
    overall_start = time.time()

    for i, seed in enumerate(seeds, 1):
        results, history = train_single_replicate(
            replicate_id=i,
            seed=seed,
            config=config,
            dataloaders=dataloaders,
            output_dir=output_dir,
            weight_mode=weight_mode,
            verbose=True,
        )
        all_results.append(results)
        all_histories.append(history)

        print(f"\n  Progress : {i}/{n_replicates} replicates done")
        print(f"  Mean R²  : {np.mean([r['best_val_r2']  for r in all_results]):.4f}")
        print(f"  Mean MAE_I: {np.mean([r['best_val_mae_i'] for r in all_results]):.2f}")

    overall_time = time.time() - overall_start
  
    print("ALL REPLICATES COMPLETE")

    print(f"  Total time              : {overall_time/60:.2f} min")
    print(f"  Mean time per replicate : {overall_time/n_replicates/60:.2f} min")

    return all_results, all_histories



# STATISTICS & REPORTING


def compute_replicate_statistics(all_results):
    """Compute mean, std, CI across replicates for key metrics."""
    metrics = [
        'best_val_r2', 'best_val_mae', 'best_val_mae_i',
        'best_val_mae_s', 'best_val_mae_r', 'training_time_minutes',
    ]
    stats_dict = {}

    for metric in metrics:
        values = np.array([r[metric] for r in all_results])
        n      = len(values)
        mean   = np.mean(values)
        std    = np.std(values, ddof=1)  if n > 1 else 0.0
        sem    = stats.sem(values)       if n > 1 else 0.0
        ci     = stats.t.interval(0.95, n - 1, loc=mean, scale=sem) if n > 1 else (mean, mean)

        stats_dict[metric] = {
            'values'  : values.tolist(),
            'n'       : int(n),
            'mean'    : float(mean),
            'std'     : float(std),
            'sem'     : float(sem),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'min'     : float(values.min()),
            'max'     : float(values.max()),
            'median'  : float(np.median(values)),
            'cv'      : float(std / mean * 100) if mean != 0 else None,
        }

    return stats_dict


def create_summary_report(all_results, stats_dict, output_dir, weight_mode):
    """Write a plain-text summary report."""
    output_dir = Path(output_dir)

    lines = [
        "=" * 70,
        "REPLICATES TRAINING SUMMARY  ·  3-Parameter SIR (τ, γ, ρ)",
        "=" * 70,
        "",
        f"  Replicates   : {len(all_results)}",
        f"  Weight mode  : {weight_mode}",
        f"  Seeds        : {[r['seed'] for r in all_results]}",
        f"  Output dir   : {output_dir}",
        "",
        "  Model files:",
    ]
    for r in all_results:
        lines.append(f"    · {r['model_filename']}  (seed={r['seed']})")

    lines += [
        "",
        "=" * 70,
        "VALIDATION PERFORMANCE",
        "=" * 70,
        "",
        "MAE_I (Infected Compartment):",
        f"  Mean : {stats_dict['best_val_mae_i']['mean']:.2f}",
        f"  Std  : {stats_dict['best_val_mae_i']['std']:.2f}",
        f"  95% CI: [{stats_dict['best_val_mae_i']['ci_lower']:.2f}, "
        f"{stats_dict['best_val_mae_i']['ci_upper']:.2f}]",
        "",
        "R²:",
        f"  Mean : {stats_dict['best_val_r2']['mean']:.4f}",
        f"  Std  : {stats_dict['best_val_r2']['std']:.4f}",
        f"  95% CI: [{stats_dict['best_val_r2']['ci_lower']:.4f}, "
        f"{stats_dict['best_val_r2']['ci_upper']:.4f}]",
        "",
        "Per-Compartment MAE:",
    ]
    for comp in ['S', 'I', 'R']:
        m = stats_dict[f'best_val_mae_{comp.lower()}']
        lines.append(
            f"  MAE_{comp}: {m['mean']:7.2f} ± {m['std']:5.2f}  "
            f"(95% CI: [{m['ci_lower']:.2f}, {m['ci_upper']:.2f}])"
        )

    lines += [
        "",
        "=" * 70,
        "INDIVIDUAL REPLICATE RESULTS",
        "=" * 70,
        f"{'ID':>3} {'Seed':>6} {'Epoch':>6} {'R²':>8} {'MAE_I':>8} "
        f"{'MAE_S':>8} {'MAE_R':>8} {'Model File':<35}",
        "-" * 100,
    ]
    for r in all_results:
        lines.append(
            f"{r['replicate_id']:3d} {r['seed']:6d} {r['best_epoch']:6d} "
            f"{r['best_val_r2']:8.4f} {r['best_val_mae_i']:8.2f} "
            f"{r['best_val_mae_s']:8.2f} {r['best_val_mae_r']:8.2f} "
            f"{r['model_filename']:<35}"
        )

    lines += [
        "",
        "=" * 70,
        "NEXT STEPS",
        "=" * 70,
        "",
        "  VALIDATE:",
        f"    python step4_validate_SIR3param.py --models_dir {output_dir}",
        "",
        "  TEST (final dissertation results):",
        f"    python step5_test_SIR3param.py --models_dir {output_dir}",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    report_path = output_dir / 'REPLICATES_REPORT.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nSaved: {report_path}")
    print("\n" + report_text)


def plot_replicates_comparison(all_results, all_histories, output_dir):
    """Plot training-curve comparison across all replicates."""
    output_dir   = Path(output_dir)
    n_replicates = len(all_results)
    colors       = plt.cm.tab10(np.linspace(0, 1, n_replicates))

    fig = plt.figure(figsize=(20, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(
        'Replicate Comparison  ·  3-Parameter SIR (τ, γ, ρ)',
        fontsize=16, fontweight='bold'
    )

    # Val R²
    ax = fig.add_subplot(gs[0, 0])
    for i, (result, history) in enumerate(zip(all_results, all_histories)):
        ax.plot(range(1, len(history['val_r2']) + 1), history['val_r2'],
                color=colors[i], alpha=0.7, linewidth=2,
                label=f"M{result['replicate_id']} (s={result['seed']})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation R²')
    ax.set_title('R² Across Replicates')
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)

    # Val MAE_I
    ax = fig.add_subplot(gs[0, 1])
    for i, (result, history) in enumerate(zip(all_results, all_histories)):
        ax.plot(range(1, len(history['val_mae_i']) + 1), history['val_mae_i'],
                color=colors[i], alpha=0.7, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Validation MAE_I')
    ax.set_title('MAE_I Across Replicates'); ax.grid(True, alpha=0.3)

    # R² distribution
    ax = fig.add_subplot(gs[0, 2])
    final_r2 = [r['best_val_r2'] for r in all_results]
    ax.hist(final_r2, bins=min(10, n_replicates), color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(final_r2), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(final_r2):.4f}')
    ax.set_xlabel('Best Validation R²'); ax.set_ylabel('Frequency')
    ax.set_title('R² Distribution'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    out = output_dir / 'replicates_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")



# ENTRY POINT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train replicate SIR emulators — 3 parameters (tau, gamma, rho)"
    )
    parser.add_argument('--input',        type=str,   default='epidemic_data_age_adaptive_sobol_split_augmented.pkl')
    parser.add_argument('--output_dir',   type=str,   default='replicates_outputs')
    parser.add_argument('--n_replicates', type=int,   default=5)
    parser.add_argument('--seeds',        type=str,   default=None)
    parser.add_argument('--weight_mode',  type=str,   default='modest',
                         choices=['equal', 'modest', 'balanced'])
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=35) #30
    parser.add_argument('--lr',           type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3) #-3
    parser.add_argument('--patience',     type=int,   default=35) #35 early stopping parameter , help to know when to stop training, 
    args = parser.parse_args()

    seeds = (
        [int(s.strip()) for s in args.seeds.split(',')]
        if args.seeds is not None else None
    )

    # Model configurations
    config = {
        'n_params'        : 3,           # tau, gamma, rho
        'n_fourier'       : 64,
        'fourier_hidden'  : 32,
        'param_hidden'    : 16,
        'temporal_hidden' : 64,
        'dropout'         : 0.3,
        'n_knots'         : 12,
        'n_timepoints'    : n_timepoints,
        'total_population': 10000,
        # training
        'epochs'          : args.epochs,
        'batch_size'      : args.batch_size,
        'lr'              : args.lr,
        'weight_decay'    : args.weight_decay, # L2 regularization, applied through adam-optimizer, so it pernalize large weights (large weights more flexible model, risk of overfitting), This pushes weights toward zero during training, preventing a single parameter from dominating, 
        'patience'        : args.patience,
    }

    
    print(" SIR EMULATOR — REPLICATE TRAINING  (τ, γ, ρ)")
  
    print(f"\n  Output dir   : {args.output_dir}")
    print(f"  Replicates   : {args.n_replicates}")
    print(f"  Epochs/rep   : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Weight mode  : {args.weight_mode}")

    print(f"\nLoading data: {args.input}")
    dataloaders = create_dataloaders(args.input, batch_size=config['batch_size'])
    print(f"  Train : {len(dataloaders['train'].dataset)}")
    print(f"  Val   : {len(dataloaders['val'].dataset)}")

    all_results, all_histories = train_multiple_replicates(
        n_replicates=args.n_replicates,
        seeds=seeds,
        config=config,
        dataloaders=dataloaders,
        output_dir=args.output_dir,
        weight_mode=args.weight_mode,
    )

    # ── Summary statistics ────────────────────────────────────────────────────
  
    print("SUMMARY STATISTICS")


    stats_dict = compute_replicate_statistics(all_results)

    with open(Path(args.output_dir) / 'replicates_summary.json', 'w') as f:
        json.dump({'results': all_results, 'statistics': stats_dict,
                   'config': config, 'weight_mode': args.weight_mode}, f, indent=2)

    create_summary_report(all_results, stats_dict, args.output_dir, args.weight_mode)
    plot_replicates_comparison(all_results, all_histories, args.output_dir)

    df = pd.DataFrame(all_results)
    df.to_csv(Path(args.output_dir) / 'replicates_results.csv', index=False)
    print(f"Saved: {Path(args.output_dir) / 'replicates_results.csv'}")

    print(" TRAINING COMPLETE")
   
    print(f"\n  Mean R²    : {stats_dict['best_val_r2']['mean']:.4f}"
          f" ± {stats_dict['best_val_r2']['std']:.4f}")
    print(f"  Mean MAE_I : {stats_dict['best_val_mae_i']['mean']:.2f}"
          f" ± {stats_dict['best_val_mae_i']['std']:.2f}")
    print(f"  CV         : {stats_dict['best_val_mae_i']['cv']:.2f}%")
    print(f"\n  Validate   : python step4_validate_SIR3param.py --models_dir {args.output_dir}")
    print(f"  Test       : python step5_test_SIR3param.py --models_dir {args.output_dir}")
