"""
Compute and Visualize Metrics (TRULY Memory-Optimized GPU Version)

Key Fix: Accumulate on CPU, only move to GPU for computation
"""
import os
import argparse
import logging
import torch
import hashlib
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import sys
sys.path.insert(0, '/mnt/project')

from config import REPRESENTATION_DIR, METRIC_DIR
from metric_utils import (
    compute_prompt_entropy, compute_dataset_entropy,
    compute_curvature, compute_effective_rank, compute_l2_norm,
    compute_spectral_metrics, compute_sparsity,
    compute_cka, compute_cosine_similarity, compute_mean_shift
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")

def setup_logging():
    os.makedirs(METRIC_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Compute and visualize metrics (FIXED)')

    # Selection
    parser.add_argument('--models', type=str, nargs='+', default=['llama32/1b'],
                        choices=['llama32/1b', 'llama32/3b', 'gemma3/1b', 'gemma3/27b', 'mistral/7b',
                                 'olmo2/13b', 'olmo2/32b', 'qwen25/7b', 'qwen25/14b', 'qwen25/32b', 'qwen25/72b'],
                        help='Model configs in format "family/scale"')

    parser.add_argument('--dataset', type=str, default='mmlu',
                        help='Dataset name')

    # Metrics Control
    ALL_METRICS = ['prompt_entropy', 'dataset_entropy', 'curvature',
                   'effective_rank', 'l2_norm', 'spectral_metrics',
                   'sparsity', 'cka', 'cosine_similarity', 'mean_shift']

    parser.add_argument('--metrics', type=str, nargs='+', default=['all'],
                        choices=['all'] + ALL_METRICS,
                        help='List of metrics to compute')
    parser.add_argument('--normalization', type=str, nargs='+',
                        default=['maxEntropy'],
                        choices=['maxEntropy', 'logN', 'logD', 'logNlogD', 'raw'],
                        help='Normalization schemes for entropy metrics')

    # Memory Control
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of files to load per batch')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for file loading')
    parser.add_argument('--compute_on_cpu', action='store_true',
                        help='Force computation on CPU to save GPU memory')

    # Cache Control
    parser.add_argument('--force_recompute', action='store_true',
                        help='Ignore existing cache')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Expand "all"
    if 'all' in args.metrics:
        args.metrics = sorted(ALL_METRICS)
    else:
        args.metrics = sorted(args.metrics)

    args.models = sorted(args.models)
    args.normalization = sorted(args.normalization)

    return args

def get_cache_path(args):
    """Generate deterministic cache filename"""
    config_dict = {
        'dataset': args.dataset,
        'models': args.models,
        'metrics': args.metrics,
        'normalization': args.normalization
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    filename = f"cache_{args.dataset}_{run_hash}.pt"
    return os.path.join(METRIC_DIR, filename)

def load_single_file(path):
    """Load file to CPU only"""
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

def load_dataset_representations_v2(
    model_family, scale, variant, dataset,
    base_dir=REPRESENTATION_DIR,
    num_workers=4,
    device='cuda',
    batch_size=50
):
    """
    FIXED: Accumulate on CPU, only move to GPU for final computation

    Strategy:
    1. Load ALL files to CPU (cheap)
    2. Stack on CPU (cheap)
    3. Move to GPU only when needed for computation (controlled)
    """
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not files:
        logging.warning(f"No files found in {data_dir}")
        return None, None, None

    logging.info(f"Loading {len(files)} files to CPU in batches of {batch_size}...")

    # ===== KEY FIX: Accumulate on CPU =====
    pooled_agg = defaultdict(list)
    hidden_agg = defaultdict(list)
    has_hidden = False

    # Process in batches (to avoid overwhelming file system)
    num_batches = (len(files) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Loading {model_family}-{scale}-{variant}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        # Load batch to CPU
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            samples = list(executor.map(load_single_file, batch_files))
            samples = [s for s in samples if s is not None]

        if not samples:
            continue

        # Accumulate on CPU
        for s in samples:
            for layer, tensor in s.get('pooled_states', {}).items():
                pooled_agg[layer].append(tensor.cpu())  # Ensure CPU

            if s.get('hidden_states'):
                has_hidden = True
                for layer, tensor in s['hidden_states'].items():
                    hidden_agg[layer].append(tensor.cpu())  # Ensure CPU

        # Free batch memory
        del samples
        gc.collect()

    # Stack on CPU (much more memory available)
    logging.info("  Stacking tensors on CPU...")

    final_pooled = {}
    for k, v in pooled_agg.items():
        final_pooled[k] = torch.stack(v)  # Stack on CPU

    final_hidden = None
    if has_hidden:
        final_hidden = {}
        for k, v in hidden_agg.items():
            final_hidden[k] = torch.stack(v)  # Stack on CPU

    # Cleanup intermediate lists
    del pooled_agg, hidden_agg
    gc.collect()

    logging.info(f"  ✓ Data on CPU: pooled layers={len(final_pooled)}, " +
                 f"hidden layers={len(final_hidden) if final_hidden else 0}")

    # Data remains on CPU until needed
    return final_pooled, final_hidden, None

def compute_metrics_for_config(
    pooled, hidden,
    metrics_list, normalizations, device
):
    """
    Compute metrics with controlled GPU memory usage
    Move layers to GPU one at a time if needed
    """
    results = {}

    # Determine if we should use GPU
    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    # Sequence Metrics (from hidden states)
    if hidden is not None:
        if 'prompt_entropy' in metrics_list:
            logging.info("    Computing prompt_entropy...")
            if use_gpu:
                # Move layer by layer to GPU
                hidden_gpu = {}
                for k, v in hidden.items():
                    hidden_gpu[k] = v.to(device)
                results['prompt_entropy'] = compute_prompt_entropy(
                    hidden_gpu, normalizations=normalizations
                )
                del hidden_gpu
                torch.cuda.empty_cache()
            else:
                results['prompt_entropy'] = compute_prompt_entropy(
                    hidden, normalizations=normalizations
                )

        if 'curvature' in metrics_list:
            logging.info("    Computing curvature...")
            if use_gpu:
                hidden_gpu = {k: v.to(device) for k, v in hidden.items()}
                results['curvature'] = compute_curvature(hidden_gpu)
                del hidden_gpu
                torch.cuda.empty_cache()
            else:
                results['curvature'] = compute_curvature(hidden)

        if 'sparsity' in metrics_list:
            logging.info("    Computing sparsity...")
            if use_gpu:
                hidden_gpu = {k: v.to(device) for k, v in hidden.items()}
                results['sparsity'] = compute_sparsity(hidden_gpu)
                del hidden_gpu
                torch.cuda.empty_cache()
            else:
                results['sparsity'] = compute_sparsity(hidden)

    # Pooled Metrics
    if pooled is not None:
        if 'dataset_entropy' in metrics_list:
            logging.info("    Computing dataset_entropy...")
            if use_gpu:
                pooled_gpu = {k: v.to(device) for k, v in pooled.items()}
                results['dataset_entropy'] = compute_dataset_entropy(
                    pooled_gpu, normalizations=normalizations
                )
                del pooled_gpu
                torch.cuda.empty_cache()
            else:
                results['dataset_entropy'] = compute_dataset_entropy(
                    pooled, normalizations=normalizations
                )

        if 'effective_rank' in metrics_list:
            logging.info("    Computing effective_rank...")
            if use_gpu:
                pooled_gpu = {k: v.to(device) for k, v in pooled.items()}
                results['effective_rank'] = compute_effective_rank(pooled_gpu)
                del pooled_gpu
                torch.cuda.empty_cache()
            else:
                results['effective_rank'] = compute_effective_rank(pooled)

        if 'l2_norm' in metrics_list:
            logging.info("    Computing l2_norm...")
            if use_gpu:
                pooled_gpu = {k: v.to(device) for k, v in pooled.items()}
                results['l2_norm'] = compute_l2_norm(pooled_gpu)
                del pooled_gpu
                torch.cuda.empty_cache()
            else:
                results['l2_norm'] = compute_l2_norm(pooled)

        if 'spectral_metrics' in metrics_list:
            logging.info("    Computing spectral_metrics...")
            if use_gpu:
                pooled_gpu = {k: v.to(device) for k, v in pooled.items()}
                results['spectral_metrics'] = compute_spectral_metrics(pooled_gpu)
                del pooled_gpu
                torch.cuda.empty_cache()
            else:
                results['spectral_metrics'] = compute_spectral_metrics(pooled)

    return results

def compute_alignment_metrics(
    base_pooled, sft_pooled,
    metrics_list, device
):
    """Compute comparison metrics with controlled GPU usage"""
    results = {}
    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if base_pooled and sft_pooled:
        if 'mean_shift' in metrics_list:
            logging.info("    Computing mean_shift...")
            if use_gpu:
                base_gpu = {k: v.to(device) for k, v in base_pooled.items()}
                sft_gpu = {k: v.to(device) for k, v in sft_pooled.items()}
                results['mean_shift'] = compute_mean_shift(base_gpu, sft_gpu)
                del base_gpu, sft_gpu
                torch.cuda.empty_cache()
            else:
                results['mean_shift'] = compute_mean_shift(base_pooled, sft_pooled)

        if 'cka' in metrics_list:
            logging.info("    Computing cka...")
            if use_gpu:
                base_gpu = {k: v.to(device) for k, v in base_pooled.items()}
                sft_gpu = {k: v.to(device) for k, v in sft_pooled.items()}
                results['cka'] = compute_cka(base_gpu, sft_gpu)
                del base_gpu, sft_gpu
                torch.cuda.empty_cache()
            else:
                results['cka'] = compute_cka(base_pooled, sft_pooled)

        if 'cosine_similarity' in metrics_list:
            logging.info("    Computing cosine_similarity...")
            if use_gpu:
                base_gpu = {k: v.to(device) for k, v in base_pooled.items()}
                sft_gpu = {k: v.to(device) for k, v in sft_pooled.items()}
                results['cosine_similarity'] = compute_cosine_similarity(base_gpu, sft_gpu)
                del base_gpu, sft_gpu
                torch.cuda.empty_cache()
            else:
                results['cosine_similarity'] = compute_cosine_similarity(base_pooled, sft_pooled)

    return results

def visualize_results(results_store, output_dir):
    """Generate publication-ready plots"""
    os.makedirs(output_dir, exist_ok=True)

    TITLE_SIZE = 14
    LABEL_SIZE = 14
    TICK_SIZE = 14
    LEGEND_SIZE = 12
    LINE_WIDTH = 3
    MARKER_SIZE = 4

    for metric_name, model_data in results_store.items():
        if not model_data:
            continue

        first_val = next(iter(model_data.values()))

        # Determine sub-keys
        sub_keys = set()
        if isinstance(first_val, dict):
            for m_data in model_data.values():
                if isinstance(m_data, dict):
                    sub_keys.update(m_data.keys())
            sub_keys = sorted(list(sub_keys))
        else:
            sub_keys = [None]

        for sub_key in sub_keys:
            plt.figure(figsize=(6, 4))
            has_data = False

            for model_label, values in model_data.items():
                plot_vals = []
                if sub_key is None:
                    plot_vals = values
                elif isinstance(values, dict) and sub_key in values:
                    plot_vals = values[sub_key]

                if plot_vals:
                    plt.plot(range(len(plot_vals)), plot_vals,
                             marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                             label=model_label, alpha=0.85)
                    has_data = True

            if has_data:
                full_title = f"{metric_name} ({sub_key})" if sub_key else metric_name
                plt.title(full_title, fontsize=TITLE_SIZE, pad=15)
                plt.xlabel("Layer Depth", fontsize=LABEL_SIZE)
                ylabel = sub_key if sub_key else "Value"
                plt.ylabel(ylabel, fontsize=LABEL_SIZE)
                plt.xticks(fontsize=TICK_SIZE)
                plt.yticks(fontsize=TICK_SIZE)
                plt.legend(fontsize=LEGEND_SIZE, frameon=True)
                plt.grid(True, linestyle='--', alpha=0.4)
                plt.tight_layout()

                fname = f"{metric_name}_{sub_key}" if sub_key else metric_name
                fname = fname.replace("/", "_").replace(" ", "_")
                save_path = os.path.join(output_dir, f"{fname}.png")

                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"  Saved: {fname}.png")
            plt.show()
            plt.close()

def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("="*80)
    logger.info("FIXED Memory-Optimized Metric Computation")
    logger.info("="*80)
    logger.info(f"Models: {args.models}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Metrics: {args.metrics}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Strategy: Accumulate on CPU, compute layer-by-layer on GPU")
    logger.info("="*80)

    # Check cache
    cache_path = get_cache_path(args)
    logger.info(f"Cache: {cache_path}")

    visualization_store = None

    if os.path.exists(cache_path) and not args.force_recompute:
        logger.info("✓ Loading from cache...")
        try:
            visualization_store = torch.load(cache_path, map_location='cpu')
            logger.info("✓ Cache loaded successfully")
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            visualization_store = None

    # Compute if needed
    if visualization_store is None:
        logger.info("Computing metrics...")
        visualization_store = defaultdict(dict)

        for model_str in args.models:
            family, scale = model_str.split('/')
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {family}/{scale}")
            logger.info(f"{'='*60}")

            # Load Base (to CPU)
            logger.info("  Loading BASE variant to CPU...")
            base_pooled, base_hidden, _ = load_dataset_representations_v2(
                family, scale, 'base', args.dataset,
                num_workers=args.num_workers,
                device=args.device,
                batch_size=args.batch_size
            )

            if base_pooled is None:
                logger.warning(f"  Skipping {family}/{scale} - no data")
                continue

            # Compute Base Metrics
            logger.info("  Computing BASE metrics...")
            base_metrics = compute_metrics_for_config(
                base_pooled, base_hidden,
                args.metrics, args.normalization, args.device
            )
            for m, vals in base_metrics.items():
                # Move to CPU for storage
                if isinstance(vals, dict):
                    vals = {k: [v.cpu().item() if torch.is_tensor(v) else v for v in vs]
                           for k, vs in vals.items()}
                else:
                    vals = [v.cpu().item() if torch.is_tensor(v) else v for v in vals]
                visualization_store[m][f"{family}-{scale}-base"] = vals

            # Load Instruct (to CPU)
            logger.info("  Loading INSTRUCT variant to CPU...")
            sft_pooled, sft_hidden, _ = load_dataset_representations_v2(
                family, scale, 'instruct', args.dataset,
                num_workers=args.num_workers,
                device=args.device,
                batch_size=args.batch_size
            )

            if sft_pooled is not None:
                # Compute Instruct Metrics
                logger.info("  Computing INSTRUCT metrics...")
                sft_metrics = compute_metrics_for_config(
                    sft_pooled, sft_hidden,
                    args.metrics, args.normalization, args.device
                )
                for m, vals in sft_metrics.items():
                    if isinstance(vals, dict):
                        vals = {k: [v.cpu().item() if torch.is_tensor(v) else v for v in vs]
                               for k, vs in vals.items()}
                    else:
                        vals = [v.cpu().item() if torch.is_tensor(v) else v for v in vals]
                    visualization_store[m][f"{family}-{scale}-instruct"] = vals

                # Compute Alignment
                logger.info("  Computing ALIGNMENT metrics...")
                align_metrics = compute_alignment_metrics(
                    base_pooled, sft_pooled,
                    args.metrics, args.device
                )
                for m, vals in align_metrics.items():
                    vals = [v.cpu().item() if torch.is_tensor(v) else v for v in vals]
                    visualization_store[m][f"{family}-{scale}-alignment"] = vals

                del sft_pooled, sft_hidden

            # Cleanup
            del base_pooled, base_hidden
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()

        # Save cache
        logger.info(f"\nSaving cache to {cache_path}...")
        torch.save(dict(visualization_store), cache_path)
        logger.info("✓ Cache saved")

    # Visualize
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = os.path.join(METRIC_DIR, 'plots', args.dataset, timestamp)
    logger.info(f"\nGenerating plots to {viz_dir}...")
    visualize_results(visualization_store, viz_dir)
    logger.info("✓ Done!")

if __name__ == "__main__":
    main()