"""
Compute and Visualize Metrics (Streaming & Memory Safe Version)

Optimized for 32B+ Models:
1. Sequence metrics (Entropy/Curvature) are computed strictly batch-wise (Streaming).
2. Global metrics (Dataset Entropy/CKA) utilize a capped subset of pooled states to fit in RAM.
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
import numpy as np

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
    parser = argparse.ArgumentParser(description='Compute and visualize metrics (Memory Safe)')

    # Selection
    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/32b'],
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
                        help='Normalization schemes for entropy metrics')

    # Memory Control
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of files to load per batch')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max pooled samples to load for global metrics (Dataset Entropy/CKA)')
    parser.add_argument('--compute_on_cpu', action='store_true',
                        help='Force computation on CPU')

    # Cache Control
    parser.add_argument('--force_recompute', action='store_true',
                        help='Ignore existing cache')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    if 'all' in args.metrics:
        args.metrics = sorted(ALL_METRICS)
    else:
        args.metrics = sorted(args.metrics)

    args.models = sorted(args.models)
    return args


def get_cache_path(args):
    """Generate deterministic cache filename"""
    config_dict = {
        'dataset': args.dataset,
        'models': args.models,
        'metrics': args.metrics,
        'normalization': args.normalization,
        'max_samples': args.max_samples  # Cache depends on sample size now
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    filename = f"cache_{args.dataset}_{run_hash}.pt"
    return os.path.join(METRIC_DIR, filename)


def load_single_file(path):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None


# ============================================================================
# Core Logic 1: Streaming Computation for Sequence Metrics (Hidden States)
# ============================================================================
def compute_sequence_metrics_streaming(
        model_family, scale, variant, dataset,
        metrics_list, normalizations, device,
        base_dir=REPRESENTATION_DIR,
        batch_size=10
):
    """
    Computes metrics that depend on (N, T, D) hidden states (e.g. Prompt Entropy).
    Loads batch -> Computes -> Aggregates Results -> Frees Memory.
    Never holds all hidden states in RAM.
    """
    needed_metrics = set(['prompt_entropy', 'curvature', 'sparsity']) & set(metrics_list)
    if not needed_metrics:
        return {}

    logging.info(f" ✅   [Streaming] Computing sequence metrics: {needed_metrics}...")

    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not files:
        logging.warning(f"No files found in {data_dir}")
        return {}

    # ================= KEY FIX HERE =================
    # 使用 3 层 defaultdict，确保 [metric][subkey][layer] 能够自动创建列表
    # Structure: Metric -> SubKey -> LayerIdx -> List of values
    results_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # ================================================

    # Process in batches
    num_batches = (len(files) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    Stream Processing {variant}"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        # 1. Load Batch
        batch_hidden = defaultdict(list)
        has_hidden = False

        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded_samples = list(executor.map(load_single_file, batch_files))

        for s in loaded_samples:
            if s and s.get('hidden_states'):
                has_hidden = True
                for layer, tensor in s['hidden_states'].items():
                    batch_hidden[layer].append(tensor)

        if not has_hidden:
            continue

        # 2. Stack Batch
        hidden_batch_stacked = {}
        for k, v in batch_hidden.items():
            try:
                tensor = torch.stack(v).float()
                if device == 'cuda':
                    tensor = tensor.to(device)
                hidden_batch_stacked[k] = tensor
            except Exception as e:
                logging.error(f"Stacking error layer {k}: {e}")

        # 3. Compute Metrics for this Batch
        batch_results = {}

        if 'prompt_entropy' in needed_metrics:
            batch_results['prompt_entropy'] = compute_prompt_entropy(
                hidden_batch_stacked, normalizations=normalizations
            )

        if 'curvature' in needed_metrics:
            batch_results['curvature'] = compute_curvature(hidden_batch_stacked)

        if 'sparsity' in needed_metrics:
            batch_results['sparsity'] = compute_sparsity(hidden_batch_stacked)

        # 4. Aggregate Results (Move to CPU list)
        for metric, layers_data in batch_results.items():

            # Case A: Dictionary (e.g., prompt_entropy returns {norm: [val_layer0, val_layer1...]})
            if isinstance(layers_data, dict):
                for sub_key, vals in layers_data.items():
                    for layer_i, val in enumerate(vals):
                        # 现在这里不会报错了，因为 [layer_i] 会自动初始化为一个 list
                        results_agg[metric][sub_key][layer_i].append(val)

            # Case B: List (e.g., sparsity returns [val_layer0, val_layer1...])
            elif isinstance(layers_data, list):
                for layer_i, val in enumerate(layers_data):
                    # 同样，这里会自动创建 'raw' 键和 layer_i 键
                    results_agg[metric]['raw'][layer_i].append(val)

        # 5. Cleanup
        del hidden_batch_stacked, batch_hidden, loaded_samples
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 6. Final Reduction (Average across batches)
    final_results = {}
    for metric, sub_dict in results_agg.items():
        final_results[metric] = {}
        for sub_key, layers_dict in sub_dict.items():
            # layers_dict is {layer_idx: [batch1_val, batch2_val...]}
            sorted_layers = sorted(layers_dict.keys())
            if not sorted_layers: continue

            averaged_vals = []
            max_layer = max(sorted_layers)
            # 重建列表：按层顺序排列
            for i in range(max_layer + 1):
                if i in layers_dict:
                    vals = layers_dict[i]
                    averaged_vals.append(sum(vals) / len(vals))
                else:
                    averaged_vals.append(0.0)

            final_results[metric][sub_key] = averaged_vals

    return final_results

# ============================================================================
# Core Logic 2: Subset Loading for Pooled Metrics (Global Matrix)
# ============================================================================
def load_pooled_subset(
        model_family, scale, variant, dataset,
        max_samples=5000,
        base_dir=REPRESENTATION_DIR,
        batch_size=50
):
    """
    Loads a random subset of pooled states to fit in memory.
    Returns Dict[layer_idx, Tensor(N, D)]
    """
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not files:
        return None

    # Shuffle files to get random sample
    # (Assuming 1 file = 1 sample usually, or small batches)
    import random
    random.seed(42)
    random.shuffle(files)

    pooled_agg = defaultdict(list)
    current_count = 0

    # Only load enough files to hit max_samples
    files_to_load = files  # We might stop early

    pbar = tqdm(total=min(len(files), max_samples), desc=f"    Loading Pooled ({variant})")

    for i in range(0, len(files), batch_size):
        if current_count >= max_samples:
            break

        batch_files = files[i: i + batch_size]
        with ThreadPoolExecutor(max_workers=4) as executor:
            samples = list(executor.map(load_single_file, batch_files))

        for s in samples:
            if not s: continue
            if current_count >= max_samples: break

            for layer, tensor in s.get('pooled_states', {}).items():
                pooled_agg[layer].append(tensor.float().cpu())  # Keep on CPU

            current_count += 1
            pbar.update(1)

    pbar.close()

    # Stack
    final_pooled = {}
    for k, v in pooled_agg.items():
        final_pooled[k] = torch.stack(v)

    logging.info(f"    ✓ Loaded {current_count} pooled samples for global metrics.")
    return final_pooled


def compute_pooled_metrics_in_memory(
        pooled_states,
        metrics_list, normalizations, device
):
    """Compute metrics that need the whole dataset matrix (e.g. Dataset Entropy, CKA)"""
    results = {}
    needed = set(['dataset_entropy', 'effective_rank', 'l2_norm', 'spectral_metrics']) & set(metrics_list)

    if not pooled_states or not needed:
        return results

    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    # Move entire pooled matrix to GPU layer by layer if possible
    # (Since we capped max_samples, this should fit in 24GB+ VRAM usually,
    # e.g. 5000 * 4096 * 4 bytes = 80MB per layer. Totally fine.)

    if 'dataset_entropy' in metrics_list:
        logging.info(" ⭕️    Computing dataset_entropy...")
        # Process layer by layer to be safe
        res = {norm: [] for norm in normalizations}
        for layer in sorted(pooled_states.keys()):
            tensor = pooled_states[layer]
            if use_gpu: tensor = tensor.to(device)

            # Create temp dict for utils compatibility
            tmp_dict = {layer: tensor}
            layer_res = compute_dataset_entropy(tmp_dict, normalizations=normalizations)

            for norm in normalizations:
                if layer_res[norm]:
                    res[norm].append(layer_res[norm][0])

            if use_gpu: del tensor; torch.cuda.empty_cache()
        results['dataset_entropy'] = res

    # Helper for other metrics that follow same pattern
    def run_metric(name, func):
        if name in metrics_list:
            logging.info(f" ⭕️  Computing {name}...")
            # Some utils expect Dict[layer, tensor]
            # We can pass the whole dict if it fits on GPU, or loop.
            # Looping is safer.
            metrics_out = defaultdict(list)

            for layer in sorted(pooled_states.keys()):
                tensor = pooled_states[layer]
                if use_gpu: tensor = tensor.to(device)

                tmp_dict = {layer: tensor}
                val = func(tmp_dict)  # func usually returns list or dict

                # Unpack result
                if isinstance(val, dict):  # e.g. spectral_metrics
                    for sub_k, sub_v in val.items():
                        metrics_out[sub_k].append(sub_v[0])
                elif isinstance(val, list):
                    metrics_out['raw'].append(val[0])

                if use_gpu: del tensor; torch.cuda.empty_cache()

            if isinstance(val, dict):
                results[name] = dict(metrics_out)
            else:
                results[name] = metrics_out['raw'] if 'raw' in metrics_out else []

    run_metric('effective_rank', compute_effective_rank)
    run_metric('l2_norm', compute_l2_norm)
    run_metric('spectral_metrics', compute_spectral_metrics)

    return results


def compute_alignment_metrics_safe(
        base_pooled, sft_pooled,
        metrics_list, device
):
    """Compute comparison metrics (CKA)"""
    results = {}
    needed = set(['cka', 'cosine_similarity', 'mean_shift']) & set(metrics_list)
    if not base_pooled or not sft_pooled or not needed:
        return results

    logging.info("    Computing alignment metrics (Base vs SFT)...")

    # Ensure layers match
    common_layers = sorted(list(set(base_pooled.keys()) & set(sft_pooled.keys())))

    # Truncate to same length (since we did random subsampling)
    min_len = min(base_pooled[common_layers[0]].shape[0], sft_pooled[common_layers[0]].shape[0])

    # Prepare aligned dicts
    base_aligned = {}
    sft_aligned = {}

    for l in common_layers:
        base_aligned[l] = base_pooled[l][:min_len]
        sft_aligned[l] = sft_pooled[l][:min_len]

    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if 'cka' in metrics_list:
        if use_gpu:
            # CKA is heavy, move to GPU layer by layer
            res = []
            for l in common_layers:
                b = base_aligned[l].to(device)
                s = sft_aligned[l].to(device)
                res.append(compute_cka({l: b}, {l: s})[0])
                del b, s
                torch.cuda.empty_cache()
            results['cka'] = res
        else:
            results['cka'] = compute_cka(base_aligned, sft_aligned)

    if 'cosine_similarity' in metrics_list:
        # Lighter, can run on CPU if needed, but GPU faster
        if use_gpu:
            base_gpu = {k: v.to(device) for k, v in base_aligned.items()}
            sft_gpu = {k: v.to(device) for k, v in sft_aligned.items()}
            results['cosine_similarity'] = compute_cosine_similarity(base_gpu, sft_gpu)
            del base_gpu, sft_gpu
        else:
            results['cosine_similarity'] = compute_cosine_similarity(base_aligned, sft_aligned)

    if 'mean_shift' in metrics_list:
        # Very light
        results['mean_shift'] = compute_mean_shift(base_aligned, sft_aligned)

    return results


# ============================================================================
# Main Execution
# ============================================================================

def visualize_results(results_store, output_dir):
    """Simplified Plotting"""
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, model_data in results_store.items():
        if not model_data: continue

        # Determine structure depth
        first_val = next(iter(model_data.values()))
        if isinstance(first_val, dict):
            # Complex metric (e.g. entropy with normalizations)
            sub_keys = sorted(list(first_val.keys()))
        else:
            # Simple list
            sub_keys = [None]

        for sub_key in sub_keys:
            plt.figure(figsize=(6, 4), dpi=300)
            has_data = False

            for model_label, values in model_data.items():
                y_vals = values[sub_key] if sub_key else values
                if not y_vals: continue

                x_vals = range(len(y_vals))
                plt.plot(x_vals, y_vals, marker='o', markersize=8, linewidth=2, label=model_label, alpha=0.8)
                has_data = True

            if has_data:
                title = f"{metric_name} - {sub_key}" if sub_key else metric_name
                plt.title(title)
                plt.xlabel("Layer Depth")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)

                clean_name = title.replace("/", "_").replace(" ", "_")
                plt.savefig(os.path.join(output_dir, f"{clean_name}.png"), dpi=300)
                plt.show()
                plt.close()


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info(f"✅Target Models: {args.models}")
    logger.info(f"Memory Limit (Max Samples): {args.max_samples}")

    cache_path = get_cache_path(args)
    visualization_store = defaultdict(dict)

    if os.path.exists(cache_path) and not args.force_recompute:
        logger.info(f"✅Loading from cache: {cache_path}")
        visualization_store = torch.load(cache_path)
    else:
        for model_str in args.models:
            family, scale = model_str.split('/')
            logger.info(f"✅Processing {family}/{scale}...")

            # ------------------------------------------------------------
            # 1. BASE Model
            # ------------------------------------------------------------
            # A. Sequence Metrics (Streaming - Safe)
            base_seq_metrics = compute_sequence_metrics_streaming(
                family, scale, 'base', args.dataset,
                args.metrics, args.normalization, args.device,
                batch_size=args.batch_size
            )
            for m, vals in base_seq_metrics.items():
                visualization_store[m][f"{family}-{scale}-base"] = vals

            # B. Global Metrics (Subsampling - Safe)
            base_pooled = load_pooled_subset(
                family, scale, 'base', args.dataset,
                max_samples=args.max_samples,
                batch_size=args.batch_size
            )

            if base_pooled:
                base_pooled_metrics = compute_pooled_metrics_in_memory(
                    base_pooled, args.metrics, args.normalization, args.device
                )
                for m, vals in base_pooled_metrics.items():
                    visualization_store[m][f"{family}-{scale}-base"] = vals

            # ------------------------------------------------------------
            # 2. INSTRUCT Model
            # ------------------------------------------------------------
            sft_seq_metrics = compute_sequence_metrics_streaming(
                family, scale, 'instruct', args.dataset,
                args.metrics, args.normalization, args.device,
                batch_size=args.batch_size
            )
            for m, vals in sft_seq_metrics.items():
                visualization_store[m][f"{family}-{scale}-instruct"] = vals

            sft_pooled = load_pooled_subset(
                family, scale, 'instruct', args.dataset,
                max_samples=args.max_samples,
                batch_size=args.batch_size
            )

            if sft_pooled:
                sft_pooled_metrics = compute_pooled_metrics_in_memory(
                    sft_pooled, args.metrics, args.normalization, args.device
                )
                for m, vals in sft_pooled_metrics.items():
                    visualization_store[m][f"{family}-{scale}-instruct"] = vals

            # ------------------------------------------------------------
            # 3. Alignment Metrics (Base vs Instruct)
            # ------------------------------------------------------------
            if base_pooled and sft_pooled:
                align_metrics = compute_alignment_metrics_safe(
                    base_pooled, sft_pooled, args.metrics, args.device
                )
                for m, vals in align_metrics.items():
                    visualization_store[m][f"{family}-{scale}-alignment"] = vals

            # Cleanup GPU for next model
            del base_pooled, sft_pooled
            gc.collect()
            torch.cuda.empty_cache()

        # Save Cache
        torch.save(dict(visualization_store), cache_path)

    # Visualize
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = os.path.join(METRIC_DIR, 'plots', args.dataset, timestamp)
    visualize_results(visualization_store, viz_dir)
    logger.info("✅Done.")


if __name__ == "__main__":
    main()