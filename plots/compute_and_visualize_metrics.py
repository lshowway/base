"""
Compute and Visualize Metrics (Multi-Variant & Memory Safe & Colorful)

Supports: Base, SFT, DPO, RLVR, Instruct
Logic:
1. Load Base (Keep in Memory)
2. Loop over [SFT, DPO, RLVR, Instruct]:
   - Load Variant
   - Compute Variant Metrics
   - Compute Alignment (Base vs Variant)
   - Unload Variant (GC)
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
sys.path.insert(0, os.getcwd())

from config import REPRESENTATION_DIR, METRIC_DIR
from metric_utils import (
    compute_prompt_entropy, compute_dataset_entropy,
    compute_curvature, compute_effective_rank, compute_l2_norm,
    compute_spectral_metrics, compute_sparsity,
    compute_cka, compute_cosine_similarity, compute_mean_shift
)

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")

def setup_logging():
    os.makedirs(METRIC_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s' # Simplified format
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Compute metrics for Base/SFT/DPO/RLVR/Instruct')

    parser.add_argument('--models', type=str, nargs='+', default=['olmo2/7b'],
                        help='Model configs in format "family/scale"')

    parser.add_argument('--dataset', type=str, default='mmlu',
                        help='Dataset name')

    # Variants to process
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['sft', 'dpo', 'rlvr', 'instruct'],
                        choices=['sft', 'dpo', 'rlvr', 'instruct'],
                        help='Variants to compare against Base')

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
                        help='Number of files to load per batch for streaming')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Max pooled samples to load for global metrics')
    parser.add_argument('--force_recompute', action='store_true')
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
        'variants': args.variants,
        'metrics': args.metrics,
        'normalization': args.normalization,
        'max_samples': args.max_samples
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    run_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
    filename = f"cache_multivar_{args.dataset}_{run_hash}.pt"
    return os.path.join(METRIC_DIR, filename)

def load_single_file(path):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        logging.error(f"❌ Error loading {path}: {e}")
        return None

# ============================================================================
# Streaming & Safe Computation Functions
# ============================================================================

def compute_sequence_metrics_streaming(
    model_family, scale, variant, dataset,
    metrics_list, normalizations, device,
    base_dir=REPRESENTATION_DIR,
    batch_size=10
):
    """Streaming computation for Hidden States (Entropy, Curvature)"""
    needed_metrics = set(['prompt_entropy', 'curvature', 'sparsity']) & set(metrics_list)
    if not needed_metrics:
        return {}

    logging.info(f"    🌊 [Stream] Computing sequence metrics ({variant})...")
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not files:
        logging.warning(f"    ⚠️ No files found for {variant} in {data_dir}")
        return {}

    # Use 3-layer dict to prevent IndexError
    results_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    num_batches = (len(files) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"    🌊 Processing {variant}", leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(files))
        batch_files = files[start_idx:end_idx]

        batch_hidden = defaultdict(list)
        has_hidden = False

        with ThreadPoolExecutor(max_workers=4) as executor:
            loaded_samples = list(executor.map(load_single_file, batch_files))

        for s in loaded_samples:
            if s and s.get('hidden_states'):
                has_hidden = True
                for layer, tensor in s['hidden_states'].items():
                    batch_hidden[layer].append(tensor)

        if not has_hidden: continue

        hidden_batch_stacked = {}
        for k, v in batch_hidden.items():
            try:
                tensor = torch.stack(v).float()
                if device == 'cuda': tensor = tensor.to(device)
                hidden_batch_stacked[k] = tensor
            except Exception: pass

        batch_results = {}
        if 'prompt_entropy' in needed_metrics:
            batch_results['prompt_entropy'] = compute_prompt_entropy(hidden_batch_stacked, normalizations=normalizations)
        if 'curvature' in needed_metrics:
            batch_results['curvature'] = compute_curvature(hidden_batch_stacked)
        if 'sparsity' in needed_metrics:
            batch_results['sparsity'] = compute_sparsity(hidden_batch_stacked)

        # Aggregate
        for metric, layers_data in batch_results.items():
            if isinstance(layers_data, dict):
                for sub_key, vals in layers_data.items():
                    for layer_i, val in enumerate(vals):
                        results_agg[metric][sub_key][layer_i].append(val)
            elif isinstance(layers_data, list):
                for layer_i, val in enumerate(layers_data):
                    results_agg[metric]['raw'][layer_i].append(val)

        del hidden_batch_stacked, batch_hidden, loaded_samples
        if device == 'cuda': torch.cuda.empty_cache()

    # Final Reduction
    final_results = {}
    for metric, sub_dict in results_agg.items():
        final_results[metric] = {}
        for sub_key, layers_dict in sub_dict.items():
            sorted_layers = sorted(layers_dict.keys())
            if not sorted_layers: continue
            averaged_vals = []
            max_layer = max(sorted_layers)
            for i in range(max_layer + 1):
                if i in layers_dict:
                    vals = layers_dict[i]
                    averaged_vals.append(sum(vals) / len(vals))
                else:
                    averaged_vals.append(0.0)
            final_results[metric][sub_key] = averaged_vals

    return final_results

def load_pooled_subset(
    model_family, scale, variant, dataset,
    max_samples=5000,
    base_dir=REPRESENTATION_DIR,
    batch_size=50
):
    """Load subset of pooled states to CPU"""
    logging.info(f"    📥 [Load] Loading pooled subset ({variant}, max={max_samples})...")
    data_dir = os.path.join(base_dir, model_family, f"{scale}_{variant}", dataset)
    files = sorted(glob(os.path.join(data_dir, "*.pt")))

    if not files: return None

    # Deterministic shuffle
    import random
    random.seed(42)
    random.shuffle(files)

    pooled_agg = defaultdict(list)
    current_count = 0

    for i in range(0, len(files), batch_size):
        if current_count >= max_samples: break
        batch_files = files[i : i+batch_size]
        with ThreadPoolExecutor(max_workers=4) as executor:
            samples = list(executor.map(load_single_file, batch_files))

        for s in samples:
            if not s: continue
            if current_count >= max_samples: break
            for layer, tensor in s.get('pooled_states', {}).items():
                pooled_agg[layer].append(tensor.float().cpu())
            current_count += 1

    final_pooled = {}
    for k, v in pooled_agg.items():
        final_pooled[k] = torch.stack(v)

    logging.info(f"    ✅ [Load] Loaded {current_count} pooled samples for {variant}")
    return final_pooled

def compute_pooled_metrics_in_memory(pooled_states, metrics_list, normalizations, device):
    """Compute global metrics (Entropy, Rank, Norm)"""
    results = {}
    needed = set(['dataset_entropy', 'effective_rank', 'l2_norm', 'spectral_metrics']) & set(metrics_list)
    if not pooled_states or not needed: return results

    logging.info(f"    🧠 [Compute] Global metrics (Size: {len(pooled_states)} layers)...")
    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if 'dataset_entropy' in metrics_list:
        res = {norm: [] for norm in normalizations}
        for layer in sorted(pooled_states.keys()):
            tensor = pooled_states[layer]
            if use_gpu: tensor = tensor.to(device)
            layer_res = compute_dataset_entropy({layer: tensor}, normalizations=normalizations)
            for norm in normalizations:
                if layer_res[norm]: res[norm].append(layer_res[norm][0])
            if use_gpu: del tensor; torch.cuda.empty_cache()
        results['dataset_entropy'] = res

    # Generic loop for others
    for metric, func in [('effective_rank', compute_effective_rank),
                         ('l2_norm', compute_l2_norm),
                         ('spectral_metrics', compute_spectral_metrics)]:
        if metric in metrics_list:
            metrics_out = defaultdict(list)
            for layer in sorted(pooled_states.keys()):
                tensor = pooled_states[layer]
                if use_gpu: tensor = tensor.to(device)
                val = func({layer: tensor})
                if isinstance(val, dict):
                    for k, v in val.items(): metrics_out[k].append(v[0])
                elif isinstance(val, list):
                    metrics_out['raw'].append(val[0])
                if use_gpu: del tensor; torch.cuda.empty_cache()

            if isinstance(val, dict): results[metric] = dict(metrics_out)
            else: results[metric] = metrics_out['raw'] if 'raw' in metrics_out else []

    return results

def compute_alignment_metrics_safe(base_pooled, target_pooled, metrics_list, device):
    """Compute alignment between Base and Target"""
    results = {}
    needed = set(['cka', 'cosine_similarity', 'mean_shift']) & set(metrics_list)
    if not base_pooled or not target_pooled or not needed: return results

    logging.info(f"    📐 [Align] Computing Alignment Metrics...")
    common_layers = sorted(list(set(base_pooled.keys()) & set(target_pooled.keys())))
    min_len = min(base_pooled[common_layers[0]].shape[0], target_pooled[common_layers[0]].shape[0])

    use_gpu = (device == 'cuda' and torch.cuda.is_available())

    if 'cka' in metrics_list:
        res = []
        for l in common_layers:
            b = base_pooled[l][:min_len]
            t = target_pooled[l][:min_len]
            if use_gpu:
                b, t = b.to(device), t.to(device)
            res.append(compute_cka({l: b}, {l: t})[0])
            if use_gpu: del b, t; torch.cuda.empty_cache()
        results['cka'] = res

    if 'cosine_similarity' in metrics_list:
        base_aligned = {l: base_pooled[l][:min_len] for l in common_layers}
        target_aligned = {l: target_pooled[l][:min_len] for l in common_layers}
        if use_gpu:
            base_gpu = {k: v.to(device) for k, v in base_aligned.items()}
            target_gpu = {k: v.to(device) for k, v in target_aligned.items()}
            results['cosine_similarity'] = compute_cosine_similarity(base_gpu, target_gpu)
            del base_gpu, target_gpu; torch.cuda.empty_cache()
        else:
            results['cosine_similarity'] = compute_cosine_similarity(base_aligned, target_aligned)

    if 'mean_shift' in metrics_list:
        base_aligned = {l: base_pooled[l][:min_len] for l in common_layers}
        target_aligned = {l: target_pooled[l][:min_len] for l in common_layers}
        results['mean_shift'] = compute_mean_shift(base_aligned, target_aligned)

    return results

# ============================================================================
# Main Loop & Visualization
# ============================================================================

def visualize_results(results_store, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, model_data in results_store.items():
        if not model_data: continue

        # Check sub-keys (normalization etc.)
        first_val = next(iter(model_data.values()))
        if isinstance(first_val, dict):
            sub_keys = sorted(list(first_val.keys()))
        else:
            sub_keys = [None]

        for sub_key in sub_keys:
            plt.figure(figsize=(6, 4), dpi=300)
            has_data = False

            # Sort keys to make legend orderly (Base first, then variants)
            sorted_keys = sorted(model_data.keys())

            for model_label, values in model_data.items():
                y_vals = values[sub_key] if sub_key else values
                if not y_vals: continue

                # Style logic
                style = '-'
                alpha = 0.8
                width = 2

                if 'base' in model_label and 'vs' not in model_label:
                    style = '--' # Base dashed
                    width = 2.5
                elif 'vs' in model_label:
                    width = 2 # Alignment lines

                x_vals = range(len(y_vals))
                plt.plot(x_vals, y_vals, linestyle=style, linewidth=width, marker='o', markersize=6, label=model_label, alpha=alpha)
                has_data = True

            if has_data:
                title = f"{metric_name} - {sub_key}" if sub_key else metric_name
                plt.ylim(0.0, 1.0)

                plt.title(title, fontsize=14)
                plt.xlabel("Layer Depth", fontsize=12)
                plt.ylabel("Value", fontsize=12)
                plt.legend(loc='upper center', ncol=2) # Legend outside
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()

                clean_name = title.replace("/", "_").replace(" ", "_")
                plt.savefig(os.path.join(output_dir, f"{clean_name}.png"), dpi=300)
                logging.info(f"    📸 [Plot] Saved: {clean_name}.png")
                plt.show()
                plt.close()

def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 20)
    logger.info(f"🚀 [Start] Metric Computation")
    logger.info(f"📍 Models: {args.models}")
    logger.info(f"🎨 Variants: {args.variants}")
    logger.info(f"💾 Max Samples: {args.max_samples}")
    logger.info("=" * 20)

    cache_path = get_cache_path(args)
    visualization_store = defaultdict(dict)

    if os.path.exists(cache_path) and not args.force_recompute:
        logger.info(f"✨ [Cache] Loading found cache: {cache_path}")
        visualization_store = torch.load(cache_path)
    else:
        for model_str in args.models:
            family, scale = model_str.split('/')
            logger.info(f"{'='*40}📦 [Model] Processing {family}/{scale}{'='*40}")

            # ------------------------------------------------------------
            # 1. 加载和计算 BASE (常驻内存)
            # ------------------------------------------------------------
            logger.info("  🛡️ [1/3] Processing BASE (Reference)...")

            # A. Base Sequence Metrics
            base_seq_metrics = compute_sequence_metrics_streaming(
                family, scale, 'base', args.dataset,
                args.metrics, args.normalization, args.device,
                batch_size=args.batch_size
            )
            for m, vals in base_seq_metrics.items():
                visualization_store[m][f"{family}-{scale}-base"] = vals

            # B. Base Pooled Data (Load into RAM)
            base_pooled = load_pooled_subset(
                family, scale, 'base', args.dataset,
                max_samples=args.max_samples,
                batch_size=args.batch_size
            )

            if base_pooled:
                # C. Base Global Metrics
                base_global_metrics = compute_pooled_metrics_in_memory(
                    base_pooled, args.metrics, args.normalization, args.device
                )
                for m, vals in base_global_metrics.items():
                    visualization_store[m][f"{family}-{scale}-base"] = vals
            else:
                logger.warning("    ⚠️ Base model data missing! Skipping alignment metrics.")

            # ------------------------------------------------------------
            # 2. 循环处理其他 Variants (SFT, DPO, RLVR, Instruct)
            # ------------------------------------------------------------
            for var in args.variants:
                logger.info(f"🎨 [2/3] Processing Variant: {var}...")

                # A. Variant Sequence Metrics (Streaming)
                var_seq_metrics = compute_sequence_metrics_streaming(
                    family, scale, var, args.dataset,
                    args.metrics, args.normalization, args.device,
                    batch_size=args.batch_size
                )
                if var_seq_metrics:
                    for m, vals in var_seq_metrics.items():
                        visualization_store[m][f"{family}-{scale}-{var}"] = vals
                else:
                    logger.warning(f"    ⚠️ Skipping {var} sequence metrics (no data)")

                # B. Variant Pooled Data (Load -> Compare -> Unload)
                var_pooled = load_pooled_subset(
                    family, scale, var, args.dataset,
                    max_samples=args.max_samples,
                    batch_size=args.batch_size
                )

                if var_pooled:
                    # C. Variant Global Metrics
                    var_global_metrics = compute_pooled_metrics_in_memory(
                        var_pooled, args.metrics, args.normalization, args.device
                    )
                    for m, vals in var_global_metrics.items():
                        visualization_store[m][f"{family}-{scale}-{var}"] = vals

                    # D. Alignment Metrics (Base vs Variant)
                    if base_pooled:
                        logger.info(f"    📐 [Align] Computing Alignment: Base vs {var}")
                        align_metrics = compute_alignment_metrics_safe(
                            base_pooled, var_pooled, args.metrics, args.device
                        )
                        for m, vals in align_metrics.items():
                            # Key naming: family-scale-base-vs-sft
                            visualization_store[m][f"{family}-{scale}-base-vs-{var}"] = vals

                # Clean up Variant data to prevent OOM
                del var_pooled
                gc.collect()
                if args.device == 'cuda': torch.cuda.empty_cache()

            # Clean up Base data after all variants are done
            logger.info("  🧹 [Clean] Unloading Base model data...")
            del base_pooled
            gc.collect()
            if args.device == 'cuda': torch.cuda.empty_cache()

        # Save Cache
        logger.info(f"💾 [Save] Saving cache to {cache_path}")
        torch.save(dict(visualization_store), cache_path)

    # Visualize
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    viz_dir = os.path.join(METRIC_DIR, 'plots', args.dataset, timestamp)
    logger.info(f"📊 [Plot] Generating plots to {viz_dir}...")
    visualize_results(visualization_store, viz_dir)
    logger.info("🎉 [Done] All tasks completed.")

if __name__ == "__main__":
    main()