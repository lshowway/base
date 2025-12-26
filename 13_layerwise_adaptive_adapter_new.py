"""
13_layerwise_adaptive_adapter.py
E13: Segment-wise Adaptive Adapter (Transfer Learning & Depth Control)

Hypothesis:
    Based on model swapping results, modifying middle layers might add information
    without overwriting critical base capabilities (head/tail).

Features:
1. Dynamic Segmentation: Rank list length determines segments.
2. Depth Control: Apply LoRA only to the last K layers of each segment.
3. Transfer Testing: Train on A, Test on [A, B, C...].
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import torch
import argparse
import logging
import gc
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# HF & PEFT
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# === Project Imports ===
sys.path.insert(0, os.getcwd())
from config import (
    MODEL_CONFIGS, DATASET_CONFIGS,
    OUTPUT_DIR, DATASET_CACHE_DIR, MODEL_CACHE_DIR
)
from model_utils import load_model, get_num_layers
from data_utils import format_sample

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Plot Styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("tab10")


def cprint(text, color='default'):
    colors = {'green': '\033[92m', 'red': '\033[91m', 'blue': '\033[94m', 'yellow': '\033[93m', 'default': '\033[0m',
              'purple': '\033[95m'}
    print(f"{colors.get(color, '')}{text}{colors['default']}")


def parse_args():
    parser = argparse.ArgumentParser(description='E13: Segment-wise Adaptive Adapter')

    # Model
    parser.add_argument('--model', type=str, default='olmo2/1b', help='Format: family/scale')

    # Data: Train on one, Test on many
    parser.add_argument('--train_dataset', type=str, default='gsm8k',
                        choices=list(DATASET_CONFIGS.keys()), help='Dataset used for SFT')
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['gsm8k', 'mmlu', 'wikitext', 'ifeval', 'humaneval', 'mt_bench', 'toxigen'],
                        help='Datasets used for evaluation (Transfer capabilities)')

    # Strategy Params (Flexible)
    # Example: "8,8,8,8" (Uniform 4 segs) or "-1,32,-1" (Middle Only)
    parser.add_argument('--adapter_configs', type=str, nargs='+',
                        # default=['8,8,8,8,8', '-1,-1,40,-1,-1', '-1,12,12,12,-1', '-1,-1,-1,-1,32'],
                        default=[
                        '8,8,8,8,8',  # 1. Baseline (Uniform)
                        '-1,-1,40,-1,-1',  # 2. 中间单峰 (Seg 3 Only)
                        '-1,12,12,12,-1',  # 3. 中间宽幅 (Seg 2,3,4) -> Sum=36 < 40
                        '-1,20,20,-1,-1',  # 4. 中间靠左-包括中间 (Seg 2,3)
                        '-1,40,-1,-1,-1',  # 5. 中间靠左-不包中间 (Seg 2 Only)
                        '-1,-1,20,20,-1',  # 6. 中间靠右-包括中间 (Seg 3,4)
                        '-1,-1,-1,40,-1',  # 7. 中间靠右-不包中间 (Seg 4 Only)
                        '-1,-1,-1,-1,40',  # 8. 高层语义强化 (Tail/Seg 5)
                        '40,-1,-1,-1,-1',  # 9. 低层特征强化 (Head/Seg 1)
                            ],
                        help='List of comma-separated ranks. Length defines num_segments. -1 means Frozen.')

    parser.add_argument('--adapter_depth', type=int, default=100,
                        help='Max layers to adapt within each segment (from the end of segment). Default 100 means full segment.')

    # LoRA Params
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Training Config
    parser.add_argument('--num_epochs', type=float, default=2.0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--train_batch_size', type=int, default=32)  # Reduced for safety
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--grad_accum', type=int, default=4)

    parser.add_argument('--n_train_samples', type=int, default=2000)
    parser.add_argument('--n_test_samples', type=int, default=100)

    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'adaptive_adapter_viz'))
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ============================================================================
# Logic Helpers
# ============================================================================

def parse_rank_string(config_str):
    """
    Input: "8,8,-1,32"
    Output: [8, 8, -1, 32]
    """
    try:
        return [int(x) for x in config_str.split(',')]
    except ValueError:
        raise ValueError(f"Invalid rank config string: {config_str}")


def calculate_segmented_rank_pattern(num_layers, rank_list, depth_per_seg):
    """
    Maps the abstract segment ranks to actual layer indices.
    Logic:
    1. Divide layers into len(rank_list) segments.
    2. For each segment, select the LAST 'depth_per_seg' layers.
    3. Apply rank. If rank is -1, it becomes 0 (Frozen).
    4. Layers not selected in step 2 are implicitly Frozen (Rank 0).
    """
    num_segments = len(rank_list)
    segment_size = num_layers / num_segments

    rank_pattern = {}  # layer_idx -> rank
    active_layers = []

    # Initialize all to 0
    for i in range(num_layers):
        rank_pattern[i] = 0

    for i, rank_val in enumerate(rank_list):
        if rank_val <= 0: continue  # Skip frozen segments (-1 or 0)

        # Calculate Segment Range
        seg_start_idx = int(i * segment_size)
        seg_end_idx = int((i + 1) * segment_size)
        seg_end_idx = min(seg_end_idx, num_layers)

        # Calculate Active Depth within Segment
        # e.g., Segment [0, 6), depth=2 -> Apply to 4, 5
        active_start = max(seg_start_idx, seg_end_idx - depth_per_seg)

        for layer_idx in range(active_start, seg_end_idx):
            rank_pattern[layer_idx] = rank_val
            active_layers.append(layer_idx)

    return rank_pattern, active_layers


def load_data_custom(dataset_name, split_key, n_samples):
    from datasets import load_dataset as hf_load

    if dataset_name not in DATASET_CONFIGS:
        cprint(f"Warning: {dataset_name} not in DATASET_CONFIGS, skipping...", 'red')
        return None

    cfg = DATASET_CONFIGS[dataset_name]
    target_split = cfg.get(split_key)

    # Fallback if split not defined (e.g. humaneval only has test)
    if not target_split:
        if split_key == 'split_train':
            target_split = 'train'
        else:
            target_split = 'test'

    cprint(f"Loading {dataset_name} | Split: {target_split}", 'yellow')
    try:
        if cfg['subset']:
            dataset = hf_load(cfg['hf_name'], cfg['subset'], split=target_split, cache_dir=DATASET_CACHE_DIR)
        else:
            dataset = hf_load(cfg['hf_name'], split=target_split, cache_dir=DATASET_CACHE_DIR)

        if 0 < n_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(n_samples))
        return dataset
    except Exception as e:
        cprint(f"Failed to load dataset {dataset_name}: {e}", 'red')
        return None


def prepare_dataset_for_sft(dataset, tokenizer, dataset_name):
    max_len = DATASET_CONFIGS[dataset_name].get('max_length', 512)

    def process_fn(example):
        text = format_sample(dataset_name, example)
        if not text: text = ""
        text = text + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=max_len)

    return dataset.map(process_fn, remove_columns=dataset.column_names)


# ============================================================================
# Core Evaluation Logic (Maintained)
# ============================================================================

class DatasetEvaluator:
    def __init__(self, tokenizer, dataset_name, device):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS.get(dataset_name, {})
        self.metric_mode = self.config.get('metric_mode', 'loss')
        self.device = device
        self.max_len = self.config.get('max_length', 512)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def extract_answer(self, text):
        text = text.strip()
        if self.dataset_name == 'gsm8k':
            nums = re.findall(r'-?[\d,]+\.?\d*', text)
            if not nums: return None
            clean_num = nums[-1].replace(',', '').replace('$', '')
            try:
                return float(clean_num)
            except:
                return None
        elif self.dataset_name == 'mmlu':
            match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
            if match: return match.group(1).upper()
            match_end = re.search(r'\b([A-D])\s*$', text, re.IGNORECASE)
            if match_end: return match_end.group(1).upper()
            return None
        return text

    def run_evaluation(self, model, dataset, batch_size):
        # cprint(f"Evaluating on {self.dataset_name} (Mode: {self.metric_mode})", 'purple')
        if self.metric_mode == 'exact_match':
            return self._evaluate_generation_batch(model, dataset, batch_size)
        else:
            return self._evaluate_perplexity(model, dataset, batch_size)

    def _evaluate_generation_batch(self, model, dataset, batch_size):
        correct = 0
        total = 0
        model.eval()
        formatted_prompts = []
        golds = []

        for example in dataset:
            prompt = format_sample(self.dataset_name, example)
            if not prompt: continue
            if 'Answer:' in prompt:
                input_prompt = prompt.split('Answer:')[0] + 'Answer:'
                gold_raw = prompt.split('Answer:')[1].strip()
            else:
                input_prompt = prompt
                gold_raw = example.get('answer', '') or example.get('canonical_solution', '')
            formatted_prompts.append(input_prompt)
            golds.append(gold_raw)

        num_samples = len(formatted_prompts)
        # Quiet tqdm for eval
        for i in range(0, num_samples, batch_size):
            batch_prompts = formatted_prompts[i: i + batch_size]
            batch_golds = golds[i: i + batch_size]
            if not batch_prompts: continue

            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True,
                                    max_length=self.max_len).to(self.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id)

            input_len = inputs.input_ids.shape[1]
            gen_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            for gen_text, gold_raw in zip(gen_texts, batch_golds):
                pred_val = self.extract_answer(gen_text)
                is_correct = False
                if self.dataset_name == 'gsm8k':
                    gold_val = self.extract_answer(gold_raw)
                    if pred_val is not None and gold_val is not None:
                        if abs(pred_val - gold_val) < 1e-3: is_correct = True
                elif self.dataset_name == 'mmlu':
                    gold_val = self.extract_answer(gold_raw)
                    if pred_val == gold_val: is_correct = True
                if is_correct: correct += 1
                total += 1
        return correct / total if total > 0 else 0.0

    def _evaluate_perplexity(self, model, dataset, batch_size):
        total_loss = 0.0
        total_steps = 0
        model.eval()
        valid_texts = []
        for i in range(len(dataset)):
            t = format_sample(self.dataset_name, dataset[i])
            if isinstance(t, list): t = t[0] if t else ""  # 兼容修复：如果返回的是列表（如mt_bench），取第一个元素
            if t: valid_texts.append(t + self.tokenizer.eos_token)

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i: i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_len,
                                    return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                labels=inputs.input_ids)
            total_loss += outputs.loss.item() * len(batch_texts)
            total_steps += len(batch_texts)

        avg_loss = total_loss / total_steps if total_steps > 0 else 99.0
        if self.metric_mode == 'perplexity':
            try:
                return math.exp(avg_loss)
            except:
                return 9999.0
        return avg_loss


# ============================================================================
# Visualization (Updated for Transfer Results)
# ============================================================================

def visualize_multi_dataset_results(results_df, output_dir, train_dataset):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Pivot for Heatmap-style or Bar Chart
    # Columns: Strategy, TestDataset, Score
    datasets = results_df['TestDataset'].unique()
    strategies = results_df['Strategy'].unique()

    # --- Plot 1: Transfer Performance Comparison (Grouped Bar) ---
    plt.figure(figsize=(8, 6), dpi=300)
    sns.barplot(data=results_df, x='TestDataset', y='Score', hue='Strategy', palette='viridis')
    plt.title(f"Transfer Learning Performance (Trained on {train_dataset})")
    plt.ylabel("Score (Acc or Low PPL)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transfer_performance.png'))
    plt.show()
    plt.close()

    # --- Plot 2: Efficiency vs Main Task (Scatter) ---
    # Filter only for the training dataset to see "Learning Efficiency"
    train_subset = results_df[results_df['TestDataset'] == train_dataset].copy()
    if not train_subset.empty:
        plt.figure(figsize=(8, 6), dpi=300)
        sns.scatterplot(data=train_subset, x='Params', y='Score', hue='Strategy', s=400, style='Strategy')
        for i, row in train_subset.iterrows():
            plt.text(row['Params'], row['Score'], f"{row['Strategy']}", fontsize=14, alpha=0.7)
        plt.title(f"Parameter Efficiency on Training Task ({train_dataset})")
        plt.xlabel("Trainable Parameters")
        plt.ylabel(f"Score on {train_dataset}")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_scatter_train_task.png'))
        plt.show()
        plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    args = parse_args()
    if args.use_wandb:
        wandb.init(project="layerwise_adapter_transfer", config=vars(args))

    cprint(f"🚀 Experiment: {args.model} | Train: {args.train_dataset}", 'purple')
    cprint(f"   Test Transfer: {args.test_datasets}", 'purple')
    cprint(f"   Adapter Depth per Segment: {args.adapter_depth}", 'blue')

    family, scale = args.model.split('/')
    num_layers = get_num_layers(family, scale)
    cprint(f"   Model Layers: {num_layers}", 'blue')

    # 1. Load Data
    # ----------------
    train_ds = load_data_custom(args.train_dataset, 'split_train', args.n_train_samples)
    if train_ds is None:
        raise ValueError("Training dataset failed to load.")

    test_datasets_map = {}
    for ds_name in args.test_datasets:
        ds = load_data_custom(ds_name, 'split_test', args.n_test_samples)
        if ds: test_datasets_map[ds_name] = ds

    # 2. Define Strategies from Strings
    # ----------------
    strategies = {}

    # Always include a Pretrained Baseline
    strategies['Pretrained'] = {'type': 'pretrained', 'allocation': {}, 'active_layers': []}

    for config_str in args.adapter_configs:
        ranks = parse_rank_string(config_str)
        # Create a descriptive name
        # e.g. "Seg4_Deep" or "Seg3_MidOnly"
        if all(r == ranks[0] for r in ranks):
            name = f"Uniform_r{ranks[0]}"
        elif ranks[0] == -1 and ranks[-1] == -1:
            name = f"MidOnly_{config_str.replace(',', '_')}"
        else:
            name = f"Pattern_{config_str.replace(',', '_')}"

        alloc, active = calculate_segmented_rank_pattern(num_layers, ranks, args.adapter_depth)

        strategies[name] = {
            'type': 'lora',
            'allocation': alloc,
            'active_layers': active
        }

    cprint(f"📋 Strategies: {list(strategies.keys())}", 'blue')
    results_list = []

    # 3. Execution Loop
    # ----------------
    for name, strategy_cfg in strategies.items():
        cprint(f"\n{'=' * 20} Running {name} {'=' * 20}", 'blue')

        # Load Base Model each time to avoid contamination
        model, tokenizer = load_model(family, scale, 'base', dtype='bfloat16', device_map='auto')

        trainable_params = 0
        total_rank_units = 0

        # Apply LoRA
        if strategy_cfg['type'] == 'lora':
            allocation = strategy_cfg['allocation']
            active_layers = strategy_cfg['active_layers']
            total_rank_units = sum(allocation.values())

            # Check if valid config (might be all -1)
            max_rank = max(allocation.values()) if allocation else 0

            if max_rank > 0 and len(active_layers) > 0:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=max_rank,  # Max rank needed for initialization
                    rank_pattern=allocation,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                    bias="none",
                    layers_to_transform=active_layers
                )
                model = get_peft_model(model, peft_config)
                trainable_params, all_params = model.get_nb_trainable_parameters()
                cprint(f"  > Active Layers: {len(active_layers)} | Params: {trainable_params:,}", 'purple')

                # Train
                cprint(f"  > SFT on {args.train_dataset}...", 'yellow')
                train_tokenized = prepare_dataset_for_sft(train_ds, tokenizer, args.train_dataset)

                training_args = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, f"{name}_ckpt"),
                    num_train_epochs=args.num_epochs,
                    per_device_train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    learning_rate=args.learning_rate,
                    logging_steps=10,
                    save_strategy="no",
                    report_to="none",
                    bf16=True,
                    dataloader_num_workers=0
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_tokenized,
                    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
                )
                trainer.train()
            else:
                cprint("  > Config resulted in 0 active layers (Frozen). Treating as Pretrained.", 'yellow')

        # Evaluate on ALL datasets
        device = next(model.parameters()).device

        for test_name, test_ds in test_datasets_map.items():
            # cprint(f"  > Testing on {test_name}...", 'white')
            evaluator = DatasetEvaluator(tokenizer, test_name, device)
            score = evaluator.run_evaluation(model, test_ds, args.eval_batch_size)

            results_list.append({
                'Strategy': name,
                'TestDataset': test_name,
                'Score': score,
                'Params': trainable_params,
                'RankUnits': total_rank_units
            })
            cprint(f"    -> {test_name}: {score:.4f}", 'green')

        del model, tokenizer
        if 'trainer' in locals(): del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Save & Plot
    # ----------------
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(args.output_dir, f'transfer_results_{args.train_dataset}.csv')
    results_df.to_csv(csv_path, index=False)
    cprint(f"\nSaved results to {csv_path}", 'blue')

    visualize_multi_dataset_results(results_df, args.output_dir, args.train_dataset)

    if args.use_wandb:
        wandb.log({"final_table": wandb.Table(dataframe=results_df)})
        wandb.finish()


if __name__ == '__main__':
    main()