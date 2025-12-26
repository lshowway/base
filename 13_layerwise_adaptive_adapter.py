"""
13_layerwise_adaptive_adapter_new.py
E13: Layer-wise Adaptive Adapter (Batch Optimized & Visualized)

Key Updates:
1. True Batch Generation (Speedup x10).
2. 'Pretrained' Baseline added.
3. Multi-Rank Support (Run multiple configs at once).
4. Correct Parameter Counting (using layers_to_transform).
5. Rich Visualization (Rank Plots & Efficiency Plots).
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
from torch.utils.data import DataLoader

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
    colors = {'green': '\033[92m', 'red': '\033[91m', 'blue': '\033[94m', 'yellow': '\033[93m', 'default': '\033[0m', 'purple': '\033[95m'}
    print(f"{colors.get(color, '')}{text}{colors['default']}")

def parse_args():
    parser = argparse.ArgumentParser(description='E13: Layer-wise Adaptive Adapter')

    # Model & Data
    parser.add_argument('--model', type=str, default='olmo2/7b', help='Format: family/scale')
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=list(DATASET_CONFIGS.keys()))

    # Strategy Params
    parser.add_argument('--split_k', type=float, default=0.8, help='Front K% layers')

    # Multi-Rank Support: Pass groups of 3 integers
    # e.g. --ranks 32 -1 64  (One config)
    # e.g. --ranks 32 -1 64 16 -1 32 (Two configs)
    parser.add_argument('--ranks', type=int, nargs='+', default=[64, 80, -1],
                        help='Groups of 3: [Baseline, Front, Back]. E.g. "32 -1 64 16 -1 32"')

    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)

    # Training Config
    parser.add_argument('--num_epochs', type=float, default=2.0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--grad_accum', type=int, default=4)

    parser.add_argument('--n_train_samples', type=int, default=5000)
    parser.add_argument('--n_test_samples', type=int, default=100)

    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_DIR, 'adaptive_adapter_viz'))
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

# ============================================================================
# Logic Helpers
# ============================================================================

def parse_rank_configs(ranks_arg):
    """Parse flat list of ranks into groups of 3"""
    if len(ranks_arg) % 3 != 0:
        raise ValueError(f"Ranks argument must be multiples of 3 (Baseline, Front, Back). Got {len(ranks_arg)} items.")
    return [ranks_arg[i:i+3] for i in range(0, len(ranks_arg), 3)]

def calculate_rank_allocation(num_layers, k, r_front, r_back):
    split_idx = int(num_layers * k)
    rank_pattern = {}
    active_layers = []

    for i in range(num_layers):
        if i < split_idx:
            rank = 0 if r_front == -1 else r_front
        else:
            rank = 0 if r_back == -1 else r_back

        rank_pattern[i] = rank
        if rank > 0:
            active_layers.append(i)

    return rank_pattern, active_layers

def verify_parameter_budget(num_layers, baseline_rank, allocation):
    total_baseline = num_layers * baseline_rank
    if isinstance(allocation, tuple): allocation = allocation[0]
    total_opt = sum(allocation.values())

    cprint(f"  > Rank Units: Baseline={total_baseline} | Optimized={total_opt}", 'blue')
    return total_opt

def load_data_custom(dataset_name, split_key, n_samples):
    from datasets import load_dataset as hf_load
    cfg = DATASET_CONFIGS[dataset_name]
    target_split = cfg.get(split_key)
    if not target_split:
        raise ValueError(f"Split '{split_key}' not defined for {dataset_name}")

    cprint(f"Loading {dataset_name} | Split: {target_split}", 'yellow')
    try:
        dataset = hf_load(cfg['hf_name'], cfg['subset'], split=target_split, cache_dir=DATASET_CACHE_DIR)
        if 0 < n_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(n_samples))
        return dataset
    except Exception as e:
        cprint(f"Failed to load dataset: {e}", 'red')
        raise e

def prepare_dataset_for_sft(dataset, tokenizer, dataset_name):
    max_len = DATASET_CONFIGS[dataset_name].get('max_length', 512)
    def process_fn(example):
        text = format_sample(dataset_name, example)
        if not text: text = ""
        text = text + tokenizer.eos_token
        return tokenizer(text, truncation=True, max_length=max_len)
    return dataset.map(process_fn, remove_columns=dataset.column_names)

# ============================================================================
# Core Evaluation Logic (Batch Optimized)
# ============================================================================

class DatasetEvaluator:
    def __init__(self, tokenizer, dataset_name, device):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.metric_mode = self.config.get('metric_mode', 'loss')
        self.device = device
        self.max_len = self.config.get('max_length', 512)

        # Ensure pad token is set for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def extract_answer(self, text):
        text = text.strip()
        if self.dataset_name == 'gsm8k':
            nums = re.findall(r'-?[\d,]+\.?\d*', text)
            if not nums: return None
            clean_num = nums[-1].replace(',', '').replace('$', '')
            try: return float(clean_num)
            except: return None
        elif self.dataset_name == 'mmlu':
            match = re.search(r'Answer:\s*([A-D])', text, re.IGNORECASE)
            if match: return match.group(1).upper()
            match_end = re.search(r'\b([A-D])\s*$', text, re.IGNORECASE)
            if match_end: return match_end.group(1).upper()
            return None
        return text

    def run_evaluation(self, model, dataset, batch_size):
        cprint(f"Starting Evaluation for {self.dataset_name} (Mode: {self.metric_mode})", 'purple')
        if self.metric_mode == 'exact_match':
            return self._evaluate_generation_batch(model, dataset, batch_size)
        else:
            return self._evaluate_perplexity(model, dataset, batch_size)

    def _evaluate_generation_batch(self, model, dataset, batch_size):
        correct = 0
        total = 0
        model.eval()

        # Pre-format dataset to avoid processing in loop
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

        # Create batches
        num_samples = len(formatted_prompts)
        pbar = tqdm(range(0, num_samples, batch_size), desc="Generating (Batch)", unit="batch")

        for i in pbar:
            batch_prompts = formatted_prompts[i : i + batch_size]
            batch_golds = golds[i : i + batch_size]

            if not batch_prompts: continue

            # Tokenize Batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_len
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64 if self.dataset_name != 'humaneval' else 512,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Batch Decode
            # Only decode the new tokens
            input_len = inputs.input_ids.shape[1]
            gen_texts = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

            # Compare
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

            pbar.set_postfix({'acc': f"{correct/total:.4f}"})

        return correct / total if total > 0 else 0.0

    def _evaluate_perplexity(self, model, dataset, batch_size):
        total_loss = 0.0
        total_steps = 0
        model.eval()

        # Prepare valid texts
        valid_texts = []
        for i in range(len(dataset)):
            t = format_sample(self.dataset_name, dataset[i])
            if t: valid_texts.append(t + self.tokenizer.eos_token)

        pbar = tqdm(range(0, len(valid_texts), batch_size), desc="Calculating PPL")

        for i in pbar:
            batch_texts = valid_texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=inputs.input_ids)
                loss = outputs.loss.item()

            total_loss += loss * len(batch_texts)
            total_steps += len(batch_texts)
            pbar.set_postfix({'loss': f"{total_loss/total_steps:.4f}"})

        avg_loss = total_loss / total_steps if total_steps > 0 else 99.0

        if self.metric_mode == 'perplexity':
            try: return math.exp(avg_loss)
            except OverflowError: return 9999.0
        else:
            return avg_loss

# ============================================================================
# Visualization
# ============================================================================

def visualize_results(results_df, rank_allocations, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Rank Allocation (Line Plot)     plt.figure(figsize=(10, 6), dpi=300)
    for name, allocation in rank_allocations.items():
        if name == 'Pretrained': continue
        layers = sorted(allocation.keys())
        ranks = [allocation[l] for l in layers]
        plt.plot(layers, ranks, marker='o', linewidth=2, label=name)

    plt.title('Rank Allocation Strategy')
    plt.xlabel('Layer Index')
    plt.ylabel('LoRA Rank')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rank_allocation.png'))
    plt.close()

    # Plot 2: Performance vs Efficiency (Scatter)     plt.figure(figsize=(10, 6), dpi=300)

    # Filter out Pretrained for param comparison (or plot it at 0 params)
    plot_df = results_df.copy()

    # Normalize params for Pretrained to 0 for visualization or exclude
    # Let's plot Pretrained as a horizontal line
    pretrained_score = plot_df[plot_df['Strategy'] == 'Pretrained']['Score'].values
    if len(pretrained_score) > 0:
        plt.axhline(y=pretrained_score[0], color='gray', linestyle='--', label='Pretrained Baseline')
        plot_df = plot_df[plot_df['Strategy'] != 'Pretrained']

    sns.scatterplot(data=plot_df, x='RankUnits', y='Score', hue='Strategy', s=200, style='Strategy')

    # Add labels
    for i, row in plot_df.iterrows():
        plt.text(row['RankUnits'], row['Score'], f"{row['Score']:.4f}",
                 va='bottom', ha='center', fontsize=9)

    plt.title('Performance vs Parameter Efficiency')
    plt.xlabel('Total Rank Units (Proxy for Trainable Params)')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'efficiency_scatter.png'))
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    args = parse_args()

    if args.use_wandb:
        wandb.init(project="layerwise_adapter", config=vars(args))

    cprint(f"🚀 Experiment: {args.model} | {args.dataset}", 'purple')
    cprint(f"   Batch Sizes -> Train: {args.train_batch_size} | Eval: {args.eval_batch_size}", 'blue')

    # Setup Data
    train_ds = load_data_custom(args.dataset, 'split_train', args.n_train_samples)
    test_ds = load_data_custom(args.dataset, 'split_test', args.n_test_samples)

    family, scale = args.model.split('/')
    num_layers = get_num_layers(family, scale)

    # Define Strategies
    strategies = {}
    rank_configs = parse_rank_configs(args.ranks)

    # 1. Add Pretrained (Zero-shot)
    strategies['Pretrained'] = {'type': 'pretrained', 'allocation': {}, 'active_layers': []}

    # 2. Add Configured Strategies
    for idx, (base_r, front_r, back_r) in enumerate(rank_configs):
        # Baseline (Uniform)
        base_name = f"Uniform_r{base_r}_Set{idx+1}"
        strategies[base_name] = {
            'type': 'lora',
            'allocation': {i: base_r for i in range(num_layers)},
            'active_layers': list(range(num_layers))
        }

        # Optimized (Split)
        opt_name = f"Split_r{front_r}-{back_r}_Set{idx+1}"
        alloc, active = calculate_rank_allocation(num_layers, args.split_k, front_r, back_r)
        strategies[opt_name] = {
            'type': 'lora',
            'allocation': alloc,
            'active_layers': active
        }

    cprint(f"📋 Strategies to Run: {list(strategies.keys())}", 'blue')

    results_list = []

    for name, strategy_cfg in strategies.items():
        cprint(f"\n{'='*20} Running {name} {'='*20}", 'blue')

        # Load Model
        model, tokenizer = load_model(family, scale, 'base', dtype='bfloat16', device_map='auto')

        # Apply LoRA (if not pretrained)
        total_rank_units = 0
        trainable_params = 0

        if strategy_cfg['type'] == 'lora':
            allocation = strategy_cfg['allocation']
            active_layers = strategy_cfg['active_layers']
            total_rank_units = sum(allocation.values())

            cprint(f"  > Rank Units: {total_rank_units}", 'blue')

            # --- CRITICAL FIX FOR PARAMETER COUNTING ---
            # We must use 'layers_to_transform' to explicitly tell PEFT which layers to adapt.
            # Otherwise, it might create adapters (even with rank 0 logic) or miscount.
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=max(allocation.values()),
                rank_pattern=allocation,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                layers_to_transform=active_layers # This ensures frozen layers have NO adapters
            )
            model = get_peft_model(model, peft_config)

            # Print Params
            trainable_params, all_params = model.get_nb_trainable_parameters()
            cprint(f"  > Trainable Params: {trainable_params:,} ({trainable_params/all_params:.4%})", 'purple')

            # SFT
            cprint("  > Starting SFT...", 'yellow')
            train_tokenized = prepare_dataset_for_sft(train_ds, tokenizer, args.dataset)

            training_args = TrainingArguments(
                output_dir=os.path.join(args.output_dir, f"{name}_ckpt"),
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=args.learning_rate,
                logging_steps=10,
                save_strategy="no",
                report_to="none", # We log manually or via main wandb
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
            cprint("  > Pretrained Mode (Skipping SFT)", 'yellow')

        # Evaluation
        evaluator = DatasetEvaluator(tokenizer, args.dataset, next(model.parameters()).device)
        score = evaluator.run_evaluation(model, test_ds, args.eval_batch_size)

        cprint(f"🏁 {name} Result: {score:.4f}", 'green')

        results_list.append({
            'Strategy': name,
            'RankUnits': total_rank_units,
            'Params': trainable_params,
            'Score': score
        })

        if args.use_wandb:
            wandb.log({f"{name}_score": score, f"{name}_params": trainable_params})

        del model, tokenizer, evaluator
        if 'trainer' in locals(): del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary & Visualization
    results_df = pd.DataFrame(results_list)
    cprint(f"\nFinal Results:\n{results_df}", 'blue')

    # Save CSV
    results_df.to_csv(os.path.join(args.output_dir, 'final_results.csv'), index=False)

    # Extract allocations for plotting
    allocations = {n: s['allocation'] for n, s in strategies.items() if s['type'] == 'lora'}
    visualize_results(results_df, allocations, args.output_dir)
    cprint(f"✅ Saved plots and results to {args.output_dir}", 'green')

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()