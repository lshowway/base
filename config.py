"""
Configuration file for models and datasets
"""
import os

# ============================================================================
# Directories
# ============================================================================
DATASET_CACHE_DIR = os.getenv('DATASET_CACHE_DIR', '/root/autodl-tmp/base_sft/dataset_cache')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/root/autodl-tmp/base_sft/model_cache')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/root/autodl-tmp/base_sft/outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
REPRESENTATION_DIR = os.path.join(OUTPUT_DIR, 'representations')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
METRIC_DIR = os.path.join(OUTPUT_DIR, 'metrics')

# ============================================================================
# Model Configurations
# ============================================================================
MODEL_CONFIGS = {
    'mistral': {
        '7b': {
            'base': 'mistralai/Mistral-7B-v0.1',
            'sft': 'mistralai/Mistral-7B-Instruct-v0.1',
            'num_layers': 32,
        },
    },
    'olmo2': {
        # 1B: 完美的科研对象，拥有独立的 RLVR1 中间件 (基于您提供的 0425 版本)
        '1b': {
            'base': 'allenai/OLMo-2-0425-1B',
            'sft': 'allenai/OLMo-2-0425-1B-SFT',
            'dpo': 'allenai/OLMo-2-0425-1B-DPO',
            'rlvr': 'allenai/OLMo-2-0425-1B-RLVR1', # 独立的 RLVR 检查点
            'instruct': 'allenai/OLMo-2-0425-1B-Instruct', # 最终成品
            'num_layers': 16, # Estimated for 1B size
        },
        # 7B: 标准 Tulu 3 流程 (Base -> SFT -> DPO -> Instruct/RLVR)
        '7b': {
            'base': 'allenai/OLMo-2-1124-7B',
            'sft': 'allenai/OLMo-2-1124-7B-SFT',
            'dpo': 'allenai/OLMo-2-1124-7B-DPO',
            'rlvr': 'allenai/OLMo-2-1124-7B-Instruct', # 7B/13B 的 Instruct 通常即为 RLVR 后的产物
            # 'instruct': 'allenai/OLMo-2-1124-7B-Instruct',
            'num_layers': 32, # Confirmed
        },
        # 13B: 中间件齐全，适合做中等规模实验
        '13b': {
            'base': 'allenai/OLMo-2-1124-13B',
            'sft': 'allenai/OLMo-2-1124-13B-SFT',
            'dpo': 'allenai/OLMo-2-1124-13B-DPO',
            'rlvr': 'allenai/OLMo-2-1124-13B-Instruct',
            # 'instruct': 'allenai/OLMo-2-1124-13B-Instruct',
            'num_layers': 40, # Confirmed
        },
        # 32B: 最新的大杯模型 (0325 版本)
        '32b': {
            'base': 'allenai/OLMo-2-0325-32B',
            'sft': 'allenai/OLMo-2-0325-32B-SFT',
            'dpo': 'allenai/OLMo-2-0325-32B-DPO', # 存在于官方 Repo 树中
            'rlvr': 'allenai/OLMo-2-0325-32B-Instruct',
            # 'instruct': 'allenai/OLMo-2-0325-32B-Instruct',
            'num_layers': 64, # Estimated, verify with config.json
        },
    },
    'gemma3': {
        '1b': {
            'base': 'google/gemma-3-1b-pt',
            'instruct': 'google/gemma-3-1b-it',
            'num_layers': 18,
        },
        '27b': {
            'base': 'google/gemma-3-27b-pt',
            'instruct': 'google/gemma-3-27b-it',
            'num_layers': 46,
        },
    },
    'qwen25': {
        '7b': {
            'base': 'Qwen/Qwen2.5-7B',
            'instruct': 'Qwen/Qwen2.5-7B-Instruct',
            'num_layers': 28,
        },
        '14b': {
            'base': 'Qwen/Qwen2.5-14B',
            'instruct': 'Qwen/Qwen2.5-14B-Instruct',
            'num_layers': 48,
        },
        '32b': {
            'base': 'Qwen/Qwen2.5-32B',
            'instruct': 'Qwen/Qwen2.5-32B-Instruct',
            'num_layers': 64,
        },
        '72b': {
            'base': 'Qwen/Qwen2.5-72B',
            'instruct': 'Qwen/Qwen2.5-72B-Instruct',
            'num_layers': 80,
        },
    },
    'llama32': {
        '1b': {
            'base': 'meta-llama/Llama-3.2-1B',
            'instruct': 'meta-llama/Llama-3.2-1B-Instruct',
            'num_layers': 16,
        },
        '3b': {
            'base': 'meta-llama/Llama-3.2-3B',
            'instruct': 'meta-llama/Llama-3.2-3B-Instruct',
            'num_layers': 28,
        },
    },
}
# ============================================================================
# Dataset Configurations
# ============================================================================
MAX_LENGTH = 128
N_POOLED = 200
N_TOKEN = 50
DATASET_CONFIGS = {
    # Token-level datasets: save both pooled (1000 samples) and token-level (100 samples)
    'mmlu': {
        'hf_name': 'cais/mmlu',
        'subset': 'all',
        'split': 'test',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,  # Number of samples for pooled representations
        'n_samples_token': N_TOKEN,     # Number of samples for token-level representations
        'max_length': MAX_LENGTH,
        'format_type': 'multiple_choice',
    },
    'gsm8k': {
        'hf_name': 'openai/gsm8k',
        'subset': 'main',
        'split': 'test',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'qa',
    },
    'wikitext': {
        'hf_name': 'Salesforce/wikitext',
        'subset': 'wikitext-103-v1',
        'split': 'test',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'text',
    },
    'ifeval': {
        'hf_name': 'google/IFEval',
        'subset': None,
        'split': 'train',
        'type': 'token-level',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': N_TOKEN,
        'max_length': MAX_LENGTH,
        'format_type': 'instruction',
    },

    # Pooled-only datasets
    'humaneval': {
        'hf_name': 'openai_humaneval',
        'subset': None,
        'split': 'test',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'code',
    },
    'mt_bench': {
        'hf_name': 'HuggingFaceH4/mt_bench_prompts',
        'subset': None,
        'split': 'train',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'conversation',
    },
    'toxigen': {
        'hf_name': 'toxigen/toxigen-data',
        'subset': 'annotated',
        'split': 'test',
        'type': 'pooled',
        'n_samples_pooled': N_POOLED,
        'n_samples_token': 0,
        'max_length': MAX_LENGTH,
        'format_type': 'text_classification',
    },
}

# ============================================================================
# Default Parameters
# ============================================================================
# DEFAULT_BATCH_SIZE = 32
# DEFAULT_SEED = 42
# DEFAULT_DTYPE = 'float32'