import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os 

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 多卡时，须注释掉

def load_gpt2_model(model_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    print("Model Loaded")

    return model, tokenizer


def load_llm_model(model_name_or_path):
    print('Loading: ', model_name_or_path)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device_count = torch.cuda.device_count()
    print('CUDA device count:', device_count)

    # 设定每个设备的最大内存限制
    devices = [i for i in range(device_count)] if device_count > 0 else ["cpu"]
    # max_memory = {k: '30GB' for k in devices}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir="/root/dataDisk/SA/tmp/pycharm_project_924",
                                                 # low_cpu_mem_usage=False if 'cpu' in devices else True ,
                                                 # device_map=None if 'cpu' in devices else 'auto',
                                                #  load_in_4bit=True,
                                                 # load_in_8bit=True,
                                                 low_cpu_mem_usage=True,
                                                 device_map='auto',
                                                 torch_dtype=torch.float32,
                                                 # max_memory=max_memory
                                                 )
    return model, tokenizer


def load_gemma2(model_name_or_path):
    from transformers.models.gemma2 import Gemma2ForCausalLM
    from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    max_memory = {k: '32GB' for k in global_devices}
    tokenizer = GemmaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = Gemma2ForCausalLM.from_pretrained(model_name_or_path, cache_dir='../../',
                                             low_cpu_mem_usage=True, device_map='auto', # 多卡需要
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    return model, tokenizer


def load_llama(model_name_or_path):
    from transformers import LlamaForCausalLM
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '32GB' for k in global_devices}
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir='../../',
                                             low_cpu_mem_usage=True, device_map='auto',
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    return model, tokenizer


def load_models(model_name_or_path):
    if 'gemma2' in model_name_or_path:
        return load_gemma2(model_name_or_path)
    elif 'llama' in model_name_or_path.lower():
        return load_llama(model_name_or_path)
    else:
        raise ValueError('Model not found')

MODELS = [
    "/opt/models/Llama-2-13b-hf"
    # "../../Qwen2-7B"
    # "../../gemma2-9b"
    # "../../llama2-13b"
    # "../../Llama-2-7b-hf",
    # "/data0/models-yh/gpt2",
    # '../models/gemma',
    # "../models/Llama-2-13b-hf"
    # "../models/Mistral-7B"
    # "../models/Llama2"
    # '/home/incoming/LLM/llama2/llama2-7b',
    # '/home/incoming/LLM/mistral/mistral-7b-v0_1',
    # '/home/incoming/LLM/gemma/gemma-7b',
    # '/home/incoming/LLM/llama3/llama3-8b',
    # '/home/incoming/LLM/llama2/llama2-13b',
    # '/home/incoming/LLM/llama1/llama-30b',
    # '/home/incoming/LLM/qwen1_5/qwen1_5-7b',
    # '/home/incoming/LLM/qwen1_5/qwen1_5-14b',
    # '/home/incoming/LLM/qwen1_5/qwen1_5-32b-chat'
]
