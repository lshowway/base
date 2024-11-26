import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import torch

def plot_ffn_activations(ffn_activations_data1: List[Dict], ffn_activations_data2: List[Dict], output_path: str):
    """绘制FFN激活值对比图"""
    layers = list(ffn_activations_data1[0].keys())
    mean_activations1 = {layer: torch.stack([data[layer].mean() for data in ffn_activations_data1]) for layer in layers}
    mean_activations2 = {layer: torch.stack([data[layer].mean() for data in ffn_activations_data2]) for layer in layers}
    
    plt.figure(figsize=(12, 6))
    for layer in layers:
        plt.plot(mean_activations1[layer].cpu().numpy(), label=f'Base Layer {layer}')
        plt.plot(mean_activations2[layer].cpu().numpy(), label=f'Chat Layer {layer}', linestyle='--')
    
    plt.xlabel('Generation Step')
    plt.ylabel('Average FFN Activation')
    plt.title('Average FFN Activation During Generation')
    plt.legend()
    plt.savefig(f'{output_path}.png')
    plt.close()

def plot_hidden_states(hidden_states_data1: List[Dict], hidden_states_data2: List[Dict], output_path: str):
    """绘制隐藏状态对比图"""
    layers = list(hidden_states_data1[0].keys())
    mean_hidden_states1 = {layer: torch.stack([data[layer].mean() for data in hidden_states_data1]) for layer in layers}
    mean_hidden_states2 = {layer: torch.stack([data[layer].mean() for data in hidden_states_data2]) for layer in layers}
    
    plt.figure(figsize=(12, 6))
    for layer in layers:
        plt.plot(mean_hidden_states1[layer].cpu().numpy(), label=f'Base Layer {layer}')
        plt.plot(mean_hidden_states2[layer].cpu().numpy(), label=f'Chat Layer {layer}', linestyle='--')
    
    plt.xlabel('Generation Step')
    plt.ylabel('Average Hidden State Value')
    plt.title('Average Hidden State Value During Generation')
    plt.legend()
    plt.savefig(f'{output_path}.png')
    plt.close()

def plot_generated_token_logits(tokens1: List[str], logits1: List[float], 
                              tokens2: List[str], logits2: List[float], output_path: str):
    """绘制生成token的logit值变化"""
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(logits1)), logits1, marker='o', label='Base')
    plt.plot(range(len(logits2)), logits2, marker='s', label='Chat')
    
    for i, (token, logit) in enumerate(zip(tokens1, logits1)):
        if token.strip() and not token.isspace():
            plt.annotate(token, (i, logit), textcoords="offset points", xytext=(0,10), ha='center')
    
    for i, (token, logit) in enumerate(zip(tokens2, logits2)):
        if token.strip() and not token.isspace():
            plt.annotate(token, (i, logit), textcoords="offset points", xytext=(0,-10), ha='center')
    
    plt.xlabel('Generation Step')
    plt.ylabel('Logit Value')
    plt.title('Generated Token Logit Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}.png')
    plt.close()

def plot_logits_entropy(logits1: List[np.ndarray], logits2: List[np.ndarray], output_path: str):
    """绘制logits分布的熵变化"""
    from scipy.stats import entropy
    
    entropies1 = [entropy(np.exp(logits)) for logits in logits1]
    entropies2 = [entropy(np.exp(logits)) for logits in logits2]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(entropies1)), entropies1, label='Base')
    plt.plot(range(len(entropies2)), entropies2, label='Chat')
    plt.xlabel('Generation Step')
    plt.ylabel('Entropy')
    plt.title('Logits Distribution Entropy')
    plt.legend()
    plt.savefig(f'{output_path}.png')
    plt.close() 