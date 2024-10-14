import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os

class LlamaWrapper:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Register hooks for all layers
        self.ffn_activations = {}
        self.hidden_states = {}
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.act_fn.register_forward_hook(self.get_ffn_activation(i))
            layer.register_forward_hook(self.get_hidden_state(i))
    
    def get_ffn_activation(self, layer_num):
        def hook(module, input, output):
            self.ffn_activations[layer_num] = output.detach()
        return hook

    def get_hidden_state(self, layer_num):
        def hook(module, input, output):
            self.hidden_states[layer_num] = output[0].detach()
        return hook
    
    def generate(self, input_ids, max_new_tokens=50):
        generated = []
        ffn_activations_list = []
        hidden_states_list = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            ffn_activations_list.append({layer: self.ffn_activations[layer][:, -1, :] for layer in self.ffn_activations})
            hidden_states_list.append({layer: self.hidden_states[layer][:, -1, :] for layer in self.hidden_states})
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated, ffn_activations_list, hidden_states_list
    
    def reset_all(self):
        self.ffn_activations = {}
        self.hidden_states = {}

def process_dataset(model, tokenizer, dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    ffn_activations_data = []
    hidden_states_data = []
    
    for item in tqdm(dataset, desc="Processing prompts"):
        input_ids = tokenizer(item['question'], return_tensors="pt").input_ids.to(model.device)
        
        model.reset_all()
        generated, ffn_activations_list, hidden_states_list = model.generate(input_ids)
        
        generated_text = tokenizer.decode(generated)
        if item['answer_matching_behavior'] in generated_text:
            ffn_activations_data.extend(ffn_activations_list)
            hidden_states_data.extend(hidden_states_list)
    
    return ffn_activations_data, hidden_states_data

def plot_ffn_activations(ffn_activations_data):
    layers = list(ffn_activations_data[0].keys())
    mean_activations = {layer: torch.stack([data[layer].mean() for data in ffn_activations_data]) for layer in layers}
    
    plt.figure(figsize=(12, 6))
    for layer in layers:
        plt.plot(mean_activations[layer].cpu().numpy(), label=f'Layer {layer}')
    
    plt.xlabel('Generation Step')
    plt.ylabel('Mean FFN Activation')
    plt.title('Mean FFN Activations Across Layers During Generation')
    plt.legend()
    plt.savefig('ffn_activations.png')
    plt.close()

def plot_hidden_states(hidden_states_data):
    layers = list(hidden_states_data[0].keys())
    mean_hidden_states = {layer: torch.stack([data[layer].mean() for data in hidden_states_data]) for layer in layers}
    
    plt.figure(figsize=(12, 6))
    for layer in layers:
        plt.plot(mean_hidden_states[layer].cpu().numpy(), label=f'Layer {layer}')
    
    plt.xlabel('Generation Step')
    plt.ylabel('Mean Hidden State Value')
    plt.title('Mean Hidden State Values Across Layers During Generation')
    plt.legend()
    plt.savefig('hidden_states.png')
    plt.close()

def main(model_name, dataset_path):
    model = LlamaWrapper(model_name)
    ffn_activations_data, hidden_states_data = process_dataset(model, model.tokenizer, dataset_path)
    
    plot_ffn_activations(ffn_activations_data)
    plot_hidden_states(hidden_states_data)
    
    return ffn_activations_data, hidden_states_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['meta-llama/Llama-2-7b', 'meta-llama/Llama-2-7b-chat'], required=True)
    parser.add_argument('--dataset', type=str, default='test_dataset_ab.json')
    args = parser.parse_args()

    ffn_activations, hidden_states = main(args.model, args.dataset)
    print(f"FFN Activations: {len(ffn_activations)} samples")
    print(f"Hidden States: {len(hidden_states)} samples")
