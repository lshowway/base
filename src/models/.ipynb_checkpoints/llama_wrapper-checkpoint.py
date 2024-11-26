import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Tuple
import json
from tqdm import tqdm
from ..visualization.plotters import plot_ffn_activations, plot_hidden_states
from ..visualization.plotters import plot_generated_token_logits, plot_logits_entropy
import os

class LlamaWrapper:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto',
            torch_dtype=torch.float16
        )
        
        # 注册钩子
        self.ffn_activations = {}
        self.hidden_states = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册所有需要的钩子"""
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.act_fn.register_forward_hook(self._get_ffn_activation(i))
            layer.register_forward_hook(self._get_hidden_state(i))
    
    def _get_ffn_activation(self, layer_num):
        def hook(module, input, output):
            self.ffn_activations[layer_num] = output.detach()
        return hook

    def _get_hidden_state(self, layer_num):
        def hook(module, input, output):
            self.hidden_states[layer_num] = output[0].detach()
        return hook
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> Tuple[List[Any], ...]:
        """生成文本并收集激活值"""
        generated = []
        ffn_activations_list = []
        hidden_states_list = []
        logits_list = []
        generated_tokens = []
        generated_logits = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            ffn_activations_list.append({layer: self.ffn_activations[layer][:, -1, :] 
                                       for layer in self.ffn_activations})
            hidden_states_list.append({layer: self.hidden_states[layer][:, -1, :] 
                                     for layer in self.hidden_states})
            logits_list.append(logits.squeeze().cpu().numpy())
            generated_tokens.append(self.tokenizer.decode(next_token))
            generated_logits.append(logits.squeeze()[next_token].item())
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated, ffn_activations_list, hidden_states_list, logits_list, generated_tokens, generated_logits
    
    def compare_outputs(self, other_model: 'LlamaWrapper', config: 'Config'):
        """任务1: 比较两个模型的输出"""
        with open(config.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if config.max_samples:
            dataset = dataset[:config.max_samples]
            
        results = []
        for item in tqdm(dataset, desc="处理数据"):
            input_text = item['question']
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            other_input_ids = other_model.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # 获取两个模型的输出
            _, _, _, logits_base, tokens_base, _ = self.generate(input_ids)
            _, _, _, logits_chat, tokens_chat, _ = other_model.generate(other_input_ids)
            
            results.append({
                'input': input_text,
                'base_output': ''.join(tokens_base),
                'chat_output': ''.join(tokens_chat),
                'base_logits': logits_base,
                'chat_logits': logits_chat
            })
        
        # 保存结果
        output_path = os.path.join(config.output_dir, 'task1_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def analyze_internals(self, other_model: 'LlamaWrapper', config: 'Config'):
        """任务2: 分析模型内部状态"""
        with open(config.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if config.max_samples:
            dataset = dataset[:config.max_samples]
            
        for idx, item in enumerate(tqdm(dataset, desc="分析内部状态")):
            input_text = item['question']
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            other_input_ids = other_model.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # 获取两个模型的激活值
            _, ffn_base, hidden_base, _, _, _ = self.generate(input_ids)
            _, ffn_chat, hidden_chat, _, _, _ = other_model.generate(other_input_ids)
            
            # 绘制FFN激活值对比图
            plot_ffn_activations(
                ffn_base, ffn_chat,
                os.path.join(config.output_dir, f'ffn_activations_{idx}')
            )
            
            # 绘制隐藏状态对比图
            plot_hidden_states(
                hidden_base, hidden_chat,
                os.path.join(config.output_dir, f'hidden_states_{idx}')
            )
    
    def analyze_generation(self, other_model: 'LlamaWrapper', config: 'Config'):
        """任务3: 分析生成过程"""
        with open(config.dataset_path, 'r') as f:
            dataset = json.load(f)
        
        if config.max_samples:
            dataset = dataset[:config.max_samples]
            
        for idx, item in enumerate(tqdm(dataset, desc="分析生成过程")):
            input_text = item['question']
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            other_input_ids = other_model.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            # 获取生成过程的数据
            _, _, _, logits_base, tokens_base, logits_values_base = self.generate(input_ids)
            _, _, _, logits_chat, tokens_chat, logits_values_chat = other_model.generate(other_input_ids)
            
            # 绘制生成token的logit值变化
            plot_generated_token_logits(
                tokens_base, logits_values_base,
                tokens_chat, logits_values_chat,
                os.path.join(config.output_dir, f'token_logits_{idx}')
            )
            
            # 绘制logits分布的熵变化
            plot_logits_entropy(
                logits_base, logits_chat,
                os.path.join(config.output_dir, f'logits_entropy_{idx}')
            )