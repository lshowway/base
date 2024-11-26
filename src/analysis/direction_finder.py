import torch
import numpy as np
from typing import List, Tuple
from sklearn.cross_decomposition import PLSRegression
from ..models.llama_wrapper import LlamaWrapper
import os
import json
from tqdm import tqdm

class DirectionFinder:
    def __init__(self, config):
        """初始化DirectionFinder
        
        Args:
            config: 配置对象,包含模型路径等信息
        """
        self.config = config
        self.model_base = LlamaWrapper(config.base_model_path)
        self.model_chat = LlamaWrapper(config.chat_model_path)
        
    def get_layer_hidden_states(self, model: LlamaWrapper, prompt: str, layer_idx: float = 0.3) -> torch.Tensor:
        """获取指定层的hidden states
        
        Args:
            model: LlamaWrapper模型实例
            prompt: 输入文本
            layer_idx: 相对层深,范围[0,1]
            
        Returns:
            指定层的hidden states
        """
        inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 获取总层数并计算目标层号
        num_layers = len(model.model.model.layers)
        layer = int(layer_idx * num_layers)
        
        # 使用hook获取hidden states
        hidden_states = []
        def hook(module, input, output):
            hidden_states.append(output[0])
        
        handle = model.model.model.layers[layer].register_forward_hook(hook)
        
        with torch.no_grad():
            model.model(**inputs)
            
        handle.remove()
        
        # 返回最后一个token的hidden states
        return hidden_states[0][:, -1, :]

    def collect_hidden_states(self, prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """收集base和chat模型的hidden states
        
        Args:
            prompts: 输入文本列表
            
        Returns:
            base模型和chat模型的hidden states数组
        """
        base_hidden = []
        chat_hidden = []
        
        for prompt in tqdm(prompts, desc="收集hidden states"):
            base_h = self.get_layer_hidden_states(self.model_base, prompt, self.config.layer_idx)
            chat_h = self.get_layer_hidden_states(self.model_chat, prompt, self.config.layer_idx)
            
            base_hidden.append(base_h.cpu().numpy())
            chat_hidden.append(chat_h.cpu().numpy())
        
        return np.array(base_hidden), np.array(chat_hidden)

    def find_direction(self, base_hidden: np.ndarray, chat_hidden: np.ndarray) -> np.ndarray:
        """使用PLS找到从base到chat的转换方向
        
        Args:
            base_hidden: base模型的hidden states
            chat_hidden: chat模型的hidden states
            
        Returns:
            转换方向矩阵
        """
        if base_hidden.shape != chat_hidden.shape:
            raise ValueError("base_hidden和chat_hidden的维度必须相同")
            
        X = base_hidden.reshape(base_hidden.shape[0], -1)
        Y = chat_hidden.reshape(chat_hidden.shape[0], -1)
        
        pls = PLSRegression(n_components=min(self.config.n_components, X.shape[1]))
        pls.fit(X, Y)
        
        return pls.coef_

    def apply_direction(self, hidden: torch.Tensor, direction: np.ndarray, alpha: float) -> torch.Tensor:
        """沿指定方向修改hidden states
        
        Args:
            hidden: 原始hidden states
            direction: 修改方向
            alpha: 修改强度
            
        Returns:
            修改后的hidden states
        """
        if hidden.shape[-1] != direction.shape[0]:
            raise ValueError("hidden和direction的维度不匹配")
            
        direction_tensor = torch.from_numpy(direction).to(hidden.device).to(hidden.dtype)
        return hidden + alpha * (direction_tensor @ hidden.T).T

    def evaluate_direction(self, prompt: str, direction: np.ndarray, alphas: List[float]) -> dict:
        """评估转换方向的效果
        
        Args:
            prompt: 输入文本
            direction: 转换方向
            alphas: 不同的修改强度
            
        Returns:
            包含评估结果的字典
        """
        base_hidden = self.get_layer_hidden_states(self.model_base, prompt)
        chat_hidden = self.get_layer_hidden_states(self.model_chat, prompt)
        
        results = {
            'prompt': prompt,
            'base_outputs': [],
            'chat_outputs': []
        }
        
        for alpha in alphas:
            # 修改hidden states
            patched_base = self.apply_direction(base_hidden, direction, alpha)
            patched_chat = self.apply_direction(chat_hidden, direction, alpha)
            
            # 生成输出
            with torch.no_grad():
                base_output = self.model_base.model.generate(
                    inputs_embeds=patched_base.unsqueeze(1),
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.model_base.tokenizer.eos_token_id,
                    do_sample=False
                )
                chat_output = self.model_chat.model.generate(
                    inputs_embeds=patched_chat.unsqueeze(1),
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.model_chat.tokenizer.eos_token_id,
                    do_sample=False
                )
            
            results['base_outputs'].append({
                'alpha': alpha,
                'text': self.model_base.tokenizer.decode(base_output[0], skip_special_tokens=True)
            })
            results['chat_outputs'].append({
                'alpha': alpha,
                'text': self.model_chat.tokenizer.decode(chat_output[0], skip_special_tokens=True)
            })
        
        return results

    def analyze(self):
        """执行完整的方向分析流程"""
        # 准备测试prompt
        test_prompts = [
            "Tell me about yourself.",
            "What is the capital of France?",
            "How to make a cake?",
            "Explain quantum computing.",
            "Write a short story."
        ]
        
        # 收集hidden states
        base_hidden, chat_hidden = self.collect_hidden_states(test_prompts)
        
        # 找到转换方向
        direction = self.find_direction(base_hidden, chat_hidden)
        
        # 设置测试的alpha值
        alphas = [-1.0, -0.5, 0.0, 0.5, 1.0] if self.config.alphas is None else self.config.alphas
        
        # 评估方向效果
        results = []
        for prompt in test_prompts:
            result = self.evaluate_direction(prompt, direction, alphas)
            results.append(result)
        
        # 保存结果
        output_path = os.path.join(self.config.output_dir, 'direction_analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False) 