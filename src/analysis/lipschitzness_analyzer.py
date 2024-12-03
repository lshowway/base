import torch
import numpy as np
import matplotlib.pyplot as plt
from ..models.llama_wrapper import LlamaWrapper
from typing import List, Tuple
import os
import json
from tqdm import tqdm

class LipschitzAnalyzer:
    def __init__(self, config):
        """初始化Lipschitz分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.model_base = LlamaWrapper(config.base_model_path)
        self.model_chat = LlamaWrapper(config.chat_model_path)
        
    def compute_unembedding_lipschitz(self, 
                                    model: LlamaWrapper,
                                    hidden_states: torch.Tensor,
                                    num_perturbations: int = 100,
                                    epsilon: float = 1e-2) -> List[float]:
        """计算unembedding层的Lipschitz常数
        
        Args:
            model: 模型实例
            hidden_states: 输入隐藏状态
            num_perturbations: 扰动次数
            epsilon: 扰动大小
            
        Returns:
            Lipschitz常数列表
        """
        lipschitz_constants = []
        device = hidden_states.device
        
        # 获取原始logits输出
        with torch.no_grad():
            original_logits = model.model.lm_head(hidden_states)
        
        # 添加随机扰动并计算Lipschitz常数
        for _ in range(num_perturbations):
            # 生成随机扰动
            perturbation = torch.randn_like(hidden_states).to(device) * epsilon
            perturbed_hidden = hidden_states + perturbation
            
            # 计算扰动后的logits
            with torch.no_grad():
                perturbed_logits = model.model.lm_head(perturbed_hidden)
            
            # 计算输入和输出的距离
            input_dist = torch.norm(perturbation).to(device)
            output_dist = torch.norm(perturbed_logits - original_logits).to(device)
            
            # 计算Lipschitz常数
            if input_dist > 0:
                lipschitz_constant = output_dist / input_dist
                lipschitz_constants.append(lipschitz_constant.item())
        
        return lipschitz_constants
    
    def get_hidden_states(self, 
                         model: LlamaWrapper, 
                         input_ids: torch.Tensor) -> torch.Tensor:
        """获取最后一层的隐藏状态
        
        Args:
            model: 模型实例
            input_ids: 输入token ids
            
        Returns:
            最后一层的隐藏状态
        """
        with torch.no_grad():
            outputs = model.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        return hidden_states
    
    def analyze_unembedding_lipschitz(self, 
                                    prompts: List[str]) -> Tuple[List[float], List[float]]:
        """分析unembedding层的Lipschitz特性
        
        Args:
            prompts: 输入提示列表
            
        Returns:
            (base_constants, chat_constants)元组
        """
        base_constants = []
        chat_constants = []
        
        for prompt in tqdm(prompts, desc="分析Lipschitz常数"):
            # 准备输入
            input_ids = self.model_base.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model_base.device)
            
            # 获取隐藏状态
            base_hidden = self.get_hidden_states(self.model_base, input_ids)
            chat_hidden = self.get_hidden_states(self.model_chat, input_ids)
            
            # 计算Lipschitz常数
            base_lip = self.compute_unembedding_lipschitz(
                self.model_base, 
                base_hidden,
                self.config.num_perturbations,
                self.config.perturbation_epsilon
            )
            chat_lip = self.compute_unembedding_lipschitz(
                self.model_chat,
                chat_hidden,
                self.config.num_perturbations,
                self.config.perturbation_epsilon
            )
            
            base_constants.extend(base_lip)
            chat_constants.extend(chat_lip)
        
        return base_constants, chat_constants
    
    def plot_lipschitz_distribution(self,
                                  base_constants: List[float],
                                  chat_constants: List[float],
                                  output_path: str):
        """绘制Lipschitz常数分布对比图
        
        Args:
            base_constants: base模型的Lipschitz常数
            chat_constants: chat模型的Lipschitz常数
            output_path: 输出路径
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制核密度估计
        plt.hist(base_constants, bins=50, density=True, alpha=0.5, label='Base Model')
        plt.hist(chat_constants, bins=50, density=True, alpha=0.5, label='Chat Model')
        
        plt.xlabel('Lipschitz Constant')
        plt.ylabel('Density')
        plt.title('Distribution of Unembedding Layer Lipschitz Constants')
        plt.legend()
        plt.grid(True)
        
        # 添加统计信息
        base_stats = f'Base Mean: {np.mean(base_constants):.2f}\nBase Std: {np.std(base_constants):.2f}'
        chat_stats = f'Chat Mean: {np.mean(chat_constants):.2f}\nChat Std: {np.std(chat_constants):.2f}'
        plt.text(0.02, 0.98, base_stats, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(0.98, 0.98, chat_stats, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def analyze(self):
        """执行完整的Lipschitz分析"""
        # 测试提示
        test_prompts = [
            "Tell me about artificial intelligence.",
            "What is the capital of France?",
            "Explain quantum computing.",
            "How does photosynthesis work?",
            "What is the theory of relativity?"
        ]
        
        # 分析unembedding层的Lipschitz常数
        base_constants, chat_constants = self.analyze_unembedding_lipschitz(test_prompts)
        
        # 绘制分布图
        output_path = os.path.join(self.config.output_dir, 'unembedding_lipschitz_distribution.png')
        self.plot_lipschitz_distribution(base_constants, chat_constants, output_path)
        
        # 保存统计结果
        results = {
            'base_model': {
                'mean': float(np.mean(base_constants)),
                'std': float(np.std(base_constants)),
                'min': float(np.min(base_constants)),
                'max': float(np.max(base_constants)),
                'median': float(np.median(base_constants))
            },
            'chat_model': {
                'mean': float(np.mean(chat_constants)),
                'std': float(np.std(chat_constants)),
                'min': float(np.min(chat_constants)),
                'max': float(np.max(chat_constants)),
                'median': float(np.median(chat_constants))
            }
        }
        
        results_path = os.path.join(self.config.output_dir, 'unembedding_lipschitz_stats.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)