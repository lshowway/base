import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from torch.autograd import grad
import seaborn as sns

class LipschitzAnalyzer:
    def __init__(
        self,
        base_model,  # 基础模型
        instructed_model,  # 指令微调后的模型
        tokenizer
    ):
        self.base_model = base_model
        self.instructed_model = instructed_model
        self.tokenizer = tokenizer
        
    def compute_local_lipschitz(
        self,
        model,
        input_text: str,
        eps: float = 1e-6
    ) -> float:
        """
        计算模型在给定输入处的局部Lipschitz常数
        按论文公式：Lip_f(x) = ||∇_x f_θ(x)||_2
        """
        # 对输入文本进行编码
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 确保需要计算梯度
        inputs['input_ids'].requires_grad_(True)
        
        # 获取模型输出（最后一个token的logits）
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 获取最可能的输出的logit
        confident_logit = torch.max(logits, dim=-1)[0]
        
        # 计算梯度
        gradient = grad(
            confident_logit.sum(),
            inputs['input_ids'],
            create_graph=True
        )[0]
        
        # 计算梯度的L2范数（Lipschitz常数）
        lipschitz = torch.norm(gradient.float(), p=2).item()
        
        return lipschitz

    def analyze_lipschitz_distribution(
        self,
        samples: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """分析一组样本的Lipschitz常数分布"""
        base_lips = []
        inst_lips = []
        
        for text in samples:
            # 计算基础模型的Lipschitz常数
            base_lip = self.compute_local_lipschitz(self.base_model, text)
            base_lips.append(base_lip)
            
            # 计算指令微调模型的Lipschitz常数
            inst_lip = self.compute_local_lipschitz(self.instructed_model, text)
            inst_lips.append(inst_lip)
            
        return np.array(base_lips), np.array(inst_lips)

    def visualize_lipschitz_distribution(
        self,
        samples: List[str],
        title: str = "Lipschitz Constants Distribution"
    ):
        """可视化Lipschitz常数分布"""
        base_lips, inst_lips = self.analyze_lipschitz_distribution(samples)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制直方图
        plt.hist(
            base_lips,
            bins=30,
            alpha=0.5,
            color='blue',
            label='Base Model'
        )
        plt.hist(
            inst_lips,
            bins=30,
            alpha=0.5,
            color='green',
            label='Instruction-tuned Model'
        )
        
        plt.xlabel('Local Lipschitzness')
        plt.ylabel('Number of Samples')
        plt.title(title)
        plt.legend()
        
        # 计算一些统计信息
        stats = {
            'base_mean': np.mean(base_lips),
            'base_std': np.std(base_lips),
            'inst_mean': np.mean(inst_lips),
            'inst_std': np.std(inst_lips),
            'relative_change': (np.mean(inst_lips) - np.mean(base_lips)) / np.mean(base_lips)
        }
        
        # 在图上添加统计信息
        info_text = f"""
        Base Model: mean={stats['base_mean']:.2f}, std={stats['base_std']:.2f}
        Inst Model: mean={stats['inst_mean']:.2f}, std={stats['inst_std']:.2f}
        Relative Change: {stats['relative_change']*100:.1f}%
        """
        plt.text(0.02, 0.98, info_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return plt.gcf(), stats

def main():
    # 示例使用
    analyzer = LipschitzAnalyzer(
        base_model=your_base_model,
        instructed_model=your_instructed_model,
        tokenizer=your_tokenizer
    )
    
    # 准备测试样本
    test_samples = [
        "What is machine learning?",
        "Explain quantum computing",
        "Write a poem about spring",
        # 添加更多测试样本
    ]
    
    # 计算并可视化Lipschitz常数分布
    fig, stats = analyzer.visualize_lipschitz_distribution(test_samples)
    
    # 保存结果
    fig.savefig('lipschitz_distribution.png')
    print("Analysis Stats:", stats)

if __name__ == "__main__":
    main()
