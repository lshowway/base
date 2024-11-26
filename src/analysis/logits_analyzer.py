import torch
import numpy as np
import matplotlib.pyplot as plt
from ..models.llama_wrapper import LlamaWrapper

class LogitsAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model_base = LlamaWrapper(config.base_model_path)
        self.model_chat = LlamaWrapper(config.chat_model_path)
    
    def plot_topk_coverage(self, logits_list, model_type, output_path):
        k_values = range(1, 11)
        weights = range(1, 51)
        
        plt.figure(figsize=(12, 8))
        
        for k in k_values:
            coverages = []
            for w in weights:
                total_coverage = 0
                for logits in logits_list:
                    # Convert numpy array to tensor if needed
                    if isinstance(logits, np.ndarray):
                        logits = torch.from_numpy(logits)
                    weighted_logits = logits * w
                    topk_indices = torch.topk(weighted_logits, k).indices
                    total_probs = torch.softmax(weighted_logits, dim=-1)
                    # 修复维度问题 - 移除多余的索引
                    coverage = total_probs[topk_indices].sum().item()
                    total_coverage += coverage
                coverages.append(total_coverage / len(logits_list))
            
            plt.plot(weights, coverages, label=f'k={k}', marker='o', markersize=3)
        
        plt.xlabel('Weight')
        plt.ylabel('Coverage')
        plt.title(f'Top-K Coverage vs Weight for {model_type}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
    
    def analyze(self):
        test_prompt = "Tell me a story about a brave knight."
        
        input_ids_base = self.model_base.tokenizer(test_prompt, return_tensors="pt").input_ids.to(self.model_base.device)
        input_ids_chat = self.model_chat.tokenizer(test_prompt, return_tensors="pt").input_ids.to(self.model_chat.device)
        
        _, _, _, logits_base, _, _ = self.model_base.generate(input_ids_base)
        _, _, _, logits_chat, _, _ = self.model_chat.generate(input_ids_chat)
        
        self.plot_topk_coverage(logits_base, "Base",
                              f"{self.config.output_dir}/base_topk_coverage.png")
        self.plot_topk_coverage(logits_chat, "Chat", 
                              f"{self.config.output_dir}/chat_topk_coverage.png") 