import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LlamaWrapper:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        
        # 注册所有层的钩子
        self.ffn_activations = {}
        self.logits_list = []
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.act_fn.register_forward_hook(self.get_ffn_activation(i))
    
    def get_ffn_activation(self, layer_num):
        def hook(module, input, output):
            self.ffn_activations[layer_num] = output.detach()
        return hook
    
    def generate(self, input_ids, max_new_tokens=50):
        generated = []
        ffn_activations_list = []
        generated_tokens = []
        logits_list = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            ffn_activations_list.append({layer: self.ffn_activations[layer][:, -1, :] for layer in self.ffn_activations})
            generated_tokens.append(self.tokenizer.decode(next_token))
            logits_list.append(logits)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return ffn_activations_list, generated_tokens, logits_list

def plot_topk_coverage(logits_list, savedir, model_type):
    k_values = range(1, 11)
    weights = range(1, 51)
    
    plt.figure(figsize=(12, 8))
    
    for k in k_values:
        coverages = []
        for w in weights:
            total_coverage = 0
            for logits in logits_list:
                # 对logits进行权重缩放
                weighted_logits = logits * w
                # 获取top-k的索引
                topk_indices = torch.topk(weighted_logits, k).indices
                # 计算top-k覆盖率
                total_probs = torch.softmax(weighted_logits, dim=-1)
                coverage = total_probs[0][topk_indices].sum().item()
                total_coverage += coverage
            coverages.append(total_coverage / len(logits_list))
        
        plt.plot(weights, coverages, label=f'k={k}', marker='o', markersize=3)
    
    plt.xlabel('Weight')
    plt.ylabel('Coverage')
    plt.title(f'Top-K Coverage vs Weight for {model_type}')
    plt.legend()
    plt.grid(True)
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, f"{model_type}_topk_coverage.png"), bbox_inches="tight", dpi=300)
    plt.close()

def main():
    # 初始化两个模型
    model_name1 = "/kaggle/input/llama-2/pytorch/7b-hf/1"  # llama2 base模型
    model_name2 = "/kaggle/input/llama-2/pytorch/7b-chat-hf/1"  # llama2-chat模型
    
    model1 = LlamaWrapper(model_name1)
    model2 = LlamaWrapper(model_name2)
    
    # 测试输入
    test_prompt = "Tell me a story about a brave knight."
    
    # 为两个模型生成输入
    input_ids1 = model1.tokenizer(test_prompt, return_tensors="pt").input_ids.to(model1.device)
    input_ids2 = model2.tokenizer(test_prompt, return_tensors="pt").input_ids.to(model2.device)
    
    # 生成输出
    _, _, logits_list1 = model1.generate(input_ids1)
    _, _, logits_list2 = model2.generate(input_ids2)
    
    # 绘制top-k覆盖率图
    plot_topk_coverage(logits_list1, "/kaggle/working", "llama2_base")
    plot_topk_coverage(logits_list2, "/kaggle/working", "llama2_chat")

if __name__ == "__main__":
    main()