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
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            ffn_activations_list.append({layer: self.ffn_activations[layer][:, -1, :] for layer in self.ffn_activations})
            generated_tokens.append(self.tokenizer.decode(next_token))
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return ffn_activations_list, generated_tokens

def plot_3d_ax_llama2(ffn_activations_data, generated_tokens, savedir):
    fig = plt.figure(figsize=(48,32))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    ax = fig.add_subplot(1,1,1, projection='3d')

    # 获取FFN激活值的维度
    num_tokens = len(generated_tokens)
    num_dims = ffn_activations_data[0][0].shape[-1]  # 获取FFN层的维度大小

    # 创建网格数据
    xdata = np.array([np.linspace(0, num_tokens-1, num_tokens) for i in range(num_dims)])
    ydata = np.array([np.ones(num_tokens) * i for i in range(num_dims)])
    
    # 提取FFN层的激活值并转换为numpy数组
    zdata = np.array([data[0].cpu().numpy().squeeze() for data in ffn_activations_data]).T

    # 绘制3D线框图
    ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="blue", linewidth=1.0)

    # 设置坐标轴标签
    ax.set_xticks(np.linspace(0, num_tokens-1, num_tokens))
    ax.set_xticklabels(generated_tokens, rotation=45, ha='right', fontsize=5)
    
    ax.set_ylabel('FFN Dimensions', fontsize=12)
    ax.set_zlabel('Activation Values', fontsize=12)

    # 设置标题
    ax.set_title('Llama2 FFN Layer W2 Activations', fontsize=14, fontweight="bold", y=1.05)

    # 调整视角
    ax.view_init(elev=20, azim=45)

    # 保存图像
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(os.path.join(savedir, "llama2_ffn_w2_3d.png"), bbox_inches="tight", dpi=300)
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
    ffn_activations1, tokens1 = model1.generate(input_ids1)
    ffn_activations2, tokens2 = model2.generate(input_ids2)
    
    # 绘制3D图
    plot_3d_ax_llama2(ffn_activations1, tokens1, "/kaggle/working/llama2")
    plot_3d_ax_llama2(ffn_activations2, tokens2, "/kaggle/working/llama2-chat")
    
    # 清除显存
    del model1
    del model2
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()