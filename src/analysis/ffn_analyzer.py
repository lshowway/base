import torch
import numpy as np
import matplotlib.pyplot as plt
from ..models.llama_wrapper import LlamaWrapper

class FFNAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model_base = LlamaWrapper(config.base_model_path)
        self.model_chat = LlamaWrapper(config.chat_model_path)
    
    def plot_3d_ffn(self, ffn_activations, tokens, model_type, output_path):
        fig = plt.figure(figsize=(48,32))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.1)

        ax = fig.add_subplot(1,1,1, projection='3d')
        
        num_tokens = len(tokens)
        # 修复 ffn_activations[0] 可能是整数的问题
        if isinstance(ffn_activations[0], (int, float)):
            num_dims = 1
        else:
            num_dims = ffn_activations[0].shape[-1]
        
        xdata = np.array([np.linspace(0, num_tokens-1, num_tokens) for i in range(num_dims)])
        ydata = np.array([np.ones(num_tokens) * i for i in range(num_dims)])
        
        # 修复整数类型的 ffn_activations 处理
        if isinstance(ffn_activations[0], (int, float)):
            zdata = np.array(ffn_activations).reshape(-1, 1).T
        else:
            zdata = np.array([data.cpu().numpy().squeeze() for data in ffn_activations]).T
        
        ax.plot_wireframe(xdata, ydata, zdata, rstride=0, color="blue", linewidth=1.0)
        
        ax.set_xticks(np.linspace(0, num_tokens-1, num_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=5)
        
        ax.set_ylabel('FFN Dimensions')
        ax.set_zlabel('Activation Values')
        ax.set_title(f'{model_type} FFN Layer W2 Activations')
        
        ax.view_init(elev=20, azim=45)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
    
    def analyze(self):
        test_prompt = "Tell me a story about a brave knight."
        
        input_ids_base = self.model_base.tokenizer(test_prompt, return_tensors="pt").input_ids.to(self.model_base.device)
        input_ids_chat = self.model_chat.tokenizer(test_prompt, return_tensors="pt").input_ids.to(self.model_chat.device)
        
        ffn_base, _, _, _, tokens_base, _ = self.model_base.generate(input_ids_base)
        ffn_chat, _, _, _, tokens_chat, _ = self.model_chat.generate(input_ids_chat)
        
        self.plot_3d_ffn(ffn_base, tokens_base, "Base", 
                        f"{self.config.output_dir}/base_ffn_3d.png")
        self.plot_3d_ffn(ffn_chat, tokens_chat, "Chat",
                        f"{self.config.output_dir}/chat_ffn_3d.png") 