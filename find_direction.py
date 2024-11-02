import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cross_decomposition import PLSRegression

class DirectionFinder:
    def __init__(self, base_model_name: str, chat_model_name: str, device: str = "cuda"):
        """
        初始化模型和tokenizer
        
        Args:
            base_model_name: base模型的名称或路径
            chat_model_name: chat模型的名称或路径
            device: 运行设备
        """
        self.device = device
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device)
        self.chat_model = AutoModelForCausalLM.from_pretrained(chat_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
    def get_hidden_states(self, model: AutoModelForCausalLM, prompt: str, layer_idx: float = 0.3) -> torch.Tensor:
        """
        获取模型某一层的hidden states
        
        Args:
            model: 要获取hidden states的模型
            prompt: 输入文本
            layer_idx: 相对层深,范围[0,1]
        
        Returns:
            对应层的hidden states
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 获取总层数
        num_layers = len(model.transformer.h)
        # 将相对层深转换为实际层号
        layer = int(layer_idx * num_layers)
        
        # 获取到指定层的hidden states
        all_hidden_states = []
        def hook(module, input, output):
            all_hidden_states.append(output[0])
        
        # 注册hook
        handle = model.transformer.h[layer].register_forward_hook(hook)
        
        # 前向传播
        with torch.no_grad():
            model(**inputs)
            
        # 移除hook
        handle.remove()
        
        # 返回最后一个token的hidden states
        return all_hidden_states[0][:, -1, :]

    def collect_hidden(self, prompts: List[str], layer_idx: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集base模型和chat模型的hidden states
        
        Args:
            prompts: 输入文本列表
            layer_idx: 相对层深
            
        Returns:
            base模型和chat模型的hidden states数组
        """
        base_hidden = []
        chat_hidden = []
        
        for prompt in prompts:
            base_h = self.get_hidden_states(self.base_model, prompt, layer_idx)
            chat_h = self.get_hidden_states(self.chat_model, prompt, layer_idx)
            
            base_hidden.append(base_h.cpu().numpy())
            chat_hidden.append(chat_h.cpu().numpy())
        
        return np.array(base_hidden), np.array(chat_hidden)

    def find_directions(self, base_hidden: np.ndarray, chat_hidden: np.ndarray, n_components: int = 10) -> np.ndarray:
        """
        使用PLS找到从base到chat的转换方向
        
        Args:
            base_hidden: base模型的hidden states
            chat_hidden: chat模型的hidden states
            n_components: PLS组件数量
            
        Returns:
            转换方向矩阵
        """
        X = base_hidden.reshape(base_hidden.shape[0], -1)
        Y = chat_hidden.reshape(chat_hidden.shape[0], -1)
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(X, Y)
        
        return pls.x_weights_

    def patch_hidden(self, base_hidden: torch.Tensor, directions: np.ndarray, alpha: float) -> torch.Tensor:
        """
        沿指定方向修改hidden states
        
        Args:
            base_hidden: 要修改的hidden states
            directions: 修改方向
            alpha: 修改强度
            
        Returns:
            修改后的hidden states
        """
        directions_tensor = torch.from_numpy(directions).to(self.device)
        patched = base_hidden + alpha * directions_tensor
        return patched

    def evaluate_direction(self, prompt: str, directions: np.ndarray, alphas: List[float]) -> List[str]:
        """
        评估转换方向的效果
        
        Args:
            prompt: 输入文本
            directions: 转换方向
            alphas: 不同的修改强度
            
        Returns:
            不同修改强度下的模型输出
        """
        results = []
        base_hidden = self.get_hidden_states(self.base_model, prompt)
        
        for alpha in alphas:
            # patch hidden states
            patched_hidden = self.patch_hidden(base_hidden, directions, alpha)
            
            # 获取模型输出
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs_embeds=patched_hidden.unsqueeze(1),
                    max_new_tokens=50,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(text)
        
        return results

def main():
    # 使用示例
    finder = DirectionFinder(
        base_model_name="llama-2-7b-base",
        chat_model_name="llama-2-7b-chat"
    )
    
    # 准备一些测试prompt
    prompts = [
        "Tell me about yourself.",
        "What is the capital of France?",
        "How to make a cake?"
    ]
    
    # 收集hidden states
    base_hidden, chat_hidden = finder.collect_hidden(prompts)
    
    # 找到转换方向
    directions = finder.find_directions(base_hidden, chat_hidden)
    
    # 测试不同强度的效果
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    results = finder.evaluate_direction(prompts[0], directions, alphas)
    
    # 打印结果
    for alpha, result in zip(alphas, results):
        print(f"Alpha: {alpha}")
        print(f"Output: {result}\n")

if __name__ == "__main__":
    main()