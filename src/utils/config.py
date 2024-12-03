from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Config:
    # 必需参数（无默认值）
    base_model_path: str
    chat_model_path: str
    dataset_path: str
    output_dir: str
    task: str  # task1, task2, task3, task4_1, task4_2, task4_3
    
    # 可选参数（有默认值）
    device: str = "cuda"
    max_samples: Optional[int] = None
    max_new_tokens: int = 50
    layer_idx: float = 0.3  # 用于direction finder
    n_components: int = 10  # PLS组件数量
    alphas: Optional[List[float]] = None  # direction patch的强度列表
    
    # 数据格式配置
    question_key: str = "question"  # 输入文本的键名
    answer_key: str = "answer_matching_behavior"  # 期望输出的键名
    
    # 中文相关配置
    use_chinese: bool = True  # 是否使用中文
    chinese_prompt_template: str = "问题：{}\n回答："  # 中文提示模板
    
    # Lipschitz分析相关配置
    num_layers: int = 32  # 分析的层数
    num_perturbations: int = 100  # 每层扰动次数
    perturbation_epsilon: float = 1e-2  # 扰动大小