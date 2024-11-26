import argparse
import os
from src.utils.config import Config
from src.models.llama_wrapper import LlamaWrapper
from src.analysis.direction_finder import DirectionFinder
from src.analysis.ffn_analyzer import FFNAnalyzer
from src.analysis.logits_analyzer import LogitsAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='LLaMA分析工具')
    
    # 必需参数
    parser.add_argument('--task', type=str, required=True,
                      choices=['task1','task2','task3','task4_1','task4_2','task4_3'],
                      help='要执行的任务')
    parser.add_argument('--base_model', type=str, required=True,
                      help='base模型路径')
    parser.add_argument('--chat_model', type=str, required=True,
                      help='chat模型路径')
    
    # 可选参数
    parser.add_argument('--dataset', type=str, default='datasets/test.json',
                      help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='最大样本数')
    parser.add_argument('--device', type=str, default='cuda',
                      help='运行设备')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 创建配置
    config = Config(
        base_model_path=args.base_model,
        chat_model_path=args.chat_model,
        device=args.device,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        task=args.task,
        max_samples=args.max_samples
    )
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    if config.task == 'task4_1':
        analyzer = FFNAnalyzer(config)
        analyzer.analyze()
        
    elif config.task == 'task4_2':
        analyzer = LogitsAnalyzer(config)
        analyzer.analyze()
        
    elif config.task == 'task4_3':
        finder = DirectionFinder(config)
        finder.analyze()
        
    elif config.task in ['task1', 'task2', 'task3']:
        model_base = LlamaWrapper(config.base_model_path)
        model_chat = LlamaWrapper(config.chat_model_path)
        
        if config.task == 'task1':
            model_base.compare_outputs(model_chat, config)
        elif config.task == 'task2':
            model_base.analyze_internals(model_chat, config)
        else:  # task3
            model_base.analyze_generation(model_chat, config)

if __name__ == '__main__':
    main() 