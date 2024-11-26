# LLaMA 模型分析工具

本项目提供了一套工具来分析和比较LLaMA base和chat模型的行为差异。主要关注四个方面的分析：模型输出比较、内部状态分析、生成过程分析和深入机制分析。

## 项目结构

```
.
├── main.py                 # 主程序入口
├── src/
│   ├── models/            # 模型相关代码
│   │   ├── __init__.py
│   │   └── llama_wrapper.py  # LLaMA模型包装器
│   ├── analysis/          # 分析工具
│   │   ├── __init__.py
│   │   ├── direction_finder.py  # 模型方向分析
│   │   ├── ffn_analyzer.py      # FFN层分析
│   │   └── logits_analyzer.py   # logits分析
│   ├── visualization/     # 可视化工具
│   │   ├── __init__.py
│   │   └── plotters.py    # 绘图函数
│   └── utils/            # 工具函数
│       ├── __init__.py
│       └── config.py     # 配置类
└── README.md
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行分析

```bash
# 基础分析
python main.py --task task1 \
    --base_model /path/to/llama2-base \
    --chat_model /path/to/llama2-chat \
    --dataset data/test.json \
    --output_dir outputs/task1

# 深入分析
python main.py --task task4_3 \
    --base_model /path/to/llama2-base \
    --chat_model /path/to/llama2-chat \
    --output_dir outputs/task4_3 \
    --layer_idx 0.3 \
    --n_components 10
```

### 配置参数
详见 `src/utils/config.py` 中的 `Config` 类定义。

## 数据要求

### 输入数据格式
```json
[
  {
    "question": "输入文本",
    "answer_matching_behavior": "期望的输出行为"
  }
]
```

### 输出结果
- 任务1：`task1_results.json`
- 任务2：`ffn_activations_{idx}.png`, `hidden_states_{idx}.png`
- 任务3：`token_logits_{idx}.png`, `logits_entropy_{idx}.png`
- 任务4：各类分析图表和`direction_analysis.json`

## 注意事项

1. 内存使用
- 建议使用GPU进行分析
- 注意控制batch size和样本数量
- 及时清理中间结果

2. 运行时间
- 完整分析可能耗时较长
- 支持断点续传
- 可通过max_samples参数控制规模

3. 结果解释
- 注意区分相关性和因果关系
- 考虑模型规模带来的影响
- 结合具体任务场景解读结果

## 核心任务

### 任务1：模型输出比较
- 目标：对比base和chat模型的输出差异
- 实现：`LlamaWrapper.compare_outputs()`
- 分析内容：
  - 输出文本内容对比
  - 生成token的logits分布
  - 指令跟随能力评估

### 任务2：内部状态分析
- 目标：分析模型内部激活状态
- 实现：`LlamaWrapper.analyze_internals()`
- 关注点：
  - FFN模块的激活值
  - Transformer块的输出
  - 不同层的行为差异

### 任务3：生成过程分析
- 目标：研究token生成过程
- 实现：`LlamaWrapper.analyze_generation()`
- 分析维度：
  - token生成的logits分布变化
  - 每步生成的token及其logit值
  - logits分布的熵变化

### 任务4：深入机制分析
包含三个子任务：

#### 4.1 FFN层W2可视化
- 实现：`FFNAnalyzer`
- 功能：3D可视化FFN层的W2矩阵
- 分析：观察权重分布特征

#### 4.2 FFN层权重影响
- 实现：`LogitsAnalyzer`
- 功能：分析W2权重对logits分布的影响
- 指标：top-k coverage随权重变化

#### 4.3 模型转换方向
- 实现：`DirectionFinder`
- 目标：寻找base到chat的转换方向
- 方法：使用PLS回归分析hidden states

## 实现细节

### 数据收集
- 使用hook机制收集内部状态
- 支持批量数据处理
- 自动保存中间结果

### 可视化功能
- FFN激活值对比图
- 隐藏状态变化图
- token logits变化图
- logits熵变化图
- FFN层3D可视化
- top-k coverage图

### 结果分析
- 自动生成JSON格式结果
- 支持多维度对比分析
- 可配置的分析参数

## 使用方法

### 安装依赖