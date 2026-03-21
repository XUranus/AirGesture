# AirGesture

基于深度学习的空中手势识别系统，通过摄像头捕获手势控制电脑。

## 支持的手势

| 手势 | 动作 |
|------|------|
| **Grab** (张开→握拳) | 截图并广播到网络 |
| **Release** (握拳→张开) | 接收网络截图 |
| **上滑** | 页面向下滚动 |
| **下滑** | 页面向上滚动 |

## 项目结构

```
AirGesture-xuranus/
├── dataset/                    # 数据集
│   └── preprocess_videos.py    # 视频预处理脚本
├── modelTraining/              # 模型训练
│   ├── Gesture_Recognition_TCN.ipynb  # 训练 Notebook
│   ├── inference.py            # 实时测试
│   └── checkpoints/            # 模型文件
└── desktop/                    # 桌面应用
    ├── main.py                 # 主程序入口
    ├── gesture_detector.py     # 手势检测
    └── tcn_classifier.py       # TCN 分类器
```

---

## 一、数据预处理

### 1. 整理视频文件

```
dataset/
├── swipe_up/      # 上滑视频
├── swipe_down/    # 下滑视频
├── grab/          # 抓取视频
├── release/       # 释放视频
└── noise/         # 其他/无效手势
```

### 2. 运行预处理

```bash
cd dataset

# 预览模式
python preprocess_videos.py

# 执行预处理（含数据增强）
python preprocess_videos.py --execute --augment
```

输出到 `dataset/processed/Train/` 和 `dataset/processed/Test/`

---

## 二、模型训练

### 1. 打开 Notebook

```bash
cd modelTraining
jupyter notebook Gesture_Recognition_TCN.ipynb
```

### 2. 运行所有单元格

Notebook 会自动完成：
- 加载数据集、提取手部关键点
- 训练 TCN 模型（5 类分类）
- 导出 ONNX 模型

### 3. 输出文件

```
checkpoints/
├── config.json           # 配置文件
├── gesture_tcn_best.pth  # 最佳模型
└── gesture_tcn.onnx      # ONNX 模型
```

### 4. 测试模型

```bash
# 摄像头测试
python inference.py

# 视频测试
python inference.py --video ../dataset/processed/Test/swipe_up/swipe_up_001.mp4

# 调整参数
python inference.py --threshold 0.4
```

**控制键：** `q`=退出 `r`=重置 `l`=切换骨架显示 `p`=暂停

---

## 三、桌面应用

### 1. 配置参数

编辑 `desktop/config.py`：

```python
USE_TCN_CLASSIFIER = True    # True=TCN模型, False=规则判断
TCN_THRESHOLD = 0.5          # 置信度阈值（越低越容易触发）
TCN_WINDOW_SECONDS = 1.0     # 时间窗口
CAMERA_INDEX = 0             # 摄像头索引
```

### 2. 运行程序

```bash
cd desktop
python main.py
```

### 3. 功能说明

| 功能 | 说明 |
|------|------|
| 摄像头预览 | 实时显示手部骨架 |
| TCN 识别 | 深度学习手势分类 |
| 截图广播 | Grab 手势截图并广播 |
| 截图接收 | Release 手势接收截图 |
| 页面滚动 | 上滑/下滑手势控制滚动 |

---

## 快速开始

```bash
# 安装依赖
pip install torch opencv-python mediapipe numpy scikit-learn tqdm onnxruntime

# 数据预处理
cd dataset && python preprocess_videos.py --execute --augment

# 模型训练
cd ../modelTraining && jupyter notebook Gesture_Recognition_TCN.ipynb

# 运行应用
cd ../desktop && python main.py
```

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| 找不到摄像头 | 修改 `CAMERA_INDEX` 为 1 或 2 |
| 模型加载失败 | 确认 `checkpoints/config.json` 存在 |
| 识别率低 | 降低 `TCN_THRESHOLD` 到 0.3 |
| 手势无反应 | 保持手在画面中 1 秒以上 |

## 许可证

MIT License
