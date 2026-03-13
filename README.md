# AirGesture

开源的"隔空取物"手势应用 - PolyU DSAI5201 Spring Project

## 功能演示

通过简单的手势变化实现设备间的图片传输：

- **发送**：张开 → 握拳 → 截图并发送
- **接收**：握拳 → 张开 → 接收并打开图片

## 功能特性

- **手势识别**：基于 MediaPipe 的实时手势检测
- **状态变化触发**：通过手势状态变化（张开↔握拳）触发动作，避免误触发
- **设备发现**：自动发现局域网内的其他设备
- **文件传输**：点对点传输，支持进度显示
- **缩略图预览**：发送时显示截图预览和传输进度
- **调试模式**：可视化手势识别过程，方便调试

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.8+ |
| 手势识别 | MediaPipe Gesture Recognizer |
| GUI | tkinter (Python 内置) |
| 摄像头 | OpenCV |
| 设备发现 | zeroconf (mDNS/Bonjour) |
| 文件传输 | socket TCP |

## 安装

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/AirGesture.git
cd AirGesture

# 2. 创建虚拟环境 (推荐)
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载模型 (约 8MB)
python download_model.py
```

## 使用方法

### 正常模式

```bash
python -m src.main
```

### 调试模式

```bash
python -m src.main --debug
```

调试模式会显示：
- 摄像头画面和手部关键点
- 当前识别的手势和置信度
- 状态机状态
- 实时 FPS

## 手势操作

| 操作 | 手势变化 | 说明 |
|------|----------|------|
| 发送 | 张开 → 握拳 | 截图并启动发送进程 |
| 接收 | 握拳 → 张开 | 连接发送方并接收图片 |

**工作流程：**

```
手进入画面 → 进入手势识别模式 (RECOGNIZING)
  │
  ├─ 检测到 "张开 → 握拳" → 截图 → 启动发送进程 → 等待连接
  │                                      │
  │                                      └─ 接收方连接 → 传输图片 → 完成
  │
  └─ 检测到 "握拳 → 张开" → 启动接收进程 → 连接发送方 → 接收图片 → 打开
  │
手离开画面 → 退出识别模式 (IDLE)
```

## 调试快捷键

| 快捷键 | 功能 |
|--------|------|
| `q` | 退出程序 |
| `t` | 切换测试模式（不执行实际动作） |
| `s` | 保存当前帧截图 |
| `e` | 导出识别日志为 JSON |

## 项目结构

```
AirGesture/
├── src/
│   ├── main.py                 # 程序入口
│   ├── config.py               # 配置文件
│   ├── sender_process.py       # 发送进程 (独立)
│   ├── receiver_process.py     # 接收进程 (独立)
│   ├── gesture/
│   │   ├── recognizer.py       # MediaPipe 手势识别封装
│   │   └── debouncer.py        # 防抖处理
│   ├── core/
│   │   ├── state_machine.py    # 手势状态机
│   │   └── actions.py          # 动作执行 (截图、通知)
│   ├── network/
│   │   └── discovery.py        # mDNS 设备发现
│   ├── ui/
│   │   ├── thumbnail_window.py # 悬浮缩略图窗口
│   │   └── image_viewer.py     # 图片查看器
│   └── debug/
│       └── gesture_debugger.py # 调试可视化
├── models/
│   └── gesture_recognizer.task # MediaPipe 模型 (需下载)
├── download_model.py           # 模型下载脚本
├── requirements.txt            # Python 依赖
├── run.bat                     # Windows 快速启动
└── run_debug.bat               # Windows 调试启动
```

## 支持的手势

MediaPipe Gesture Recognizer 内置支持 7 种手势：

| 手势 | 英文名称 | 本项目使用 |
|------|----------|------------|
| 握拳 | Closed_Fist | ✅ 发送触发 |
| 张开手掌 | Open_Palm | ✅ 接收触发 |
| 指向上方 | Pointing_Up | 可扩展 |
| 拇指向下 | Thumb_Down | 可扩展 |
| 拇指向上 | Thumb_Up | 可扩展 |
| V字手势 | Victory | 可扩展 |
| 我爱你 | ILoveYou | 可扩展 |

## 架构设计

### 进程架构

```
┌─────────────────────────────────────────────────────┐
│                    主进程                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  摄像头采集  │→│  手势识别   │→│  状态机     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                            │        │
│                    触发动作时启动子进程 ↓            │
└─────────────────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ↓                               ↓
┌─────────────────────┐        ┌─────────────────────┐
│     发送进程         │        │     接收进程         │
│  - 显示缩略图        │        │  - 连接发送方        │
│  - 监听端口          │◄──────►│  - 接收文件          │
│  - 传输文件          │        │  - 打开图片          │
└─────────────────────┘        └─────────────────────┘
```

### 状态机

```
IDLE (空闲)
  │ 手进入画面
  ↓
RECOGNIZING (识别中)
  │ 检测到手势变化
  ├─ 张开 → 握拳 → 触发发送
  └─ 握拳 → 张开 → 触发接收
  │ 手离开画面
  ↓
IDLE (空闲)
```

## 依赖说明

```
opencv-python      # 摄像头和图像处理
mediapipe          # 手势识别
Pillow             # 图像处理
zeroconf           # 设备发现
pywin32            # Windows 系统通知 (仅 Windows)
```

## 注意事项

1. **模型文件**：首次运行前需要执行 `python download_model.py` 下载模型
2. **摄像头权限**：确保应用有摄像头访问权限
3. **防火墙**：如需跨设备传输，请允许 Python 通过防火墙 (端口 9527, 9528)
4. **单机测试**：可在同一台电脑上测试发送和接收流程

## 扩展建议

- 添加更多手势支持（如滑动切换图片）
- 集成剪贴板，支持文本传输
- 添加移动端应用
- 支持多文件批量传输

## License

MIT License
