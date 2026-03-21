#!/usr/bin/python

#*================================================================
#*   Copyright (C) 2026 XUranus All rights reserved.
#*
#*   File:         preprocess_videos.py
#*   Author:       XUranus
#*   Date:         2026-03-19
#*   Description:  视频预处理脚本 - 统一格式、重命名、数据增强
#*
#================================================================*/

"""
视频预处理脚本

功能：
1. 扫描 dataset 目录下的所有视频文件
2. 分类标签：上/下/抓/放 + 噪音 (swipe_up, swipe_down, grab, release, noise)
3. 按手势类别重命名文件
4. 数据增强：旋转、镜像、亮度调整等
5. 生成 labels.csv 文件

标签说明：
    - swipe_up: 上滑
    - swipe_down: 下滑
    - grab: 抓
    - release: 放
    - noise: 噪音（包括无手势、挥手、左滑、右滑等其他手势）

使用方法：
    # 预览模式（查看将要执行的操作）
    python preprocess_videos.py

    # 实际执行（不含增强）
    python preprocess_videos.py --execute

    # 执行并启用数据增强
    python preprocess_videos.py --execute --augment

    # 执行、转换格式并启用数据增强
    python preprocess_videos.py --execute --augment --convert --format avi
"""

import os
import re
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

# 支持的视频格式
VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.wmv'}

# 目标标签 (只保留这4个手势)
# - swipe_up: 上滑
# - swipe_down: 下滑
# - grab: 抓
# - release: 放
# 其他所有手势都归类为 noise (噪音)

TARGET_LABELS = {
    '上滑': 'swipe_up',
    '下滑': 'swipe_down',
    '抓握': 'grab',
    '抓': 'grab',
    '释放': 'release',
    '放': 'release',
    '松开': 'release',
    'swipe_up': 'swipe_up',
    'swipe_down': 'swipe_down',
    'grab': 'grab',
    'release': 'release',
}

# 归类为噪音的标签 (包括无手势、其他手势等)
NOISE_LABELS = {
    '无', '无手势', '噪声', 'noise', 'none',
    'wave', '挥手',
    '左滑', 'swipe_left',
    '右滑', 'swipe_right',
    '比心', 'finger_heart',
    'unknown',
}


class VideoAugmentor:
    """视频数据增强器"""

    def __init__(self, seed: int = 42):
        """初始化增强器"""
        random.seed(seed)
        np.random.seed(seed)

    def horizontal_flip(self, frame: np.ndarray) -> np.ndarray:
        """水平镜像翻转"""
        return cv2.flip(frame, 1)

    def rotate(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像

        Args:
            frame: 输入图像
            angle: 旋转角度（正值为逆时针）
        """
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            frame, matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return rotated

    def adjust_brightness(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """
        调整亮度

        Args:
            frame: 输入图像
            factor: 亮度因子 (>1 变亮, <1 变暗)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def adjust_contrast(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """
        调整对比度

        Args:
            frame: 输入图像
            factor: 对比度因子
        """
        mean = frame.mean()
        adjusted = np.clip((frame - mean) * factor + mean, 0, 255)
        return adjusted.astype(np.uint8)

    def add_gaussian_noise(self, frame: np.ndarray, sigma: float = 10) -> np.ndarray:
        """
        添加高斯噪声

        Args:
            frame: 输入图像
            sigma: 噪声标准差
        """
        noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def gaussian_blur(self, frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        高斯模糊

        Args:
            frame: 输入图像
            kernel_size: 核大小（奇数）
        """
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def zoom(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """
        缩放裁剪

        Args:
            frame: 输入图像
            scale: 缩放因子 (>1 放大)
        """
        h, w = frame.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        # 放大后取中心裁剪
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 裁剪中心区域
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2

        if scale > 1:
            cropped = resized[start_y:start_y + h, start_x:start_x + w]
        else:
            # 缩小后填充黑边
            cropped = np.zeros((h, w, 3), dtype=np.uint8)
            cropped[start_y:start_y + new_h, start_x:start_x + new_w] = resized

        return cropped

    def augment_frame(self, frame: np.ndarray, aug_type: str) -> np.ndarray:
        """
        对单帧应用增强

        Args:
            frame: 输入帧
            aug_type: 增强类型
        """
        if aug_type == 'flip':
            return self.horizontal_flip(frame)
        elif aug_type == 'rotate_small':
            return self.rotate(frame, random.uniform(-10, 10))
        elif aug_type == 'rotate_medium':
            return self.rotate(frame, random.uniform(-20, 20))
        elif aug_type == 'brightness_up':
            return self.adjust_brightness(frame, random.uniform(1.1, 1.3))
        elif aug_type == 'brightness_down':
            return self.adjust_brightness(frame, random.uniform(0.7, 0.9))
        elif aug_type == 'contrast_up':
            return self.adjust_contrast(frame, random.uniform(1.1, 1.3))
        elif aug_type == 'contrast_down':
            return self.adjust_contrast(frame, random.uniform(0.7, 0.9))
        elif aug_type == 'noise':
            return self.add_gaussian_noise(frame, random.uniform(5, 15))
        elif aug_type == 'blur':
            return self.gaussian_blur(frame, random.choice([3, 5]))
        elif aug_type == 'zoom_in':
            return self.zoom(frame, random.uniform(1.1, 1.2))
        elif aug_type == 'zoom_out':
            return self.zoom(frame, random.uniform(0.85, 0.95))
        elif aug_type == 'combo_flip_rotate':
            return self.rotate(self.horizontal_flip(frame), random.uniform(-10, 10))
        elif aug_type == 'combo_brightness_contrast':
            return self.adjust_contrast(
                self.adjust_brightness(frame, random.uniform(1.1, 1.2)),
                random.uniform(1.1, 1.2)
            )
        else:
            return frame


# 增强配置：每个原始视频生成的增强版本
AUGMENT_CONFIGS = [
    # 基础增强
    ('flip', '水平镜像'),
    ('rotate_small', '小角度旋转'),
    ('brightness_up', '亮度增加'),
    ('brightness_down', '亮度降低'),
    ('contrast_up', '对比度增加'),
    ('noise', '添加噪声'),

    # 组合增强
    ('combo_flip_rotate', '镜像+旋转'),
    ('combo_brightness_contrast', '亮度+对比度'),
]


def find_videos(root_dir: Path) -> List[Path]:
    """递归查找所有视频文件"""
    videos = set()
    for ext in VIDEO_EXTENSIONS:
        for f in root_dir.rglob(f'*{ext}'):
            videos.add(f)
        for f in root_dir.rglob(f'*{ext.upper()}'):
            videos.add(f)
    return sorted(videos)


def get_label_from_path(filepath: Path, root_dir: Path) -> str:
    """
    从路径提取标签

    返回:
        - swipe_up, swipe_down, grab, release: 目标手势
        - noise: 噪音（包括无手势和其他手势）
    """
    rel_path = filepath.relative_to(root_dir)
    parts = rel_path.parts

    if len(parts) >= 2:
        label_dir = parts[-2]
        label_lower = label_dir.lower()

        # 检查是否是目标手势
        if label_dir in TARGET_LABELS:
            return TARGET_LABELS[label_dir]
        if label_lower in TARGET_LABELS:
            return TARGET_LABELS[label_lower]

        # 其他所有情况都归类为噪音
        return 'noise'

    return 'noise'


def get_split_from_path(filepath: Path, root_dir: Path) -> str:
    """从路径提取数据集划分"""
    rel_path = filepath.relative_to(root_dir)
    parts = [p.lower() for p in rel_path.parts]

    if 'train' in parts:
        return 'Train'
    elif 'test' in parts:
        return 'Test'
    return 'Train'


def read_video(video_path: Path) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """
    读取视频所有帧

    Returns:
        frames: 帧列表
        fps: 帧率
        size: (width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0, (0, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps, (width, height)


def write_video(output_path: Path, frames: List[np.ndarray],
                fps: float, size: Tuple[int, int], target_format: str = 'avi') -> bool:
    """
    写入视频

    Args:
        output_path: 输出路径
        frames: 帧列表
        fps: 帧率
        size: (width, height)
        target_format: 目标格式
    """
    if not frames:
        return False

    # 根据实际输出文件扩展名选择编码器
    actual_ext = output_path.suffix.lower().lstrip('.')

    # 编码器映射 - 确保编码器与容器格式兼容
    fourcc_map = {
        'avi': 'XVID',   # AVI 容器使用 XVID
        'mp4': 'mp4v',   # MP4 容器使用 mp4v (MPEG-4)
    }

    # 优先使用文件实际扩展名，否则使用 target_format
    codec_key = actual_ext if actual_ext in fourcc_map else target_format
    fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(codec_key, 'XVID'))

    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not out.isOpened():
        return False

    for frame in frames:
        # 确保帧尺寸一致
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    return True


def augment_video(frames: List[np.ndarray], aug_type: str) -> List[np.ndarray]:
    """
    对视频所有帧应用增强

    Args:
        frames: 原始帧列表
        aug_type: 增强类型
    """
    augmentor = VideoAugmentor()
    augmented = []

    for frame in frames:
        aug_frame = augmentor.augment_frame(frame, aug_type)
        augmented.append(aug_frame)

    return augmented


def process_dataset(root_dir: Path, dry_run: bool = True,
                    convert_format: bool = False, target_format: str = 'avi',
                    enable_augment: bool = False, augment_count: int = 4):
    """
    处理整个数据集

    Args:
        root_dir: 数据集根目录
        dry_run: 只显示将要执行的操作
        convert_format: 是否转换视频格式
        target_format: 目标格式
        enable_augment: 是否启用数据增强
        augment_count: 每个视频生成的增强数量
    """
    videos = find_videos(root_dir)

    if not videos:
        print(f"[警告] 在 {root_dir} 下没有找到视频文件")
        return

    print(f"[信息] 找到 {len(videos)} 个视频文件")

    # 按划分和标签分组
    grouped = defaultdict(lambda: defaultdict(list))

    for video in videos:
        split = get_split_from_path(video, root_dir)
        label = get_label_from_path(video, root_dir)
        grouped[split][label].append(video)

    # 统计信息
    print("\n" + "=" * 60)
    print("[数据集统计]")
    print("=" * 60)
    total = 0
    for split in sorted(grouped.keys()):
        print(f"\n{split}/")
        split_total = 0
        for label in sorted(grouped[split].keys()):
            count = len(grouped[split][label])
            split_total += count
            print(f"  {label}: {count} 个")
        print(f"  ─────────────")
        print(f"  小计: {split_total} 个")
        total += split_total

    if enable_augment:
        print(f"\n  (增强后预计: {total * (1 + augment_count)} 个)")

    print(f"\n总计: {total} 个视频文件")

    # 创建输出目录结构
    output_base = root_dir / 'processed'
    output_dirs = {}

    # 最终标签列表
    final_labels = ['swipe_up', 'swipe_down', 'grab', 'release', 'noise']

    for split in grouped.keys():
        output_dirs[split] = output_base / split
        if not dry_run:
            output_dirs[split].mkdir(parents=True, exist_ok=True)
            for label in final_labels:
                (output_dirs[split] / label).mkdir(parents=True, exist_ok=True)

    # 生成处理计划
    print("\n" + "=" * 60)
    print("[处理计划]")
    print("=" * 60)

    process_plan = []
    augmentor = VideoAugmentor()

    for split in sorted(grouped.keys()):
        print(f"\n{split}/")
        for label in sorted(grouped[split].keys()):
            videos_list = sorted(grouped[split][label])
            print(f"  {label}/ ({len(videos_list)} 个)")

            for idx, video in enumerate(videos_list, start=1):
                ext = f'.{target_format}' if convert_format else video.suffix

                # 原始文件
                new_name = f"{label}_{idx:03d}{ext}"
                new_path = output_dirs[split] / label / new_name

                process_plan.append({
                    'source': video,
                    'output': new_path,
                    'split': split,
                    'label': label,
                    'augment_type': None,
                    'old_name': video.name,
                    'new_name': new_name,
                })
                print(f"    {video.name} -> {label}/{new_name}")

                # 增强版本
                if enable_augment:
                    # 随机选择增强类型
                    selected_augs = random.sample(AUGMENT_CONFIGS,
                                                  min(augment_count, len(AUGMENT_CONFIGS)))

                    for aug_idx, (aug_type, aug_desc) in enumerate(selected_augs, start=1):
                        aug_name = f"{label}_{idx:03d}_aug{aug_idx}{ext}"
                        aug_path = output_dirs[split] / label / aug_name

                        process_plan.append({
                            'source': video,
                            'output': aug_path,
                            'split': split,
                            'label': label,
                            'augment_type': aug_type,
                            'old_name': video.name,
                            'new_name': aug_name,
                        })
                        print(f"    {video.name} -> {label}/{aug_name} ({aug_desc})")

    if dry_run:
        print("\n" + "=" * 60)
        print("[预览模式] 以上是将要执行的操作")
        print("使用 --execute 参数实际执行")
        if enable_augment:
            print(f"数据增强已启用，每个视频生成 {augment_count} 个增强版本")
        print("=" * 60)
        return

    # 执行处理
    print("\n" + "=" * 60)
    print("[执行中...]")
    print("=" * 60)

    success_count = 0
    augment_count_actual = 0
    error_count = 0
    video_cache = {}  # 缓存已读取的视频帧

    for item in process_plan:
        source = item['source']
        output = item['output']
        aug_type = item['augment_type']

        try:
            # 读取视频（使用缓存）
            if str(source) not in video_cache:
                frames, fps, size = read_video(source)
                if not frames:
                    raise ValueError("无法读取视频帧")
                video_cache[str(source)] = (frames, fps, size)
            else:
                frames, fps, size = video_cache[str(source)]

            # 应用增强
            if aug_type:
                frames = augment_video(frames, aug_type)
                augment_count_actual += 1

            # 写入输出
            if write_video(output, frames, fps, size, target_format):
                success_count += 1
                status = f"[增强: {aug_type}]" if aug_type else ""
                print(f"  ✓ {item['old_name']} -> {item['new_name']} {status}")
            else:
                raise ValueError("写入视频失败")

        except Exception as e:
            error_count += 1
            print(f"  ✗ {item['old_name']}: {e}")

    # 更新 labels.csv
    update_labels_csv(output_base, process_plan)

    # 清理缓存
    video_cache.clear()

    print("\n" + "=" * 60)
    print(f"[完成]")
    print(f"  成功: {success_count} (原始: {success_count - augment_count_actual}, 增强: {augment_count_actual})")
    print(f"  失败: {error_count}")
    print(f"  输出目录: {output_base}")
    print("=" * 60)


def update_labels_csv(output_base: Path, process_plan: List[Dict]):
    """生成 labels.csv 文件"""
    csv_path = output_base / 'labels.csv'

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('split,filename,label,augmented\n')
        for split in ['Train', 'Test']:
            items = [x for x in process_plan if x['split'] == split]
            for item in sorted(items, key=lambda x: x['new_name']):
                augmented = 'yes' if item['augment_type'] else 'no'
                f.write(f"{split},{item['new_name']},{item['label']},{augmented}\n")

    print(f"\n[信息] 已生成 {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='视频数据集预处理脚本 - 支持数据增强',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 预览模式（默认）
    python preprocess_videos.py

    # 执行处理（不含增强）
    python preprocess_videos.py --execute

    # 执行处理并启用数据增强（每个视频生成4个增强版本）
    python preprocess_videos.py --execute --augment

    # 执行处理、启用增强并指定增强数量
    python preprocess_videos.py --execute --augment --aug-count 6

    # 执行处理、转换格式并启用增强
    python preprocess_videos.py --execute --augment --convert --format avi

保留的标签:
    swipe_up (上滑), swipe_down (下滑), grab (抓), release (放), noise (噪音/其他)

数据增强类型:
    - flip: 水平镜像
    - rotate_small: 小角度旋转
    - brightness_up/down: 亮度调整
    - contrast_up/down: 对比度调整
    - noise: 高斯噪声
    - combo_flip_rotate: 镜像+旋转组合
        """
    )
    parser.add_argument(
        '--execute', '-e',
        action='store_true',
        help='实际执行操作（默认只预览）'
    )
    parser.add_argument(
        '--convert', '-c',
        action='store_true',
        help='转换视频格式为统一格式'
    )
    parser.add_argument(
        '--format', '-f',
        default='avi',
        choices=['avi', 'mp4'],
        help='目标视频格式（默认avi）'
    )
    parser.add_argument(
        '--augment', '-a',
        action='store_true',
        help='启用数据增强'
    )
    parser.add_argument(
        '--aug-count',
        type=int,
        default=4,
        help='每个视频生成的增强数量（默认4）'
    )

    args = parser.parse_args()

    root_dir = Path(__file__).parent

    print("\n" + "=" * 60)
    print("视频数据集预处理")
    print("=" * 60)
    print(f"数据集目录: {root_dir}")
    print(f"目标格式: {args.format}")
    print(f"转换格式: {'是' if args.convert else '否'}")
    print(f"数据增强: {'是 (每视频{}个)'.format(args.aug_count) if args.augment else '否'}")
    print(f"执行模式: {'实际执行' if args.execute else '预览模式'}")
    print(f"保留标签: swipe_up, swipe_down, grab, release, noise")
    print("=" * 60)

    process_dataset(
        root_dir,
        dry_run=not args.execute,
        convert_format=args.convert,
        target_format=args.format,
        enable_augment=args.augment,
        augment_count=args.aug_count
    )


if __name__ == '__main__':
    main()
