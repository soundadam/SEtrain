#!/usr/bin/env python3
"""
Convert SoundAI datasets (48kHz FLAC) to 16kHz WAV format with volume normalization
Optimized for DNS3Dataset compatibility and relative path support
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback
import time
import argparse

# 音量归一化参数
TARGET_DBFS = -3.0  # 目标音量级别 (dBFS)
MAX_PEAK = 0.99      # 最大峰值避免削波

def normalize_volume(audio, target_dbfs=TARGET_DBFS, max_peak=MAX_PEAK):
    """对音频进行音量归一化处理"""
    rms = np.sqrt(np.mean(audio**2))
    current_dbfs = 20 * np.log10(rms + 1e-7)  # 避免log(0)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)
    normalized_audio = audio * gain
    
    # 检查峰值是否超过限制
    peak = np.max(np.abs(normalized_audio))
    if peak > max_peak:
        attenuation = max_peak / peak
        normalized_audio = normalized_audio * attenuation
    
    return normalized_audio

def convert_audio_file(args):
    """转换单个音频文件"""
    flac_path, wav_path, source_sr, target_sr, normalize = args
    
    try:
        # 创建目录
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        # 加载音频
        audio, sr = sf.read(flac_path, dtype='float32')
        
        # 验证采样率
        if sr != source_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=source_sr)
            sr = source_sr
        
        # 应用音量归一化
        if normalize:
            audio = normalize_volume(audio)
        
        # 重采样到目标采样率
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 保存为WAV
        sf.write(wav_path, audio, target_sr)
        
        return (True, flac_path, wav_path, None)
    except Exception as e:
        error_msg = f"Error converting {flac_path}: {str(e)}\n{traceback.format_exc()}"
        return (False, flac_path, wav_path, error_msg)

def find_matching_noisy_file(clean_flac_path, noisy_dir, noisy_suffix="_CH0", audio_extensions=['.flac']):
    """查找匹配的噪声文件"""
    filename = os.path.basename(clean_flac_path)
    
    # 尝试不同的文件名模式
    patterns_to_try = []
    for ext in audio_extensions:
        if filename.endswith(ext):
            base_name = filename[:-len(ext)]
            patterns_to_try.append(f"{base_name}{noisy_suffix}{ext}")
            patterns_to_try.append(filename)
    
    # 在噪声目录中搜索
    for pattern in patterns_to_try:
        noisy_path = os.path.join(noisy_dir, pattern)
        if os.path.exists(noisy_path):
            return noisy_path
    
    # 递归搜索子目录
    for root, dirs, files in os.walk(noisy_dir):
        for file in files:
            if file in patterns_to_try:
                return os.path.join(root, file)
    
    return None

def convert_soundai_datasets(
    source_root='/datasets',
    target_root=None,
    source_sr=48000,
    target_sr=16000,
    noisy_suffix="_CH0",
    num_workers=None,
    audio_extensions=['.flac'],
    normalize_volume=True
):
    """转换SoundAI数据集，适配DNS3Dataset"""
    # 设置默认目标路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if target_root is None:
        target_root = os.path.join(script_dir, 'soundai')
    
    os.makedirs(target_root, exist_ok=True)
    
    # 设置并行工作数
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Using {num_workers} parallel workers for conversion")
    if normalize_volume:
        print(f"Applying volume normalization to target level: {TARGET_DBFS} dBFS")
    
    # 定义转换映射
    conversions = [
        # test -> dev
        (
            os.path.join(source_root, 'test/dp_speech'),
            os.path.join(source_root, 'test/ma_noisy_speech'),
            os.path.join(target_root, 'dev_clean'),
            os.path.join(target_root, 'dev_noisy'),
            'train'
        ),
        # val -> train
        (
            os.path.join(source_root, 'val/dp_speech'),
            os.path.join(source_root, 'val/ma_noisy_speech'),
            os.path.join(target_root, 'train_clean'),
            os.path.join(target_root, 'train_noisy'),
            'dev'
        ),
    ]
    
    all_csvs = {}
    global_start_time = time.time()
    
    for source_clean_dir, source_noisy_dir, target_clean_dir, target_noisy_dir, dataset_name in conversions:
        print(f"\n{'='*70}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*70}")
        
        if not os.path.exists(source_clean_dir):
            print(f"Warning: Clean source directory not found, skipping: {source_clean_dir}")
            continue
        
        # 创建目标目录
        os.makedirs(target_clean_dir, exist_ok=True)
        os.makedirs(target_noisy_dir, exist_ok=True)
        
        # 查找所有干净音频文件
        clean_flac_files = []
        for root, dirs, files in os.walk(source_clean_dir):
            for file in files:
                if any(file.endswith(ext) for ext in audio_extensions):
                    clean_flac_files.append(os.path.join(root, file))
        
        print(f"Found {len(clean_flac_files)} clean audio files")
        
        if len(clean_flac_files) == 0:
            print(f"Warning: No clean audio files found in {source_clean_dir}")
            continue
        
        # 准备转换任务
        conversion_tasks = []
        file_mapping = []  # 用于CSV文件
        missing_noisy = []
        
        for index, clean_flac_path in enumerate(clean_flac_files):
            # # 生成干净WAV路径（保留原始文件名）
            # clean_filename = os.path.basename(clean_flac_path)
            # clean_wav_filename = os.path.splitext(clean_filename)[0] + '.wav'
            # clean_wav_path = os.path.join(target_clean_dir, clean_wav_filename)
            # 生成干净WAV路径（保留原始文件名）
            clean_filename = os.path.basename(clean_flac_path)
            clean_wav_filename = f"clean_{index}.wav" 
            clean_wav_path = os.path.join(target_clean_dir, clean_wav_filename) 
            # 查找匹配的噪声文件
            noisy_flac_path = find_matching_noisy_file(
                clean_flac_path, source_noisy_dir, noisy_suffix, audio_extensions
            )
            
            if noisy_flac_path and os.path.exists(noisy_flac_path):
                # # 生成噪声WAV路径（保留原始文件名）
                # noisy_filename = os.path.basename(noisy_flac_path)
                # noisy_wav_filename = os.path.splitext(noisy_filename)[0] + '.wav'
                # noisy_wav_path = os.path.join(target_noisy_dir, noisy_wav_filename)
                # 生成噪声WAV路径（仅仅保留index)
                noisy_filename = os.path.basename(noisy_flac_path)
                noisy_wav_filename = f"noisy_{index}.wav" 
                noisy_wav_path = os.path.join(target_noisy_dir, noisy_wav_filename)
                
                # 添加到转换任务
                conversion_tasks.append((clean_flac_path, clean_wav_path, source_sr, target_sr, normalize_volume))
                conversion_tasks.append((noisy_flac_path, noisy_wav_path, source_sr, target_sr, normalize_volume))
                
                # 记录映射关系（使用相对路径）
                file_mapping.append({
                    'noisy_path': os.path.relpath(noisy_wav_path, target_root),
                    'clean_path': os.path.relpath(clean_wav_path, target_root)
                })
            else:
                missing_noisy.append(clean_flac_path)
        
        # 处理缺失文件
        if missing_noisy:
            print(f"Warning: Could not find noisy counterparts for {len(missing_noisy)} clean files")
        
        # 并行转换文件
        print(f"Converting {len(conversion_tasks)} files for {dataset_name} dataset...")
        success_count = 0
        error_count = 0
        error_messages = []
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(convert_audio_file, conversion_tasks),
                total=len(conversion_tasks),
                desc=f"Converting {dataset_name}"
            ))
        
        # 处理结果
        for success, flac_path, wav_path, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                error_messages.append(error_msg)
        
        # 打印转换摘要
        print(f"\nConversion summary for {dataset_name}:")
        print(f"  Successfully converted: {success_count}/{len(conversion_tasks)} files")
        print(f"  Failed conversions: {error_count}")
        
        # 生成适配DNS3Dataset的CSV文件
        csv_path = os.path.join(target_root, f'{dataset_name}.scp')
        with open(csv_path, 'w') as f:
            for mapping in file_mapping:
                f.write(f"{mapping['noisy_path']} {mapping['clean_path']}\n")
        
        all_csvs[dataset_name] = csv_path
        print(f"Generated SCP file: {csv_path}")
        print(f"  Format: noisy_path clean_path")
        print(f"  Number of pairs: {len(file_mapping)}")
    
    # 最终摘要
    total_time = time.time() - global_start_time
    print(f"\n{'='*70}")
    print("Conversion Complete")
    print(f"{'='*70}")
    print(f"Total processing time: {total_time:.2f} seconds")
    
    for name, csv_path in all_csvs.items():
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"{name:20s}: {num_lines:6d} pairs -> {os.path.relpath(csv_path, script_dir)}")
    
    print(f"{'='*70}\n")
    
    return all_csvs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert SoundAI datasets for DNS3Dataset compatibility'
    )
    parser.add_argument(
        '--source_root',
        type=str,
        default='/datasets',
        help='Source root directory (default: /datasets)'
    )
    parser.add_argument(
        '--target_root',
        type=str,
        default=None,
        help='Target directory (default: prepare_datasets/soundai)'
    )
    parser.add_argument(
        '--source_sr',
        type=int,
        default=48000,
        help='Source sample rate (default: 48000)'
    )
    parser.add_argument(
        '--target_sr',
        type=int,
        default=16000,
        help='Target sample rate (default: 16000)'
    )
    parser.add_argument(
        '--noisy_suffix',
        type=str,
        default="_CH0",
        help='Suffix for noisy files (default: "_CH0")'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.flac'],
        help='Audio file extensions (default: .flac)'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Disable volume normalization'
    )
    parser.add_argument(
        '--target_dbfs',
        type=float,
        default=TARGET_DBFS,
        help=f'Target volume level (default: {TARGET_DBFS})'
    )
    parser.add_argument(
        '--max_peak',
        type=float,
        default=MAX_PEAK,
        help=f'Maximum peak amplitude (default: {MAX_PEAK})'
    )
    
    args = parser.parse_args()
    
    # 设置全局音量参数
    # global TARGET_DBFS, MAX_PEAK
    TARGET_DBFS = args.target_dbfs
    MAX_PEAK = args.max_peak
    
    convert_soundai_datasets(
        source_root=args.source_root,
        target_root=args.target_root,
        source_sr=args.source_sr,
        target_sr=args.target_sr,
        noisy_suffix=args.noisy_suffix,
        num_workers=args.workers,
        audio_extensions=args.extensions,
        normalize_volume=not args.no_normalize
    )