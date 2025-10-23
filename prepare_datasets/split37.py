import os
import random
from pathlib import Path
import shutil

# 设置随机种子以确保可重复性
random.seed(42)

# 获取当前脚本所在目录
dataset_path = Path(__file__).parent

# 定义原始目录
source_dir = dataset_path / 'soundai-val-test-wav'
test_clean_dir = source_dir / 'test_clean'
test_noisy_dir = source_dir / 'test_noisy'
val_clean_dir = source_dir / 'val_clean'
val_noisy_dir = source_dir / 'val_noisy'

# 创建新的划分目录
splitted_dir = dataset_path / 'soundai-splitted'
train_clean_dir = splitted_dir / 'train_clean'
train_noisy_dir = splitted_dir / 'train_noisy'
dev_clean = splitted_dir / 'dev_clean'
dev_noisy = splitted_dir / 'dev_noisy'

# 确保目录存在
for d in [train_clean_dir, train_noisy_dir, dev_clean, dev_noisy]:
    d.mkdir(parents=True, exist_ok=True)

# 收集所有干净的音频文件路径
clean_files = []
clean_files.extend(test_clean_dir.glob('*.wav'))
clean_files.extend(val_clean_dir.glob('*.wav'))

# 创建文件对列表 (clean_path, noisy_path)
file_pairs = []
for clean_path in clean_files:
    # 确定对应的噪声文件路径
    if clean_path.parent.name == 'test_clean':
        noisy_path = test_noisy_dir / f"{clean_path.stem.replace('clean','noisy')}.wav"
    else:  # val_clean
        noisy_path = val_noisy_dir / f"{clean_path.stem.replace('clean','noisy')}.wav"

    # 只有当噪声文件存在时才添加到对列表中
    if noisy_path.exists():
        file_pairs.append((clean_path, noisy_path))
    else:
        print(f"警告: 找不到对应的噪声文件 {noisy_path}")

# 随机打乱文件对
random.shuffle(file_pairs)

# 计算划分点 (73% 训练, 27% 验证)
split_idx = int(len(file_pairs) * 0.73)
train_pairs = file_pairs[:split_idx]
dev_pairs = file_pairs[split_idx:]

print(f"总文件对数量: {len(file_pairs)}")
print(f"训练集数量: {len(train_pairs)}")
print(f"验证集数量: {len(dev_pairs)}")

# 创建.scp文件路径
train_scp_path = splitted_dir / 'train.scp'
dev_scp_path = splitted_dir / 'dev.scp'

# 打开.scp文件准备写入
with open(train_scp_path, 'w') as train_scp, open(dev_scp_path, 'w') as dev_scp:
    # 复制训练集文件并写入.scp
    for clean_path, noisy_path in train_pairs:
        # 复制文件
        shutil.copy(clean_path, train_clean_dir / clean_path.name)
        shutil.copy(noisy_path, train_noisy_dir / noisy_path.name)
        
        # 写入.scp文件
        train_scp.write(f"train_noisy/{noisy_path.name} train_clean/{clean_path.name}\n")
    
    # 复制验证集文件并写入.scp
    for clean_path, noisy_path in dev_pairs:
        # 复制文件 - 修正这里的目标目录
        shutil.copy(clean_path, dev_clean / clean_path.name)  # 改为 dev_clean
        shutil.copy(noisy_path, dev_noisy / noisy_path.name)  # 改为 dev_noisy
        
        # 写入.scp文件
        dev_scp.write(f"dev_noisy/{noisy_path.name} dev_clean/{clean_path.name}\n")

print("数据集划分完成!")
print(f"训练集.scp文件已创建: {train_scp_path}")
print(f"验证集.scp文件已创建: {dev_scp_path}")