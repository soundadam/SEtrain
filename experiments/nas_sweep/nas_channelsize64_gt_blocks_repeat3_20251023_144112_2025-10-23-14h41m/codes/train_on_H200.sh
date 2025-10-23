# dns3 dataset, for training usage
curl -o -L https://box.nju.edu.cn/f/e720beaac10f4aa48f58/?dl=1
# 下载7z， 解压到 prepare_datasets/soundai-splitted这个位置

python nas_sweep.py
# watch -d  nvidia-smi 看看显存状况，nas_sweep.py会自动选择合适的batch size进行训练