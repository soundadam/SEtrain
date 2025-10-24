# dns3 dataset, for training usage
curl -o -L soundai-splitted.7z https://box.nju.edu.cn/f/e720beaac10f4aa48f58/?dl=1
# 下载7z， 解压到 prepare_datasets/soundai-splitted这个位置

python nas_sweep.py
# watch -d  nvidia-smi 看看显存状况，nas_sweep.py可能会爆显存，这时候改动configs/cfg_train.yaml里的train_dataloader: batch_size: 24，validation 压力会小些
# 