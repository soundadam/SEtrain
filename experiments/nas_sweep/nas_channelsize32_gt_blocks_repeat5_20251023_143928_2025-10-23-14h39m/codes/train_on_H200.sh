# dns3 dataset, for training usage
curl -o -L https://box.nju.edu.cn/f/df26fa2dede8436da6ae/?dl=1
# 下载7z， 解压到 prepare_datasets/soundai-splitted这个位置
python -u ./train.py \
    --config_file configs/cfg_train_onH200.yaml \
    --device 0,1 \
