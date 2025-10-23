# dns3 dataset, for training usage
curl -o -L https://box.nju.edu.cn/f/df26fa2dede8436da6ae/?dl=1
curl -o -L https://box.nju.edu.cn/f/5cca025fb9bc47daa550/?dl=1
# soundai datasets, for validation and test
python -u ./train.py \
    --config_file configs/cfg_train_onH200.yaml \
    --device 0,1 \
