cd ../

# Seoul-Core
# python dcrnn_train_pytorch.py --config_filename=data/model/pretrained/Seoul-Core/config.yaml
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/seoul-core.h5
# Seoul-Mix
# python dcrnn_train_pytorch.py --config_filename=data/model/pretrained/Seoul-Mix/config.yaml
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/seoul-mix.h5
