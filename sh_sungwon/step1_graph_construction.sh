cd ../

# Seoul-Core
python -m scripts.generate_training_data --output_dir=data/Seoul-Core --traffic_df_filename=data/seoul-core.h5

# Seoul-Mix
python -m scripts.generate_training_data --output_dir=data/Seoul-Mix --traffic_df_filename=data/seoul-mix.h5