python inference_opt.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --output_folder videos/rolling_forcing_dmd_opt \
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
    --data_path prompts/example_prompts.txt \
    --num_output_frames 126 \
    --use_ema
