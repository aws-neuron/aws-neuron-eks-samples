python inference_neuron.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
    --embedding_path prompt_embeds.pt \
    --output_path output_latent.pt \
    --num_output_frames 21 \
    --use_ema
