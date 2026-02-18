
  CUDA_VISIBLE_DEVICES=2 python3 autofigure_main.py \
  --method_file /app/examples/inputs/method.txt \
  --output_dir outputs/chart_demo_nosam \
  --provider local \
  --api_key YOUR_KEY \
  --rmbg_model_path  /root/models/RMBG-2.0 \
  --local_img_path /app/examples/inputs/test.png \
  --sam_checkpoint_path /root/models/sam3/sam3.pt \
  --sam_bpe_path /app/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --optimize_iterations 0 \
  --api_key "YOUR_API_KEY" \
  --task_type chart_code \
  --sam_prompt "axis,line,curve,bar,marker,legend,grid" \
  --enable_evaluation \
  --reference_code_path /app/examples/inputs/test.py
  
  
  CUDA_VISIBLE_DEVICES=2 python3 autofigure_main.py \
  --method_file /app/examples/inputs/method.txt \
  --output_dir outputs/chart_demo_sam \
  --provider local \
  --api_key YOUR_KEY \
  --rmbg_model_path  /root/models/RMBG-2.0 \
  --local_img_path /app/examples/inputs/test.png \
  --sam_checkpoint_path /root/models/sam3/sam3.pt \
  --sam_bpe_path /app/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
  --optimize_iterations 0 \
  --api_key "YOUR_API_KEY" \
  --task_type chart_code \
  --chart_use_sam \
  --sam_prompt "axis,line,curve,bar,marker,legend,grid" \
  --enable_evaluation \
  --reference_code_path /app/examples/inputs/test.py
  