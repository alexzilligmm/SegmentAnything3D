# Segment Anything 3D

okay you to first:
use the sam3d script to produce the predictions ( uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/sam2 --save_2dmask_path outputs/sam2 --sam_checkpoint_path checkpoints/sam2.1_hiera_tiny.pt --config_file configs/sam2.1/sam2.1_hiera_t.yaml)
format using this format uv command (uv run format --results_path outputs/sam1/scene0000_00.pth --gt_path data/scans/scene0000_00_vh_clean_2.labels.ply --output_folder test) ?? chekc this 
then format the gt: gt_format this produces .pth files (uv run gt_format --data_folder data/scans)
this the .txt files prepare_data (uv run prepare_data --scans_dir data/scans --output_dir gts)
eval_instance runs the evaluation ( uv run eval_instance --pred_path predictions --gt_path gts )