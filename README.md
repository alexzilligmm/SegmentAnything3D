# Segment Anything 3D

okay you to first:
- preprocess 2d data (python scannet-preprocess/preprocess_scannet.py --dataset_root data --output_root preprocessed-data)
- prepare rgbd images (python scannet-preprocess/prepare_2d_data/prepare_2d_data.py --scannet_path data/scans --output_path data/scannetv2_images)

- use the sam3d script to produce the predictions ( uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/sam2 --sam_checkpoint_path checkpoints/sam2.1_hiera_tiny.pt --config_file configs/sam2.1/sam2.1_hiera_t.yaml)
- format using this format uv command (uv run format --results_path outputs/${results_path}$ --gt_path data/scans --output_folder outputs/${output_folder}$) ?? chekc this 


- then format the gt: gt_format this produces .pth files (uv run gt_format --data_folder data/scans)
- this the .txt files prepare_data (uv run prepare_data --scans_dir data/scans --output_dir gts)

- eval_instance runs the evaluation ( uv run eval_instance --pred_path outputs/${output_subfolder}$  --gt_path gts )


## Other commands

uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/sam2_hiera_b+_test --num_of_scenes 5 --generator_config_file configs/sam2_hiera_b+.yaml