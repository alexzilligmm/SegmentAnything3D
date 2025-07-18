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


### Efficient hiera t after _C extension fix
uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/sam2_hiera_t_test_C --num_of_scenes 5 --generator_config_file configs/sam2_efficient_hiera_t.yaml


## Table

uv run sam3d --rgb_path data/scannetv2_images --data_path preprocessed-data --save_path outputs/sam2_hiera_t_test_C --num_of_scenes 5 --generator_config_file configs/sam2_efficient_hiera_t.yaml

modified intersection to IoU, cluster algorithm to a more coarse one.

restore intersection, performances crashed.

backup best configs:
cluster alg:

#_target_: cuml.cluster.HDBSCAN
name: sam3d
hdbscan:
  _target_: hdbscan.hdbscan_.HDBSCAN
  min_cluster_size: 40 # best:75 best cropping: 40
  min_samples: 5 # best:5
  cluster_selection_epsilon: 0.0
  max_cluster_size: 4096
  metric: 'precomputed'
  alpha: 1.0
  cluster_selection_method: 'eom'
  allow_single_cluster: false 

generator:
defaults:
    - prompt_mode: efficient
    - crop_mode: uniform
    
_target_: automask.SAM2EfficientAutomaticMaskGenerator

model: ??
number_of_points: 10 # number of points to use to prompt sam, if grid mode is selected √ is computed and the output rounded, if efficient is ignored
points_per_batch: 64 # inner batch size of the mask generator
pred_iou_thresh: 0.0 # threshold over model's predicted mask quality  
stability_score_thresh: 0.7 # filter over the IoU of the masks 
stability_score_offset: 0.7
mask_threshold: 0.0 # threshold for mask logits
post_processing_method: mm # nms, bm, mm or none
post_processing_thresh: 0.8 # box IoU cutoff used by nms
use_psr: false # use post process small regions 
crop_n_layers: 0 # number of layers of crops
crop_overlap_ratio: 0.33 
crop_n_points_downscale_factor: 2 
point_grids: null # pre-computed points grid if needed
min_mask_region_area: 25.0 # removes holes in masks with area smaller than 
output_mode: binary_mask 
use_m2m: false
cut_edges_masks: true 
multimask_output: true 

boxes IoU metric.