# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import math
import torch

from hydra.utils import instantiate

from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from automask.logger import PRSLogger
from automask.util.boxes import batched_bm, batched_mm
from automask.util.efficient_prompting import (
    build_clusters,
    build_layer_efficient_prompt,
    get_centres,
    get_cluster_alg,
    patches_labels_to_masks,
)
from automask.amg import (
    build_all_layer_random_cloud,
    generate_crop_boxes,
    generate_crop_efficient,
    MaskData,
    merge_crop_boxes,
)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.amg import (
    build_all_layer_point_grids,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    coco_encode_rle,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
    
    
)

class SAM2EfficientAutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
    def __init__(
        self,
        crop_mode,
        prompt_mode,
        model: SAM2Base,
        number_of_points: int = 1,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        post_processing_method: str = "nms",
        post_processing_thresh: float = 0.7,
        use_psr: bool = False,
        crop_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        use_m2m: bool = False,
        cut_edges_masks: bool = True,
        multimask_output: bool = True,
        **kwargs,
    ) -> None:
        """
        Using a SAM 2 model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM 2 with a HieraL backbone.

        Arguments:
          model (Sam): The SAM 2 model to use for mask prediction.
          logger (PRSLogger): The logger used in the efficient mode to store middle layer computation results
          number_of_points (int or None): total number of points to prompt sam with
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          mask_threshold (float): Threshold for binarizing the mask logits
          post_processing_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
          use_m2m (bool): Whether to add a one step refinement using previous mask predictions.
          multimask_output (bool): Whether to output multimask at each point of the grid.
        """
        prompt_mode_name = prompt_mode.name
        crop_mode_name = crop_mode.name

        assert prompt_mode_name in [
            "grid",
            "random",
            "efficient",
        ], f"prompt_mode '{prompt_mode_name}' not supported."
        
        if prompt_mode.cluster_alg.name == "no_crops" and crop_mode_name == "efficient":
            raise ValueError(
                "You are using 'no_crops' cluster algorithm with 'efficient' crop mode, "
            )

        if prompt_mode_name == "grid":
            if point_grids is not None:
                self.point_grids = point_grids  # Use provided grids directly
            else:
                assert number_of_points is not None, (
                    "When using prompt_mode = 'grid', you must provide "
                    "number_of_points"
                )
                assert number_of_points > 0, "number_of_points must be positive."

                points_per_side = int(math.sqrt(number_of_points))

                self.point_grids = build_all_layer_point_grids(
                    points_per_side,
                    crop_n_layers,
                    crop_n_points_downscale_factor,
                )
        elif prompt_mode_name == "random":
            assert number_of_points is not None, (
                "When using prompt_mode = 'random', you must provide "
                "number_of_points"
            )
            self.point_grids = build_all_layer_random_cloud(
                number_of_points,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif prompt_mode_name == "efficient":
            assert (
                prompt_mode.logger is not None
            ), "When using prompt_mode = 'efficient', logger must be provided."

            self.cluster_alg = instantiate(prompt_mode.cluster_alg.hdbscan)
            self.centres_selection = prompt_mode.centres_selection
            self.metric = prompt_mode.metric
            self.points_per_cluster = prompt_mode.points_per_cluster
            self.adapt_to_crop_size = prompt_mode.adapt_to_crop_size

            self.logger: PRSLogger = instantiate(prompt_mode.logger, model=model)
            self.logger.attach_hook()

        assert crop_mode_name in [
            "boxes",
            "efficient",
        ], f"crop_mode '{crop_mode_name}' not supported."

        if crop_mode_name == "efficient":
            assert (
                crop_mode.logger is not None
            ), "When using crop_mode = 'efficient', logger must be provided."

            self.crop_cluster_alg = instantiate(crop_mode.cluster_alg)
            self.crop_metric = crop_mode.metric
            self.crop_merging_threshold = crop_mode.crop_merging_threshold

            self.crop_logger: PRSLogger = instantiate(crop_mode.logger, model=model)
            self.crop_logger.attach_hook()

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            try:
                from pycocotools import mask as mask_utils  # type: ignore  # noqa: F401
            except ImportError as e:
                print("Please install pycocotools")
                raise e

        self.predictor = SAM2ImagePredictor(
            model,
            max_hole_area=min_mask_region_area,
            max_sprinkle_area=min_mask_region_area,
        )
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.use_psr = use_psr
        self.post_processing_method = post_processing_method
        self.post_processing_thresh = post_processing_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        self.use_m2m = use_m2m
        self.cut_edges_masks = cut_edges_masks
        self.multimask_output = multimask_output
        self.prompt_mode_name = prompt_mode_name
        self.crop_mode_name = crop_mode_name

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2AutomaticMaskGenerator":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2AutomaticMaskGenerator): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Args:
            image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
            Tuple[
            List[Dict[str, Any]], List of mask records, each a dict with:
                segmentation (Union[dict, np.ndarray]): The mask. If output_mode='binary_mask', an HW array. Otherwise, a dictionary containing the RLE.
                bbox (List[float]): The bounding box around the mask, in XYWH format.
                area (int): The area in pixels of the mask.
                predicted_iou (float): The model's prediction of the mask's quality.
                point_coords (List[List[float]]): The point coordinates used to generate this mask.
                stability_score (float): A measure of the mask's quality.
                crop_box (List[float]): The crop of the image used to generate the mask, in XYWH format.
            List[np.ndarray],      List of point coordinates used to generate the masks.
            List[List[int]],       List of crop boxes used.
            ]
        """

        # Generate masks
        mask_data, points, crop_boxes = self._generate_masks(image)
        

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        # TODO: make a function that using the masks and the scores? classify them
        # TODO: what if we have multiple crops?
        # TODO: shall we always do the classification for each crop or after the crop wise post processing?
        # TODO: we are now using the first crop (the whole image) not only to generate some masks but also to crop the image sometimes
        # TODO: shall we keep these embeddings in this case (we cannot revert the order as we need the big iage to go first to compute crops)
        # TODO: or maybe we are doing ti twice right now, if so is enough to put it as last image for next time!!!
        # TODO: shall we make another extra pass?
        for idx in range(len(mask_data["segmentations"])):
            # TODO: or crop the image again and make a classification pipeline here base on the masks?
            # TODO: or simply produce new embeddings for the whole image?
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns, points, crop_boxes

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]

        crop_boxes, layer_idxs = self.get_crops(
            image,
            orig_size,
            self.crop_n_layers,
            self.crop_overlap_ratio,
        )
        # Iterate over image crops
        data = MaskData()
        prompted_points = []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data, points = self._process_crop(
                image, crop_box, layer_idx, orig_size
            )
            
            data.cat(crop_data)
            prompted_points.append(points)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1 and self.post_processing_method == "nms":
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.post_processing_thresh,
            )
            data.filter(keep_by_nms)
        elif len(crop_boxes) > 1 and self.post_processing_method == "mm":
            masks_to_merge = batched_mm(
                data["masks"].bool(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.post_processing_thresh,
            )
            data.merge(masks_to_merge)

        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        data.to_numpy()
        return data, prompted_points, crop_boxes

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points based on the selected prompt method
        # Get points for this crop
        inputs_for_image = self.get_input(orig_size, cropped_im_size, crop_layer_idx)
        
        if inputs_for_image[0] is None and inputs_for_image[1] is None:
            return None, None

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, inputs_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_predictor()

        # Remove duplicates within this crop.
        if self.post_processing_method == "nms":
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.post_processing_thresh,
            )
            data.filter(keep_by_nms)
        # Merges duplicates within this crop.
        elif self.post_processing_method == "mm":
            masks_to_merge = batched_mm(
                data["masks"].bool(),
                data["iou_preds"],
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.post_processing_thresh,
            )
            data.merge(masks_to_merge)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["boxes"]))])

        return data, inputs_for_image

    def _process_batch(
        self,
        inputs: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        input_points, in_masks = inputs

        points = torch.as_tensor(
            input_points, dtype=torch.float32, device=self.predictor.device
        )

        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )

        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )

        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :] if not in_masks.any() else None,
            in_labels[:, None] if not in_masks.any() is None else None,
            boxes=None,
            mask_input=in_masks[:, None, :] if in_masks.any() else None,
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=points.repeat_interleave(masks.shape[1], dim=0),
            low_res_masks=low_res_masks.flatten(0, 1),
        )

        del masks

        if self.use_psr:
            data = self.postprocess_small_regions(
                data, self.min_mask_region_area, self.post_processing_thresh
            )

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            # One step refinement using previous mask predictions
            in_points = self.predictor._transforms.transform_coords(
                data["points"], normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious, _ = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        if self.cut_edges_masks:
            keep_mask = ~is_box_near_crop_edge(
                data["boxes"], crop_box, [0, 0, orig_w, orig_h]
            )
            if not torch.all(keep_mask):
                data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        # data["rles"] = mask_to_rle_pytorch(data["masks"])
        # del data["masks"]
        del data["low_res_masks"]

        return data
    
    def _get_cluster_alg_for_crop(
        self, cropped_im_size: Tuple[int, int], original_im_size: Tuple[int, int]
    ):
        if self.adapt_to_crop_size:
            return get_cluster_alg(
                self.cluster_alg,
                cropped_im_size,
                original_im_size,
            )
        else:
            return self.cluster_alg

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def refine_with_m2m(self, points, point_labels, low_res_masks, points_per_batch):
        new_masks = []
        new_iou_preds = []
        new_low_res_mask = []

        for cur_points, cur_point_labels, low_res_mask in batch_iterator(
            points_per_batch, points, point_labels, low_res_masks
        ):
            best_masks, best_iou_preds, up_low_res_mask = self.predictor._predict(
                cur_points[:, None, :],
                cur_point_labels[:, None],
                mask_input=low_res_mask[:, None, :],
                multimask_output=False,
                return_logits=True,
            )
            new_masks.append(best_masks)
            new_iou_preds.append(best_iou_preds)
            new_low_res_mask.append(up_low_res_mask)
        masks = torch.cat(new_masks, dim=0)
        low_res_mask = torch.cat(new_low_res_mask, dim=0)
        return masks, torch.cat(new_iou_preds, dim=0), low_res_mask

    def get_crops(
        self, image, orig_size, crop_n_layers, crop_overlap_ratio
    ) -> Tuple[List[List[int]], List[int]]:
        if self.crop_mode_name == "boxes":
            crop_boxes, layer_idxs = generate_crop_boxes(
                orig_size, crop_n_layers, crop_overlap_ratio
            )
        elif self.crop_mode_name == "efficient":
            self.predictor.set_image(image)
            attn_scores = self.crop_logger.get_attention_scores()

            _, num_tokens = attn_scores.shape

            patches_labels = build_clusters(
                self.crop_cluster_alg,
                attn_scores,
                metric=self.crop_metric,
            )
            masks = patches_labels_to_masks(
                patches_labels,
                orig_size,
                int(np.sqrt(num_tokens)),
                device=self.predictor.device,
                softmax=False,
            )
            boxes = batched_mask_to_box(
                masks.cpu().bool()
            )  # [K, 4] boxes in xyxy format

            # TODO: maybe we want to switch to interesection over min here too?
            mapping = batched_bm(boxes, torch.zeros_like(boxes[:, 0]), iou_threshold=self.crop_merging_threshold)

            merged_boxes = merge_crop_boxes(boxes, mapping)

            crop_boxes, layer_idxs = generate_crop_efficient(
                orig_size, merged_boxes, crop_overlap_ratio
            )

        return crop_boxes, layer_idxs

    def get_input(self, original_im_size, cropped_im_size, crop_layer_idx, **kwargs):
        """Returns prompt points for a given image crop."""
        if self.prompt_mode_name == "grid":
            points_scale = np.array(cropped_im_size)[None, ::-1]
            inputs_for_image = (
                self.point_grids[crop_layer_idx] * points_scale,
                np.full(len(points_cloud), None),
            )
        elif self.prompt_mode_name == "random":
            points_scale = np.array(cropped_im_size)[None, ::-1]
            inputs_for_image = (
                self.point_grids[crop_layer_idx] * points_scale,
                np.full(len(points_cloud), None),
            )
        elif self.prompt_mode_name == "efficient":
            
            points_scale = np.array(cropped_im_size)[None, ::-1]
            attn_scores = self.logger.get_attention_scores()  # cuda tensors
            cluster_alg = self._get_cluster_alg_for_crop(cropped_im_size, original_im_size)
                
            points_cloud, _ = build_layer_efficient_prompt(
                cluster_alg,
                attn_scores,
                method=self.centres_selection,
                metric=self.metric,
                points_per_cluster=self.points_per_cluster,
            )
            # TODO: we now need these to classify the masks
            self.logger.reset_attentions()
            if points_cloud is None or not isinstance(points_cloud, np.ndarray) or points_cloud.shape[0] == 0:
                inputs_for_image = (None, None)
            else:
                inputs_for_image = (
                    points_cloud * points_scale,
                    np.full((points_cloud.shape[0],), None)
                )
        else:
            raise ValueError("Not supported method, there might be a bug...")

        return inputs_for_image