import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import json
import cv2
from pathlib import Path


class VideoSmallObjectAugmentation:
    """
    Data augmentation strategies specifically designed for small object detection in remote sensing VIDEOS.
    
    Key differences from image-based augmentation:
    1. Temporal consistency: Augmentations must preserve object identity across frames
    2. Motion-aware: Leverage inter-frame motion to enhance small object features
    3. Video-specific augmentations: Temporal jitter, frame dropping, motion blur
    4. Multi-frame Mosaic: Combine frames from different time points
    
    Strategies:
    1. Multi-scale training with temporal consistency
    2. Temporal Mosaic: Combine frames from different videos/time points
    3. Copy-Paste with motion trajectory
    4. Frame dropping and interpolation
    5. Motion blur simulation
    6. Color jitter (consistent across frames)
    7. Temporal flip (reverse frame order)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        multi_scale_range: Tuple[int, int] = (480, 800),
        temporal_mosaic_prob: float = 0.3,
        copy_paste_prob: float = 0.2,
        frame_drop_prob: float = 0.2,
        color_jitter_prob: float = 0.4,
        motion_blur_prob: float = 0.3,
        temporal_flip_prob: float = 0.2,
        small_object_threshold: float = 0.01,  # Objects with area < 1% of image are "small"
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_frames: int = 16
    ):
        self.target_size = target_size
        self.multi_scale_range = multi_scale_range
        self.temporal_mosaic_prob = temporal_mosaic_prob
        self.copy_paste_prob = copy_paste_prob
        self.frame_drop_prob = frame_drop_prob
        self.color_jitter_prob = color_jitter_prob
        self.motion_blur_prob = motion_blur_prob
        self.temporal_flip_prob = temporal_flip_prob
        self.small_object_threshold = small_object_threshold
        self.mean = mean
        self.std = std
        self.max_frames = max_frames
        
        # Base transform (resize + normalize)
        self.base_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def __call__(self, images: List[Image.Image], all_bboxes: List[List[Dict]], is_training: bool = True):
        """
        Apply video augmentations to a sequence of frames
        
        Args:
            images: List of PIL Images (video frames in temporal order)
            all_bboxes: List of bbox lists, one list per frame
                       Each bbox dict: 'category_id', 'bbox' [x, y, w, h] in pixel coords
            is_training: Whether to apply training augmentations
        
        Returns:
            augmented_images: List of transformed image tensors
            augmented_bboxes: List of transformed bbox lists (one per frame)
        """
        if not is_training or len(images) == 0:
            # No augmentation for val/test
            return [self.base_transform(img) for img in images], all_bboxes
        
        # Convert to OpenCV format
        img_nps = [np.array(image)[:, :, ::-1] for image in images]  # RGB to BGR
        num_frames = len(img_nps)
        
        # Get original dimensions (assume all frames have same size)
        img_h, img_w = img_nps[0].shape[:2]
        
        # Store bboxes in pixel coordinates
        aug_bboxes = []
        for frame_bboxes in all_bboxes:
            frame_bboxes_px = []
            for bbox in frame_bboxes:
                x, y, w, h = bbox['bbox']
                frame_bboxes_px.append({
                    'category_id': bbox['category_id'],
                    'bbox': [x, y, w, h]  # Already in pixel coords
                })
            aug_bboxes.append(frame_bboxes_px)
        
        # 1. Consistent multi-scale resizing (same scale for all frames)
        scale_size = random.randint(self.multi_scale_range[0], self.multi_scale_range[1])
        scale_factor = scale_size / max(img_h, img_w)
        new_w = int(img_w * scale_factor)
        new_h = int(img_h * scale_factor)
        
        img_nps = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR) 
                   for img in img_nps]
        
        # Scale bboxes consistently
        for frame_bboxes in aug_bboxes:
            for bbox in frame_bboxes:
                x, y, w, h = bbox['bbox']
                bbox['bbox'] = [x * scale_factor, y * scale_factor, 
                               w * scale_factor, h * scale_factor]
        
        img_h, img_w = img_nps[0].shape[:2]
        
        # 2. Temporal Mosaic (probabilistic) - combine frames from different times
        if random.random() < self.temporal_mosaic_prob and num_frames >= 4:
            img_nps, aug_bboxes = self._apply_temporal_mosaic(img_nps, aug_bboxes, img_w, img_h)
            img_h, img_w = img_nps[0].shape[:2]
        
        # 3. Copy-Paste small objects with motion awareness (probabilistic)
        if random.random() < self.copy_paste_prob:
            img_nps, aug_bboxes = self._apply_copy_paste_motion(img_nps, aug_bboxes, img_w, img_h)
        
        # 4. Frame dropping and interpolation (simulates variable frame rate)
        if random.random() < self.frame_drop_prob and num_frames > 4:
            img_nps, aug_bboxes = self._apply_frame_dropping(img_nps, aug_bboxes)
        
        # 5. Consistent color jitter across all frames
        if random.random() < self.color_jitter_prob:
            img_nps = self._apply_consistent_color_jitter(img_nps)
        
        # 6. Motion blur (probabilistic, direction-aware)
        if random.random() < self.motion_blur_prob:
            img_nps = self._apply_motion_blur(img_nps)
        
        # 7. Temporal flip (reverse frame order, probabilistic)
        if random.random() < self.temporal_flip_prob:
            img_nps = img_nps[::-1]
            aug_bboxes = aug_bboxes[::-1]
        
        # Convert back to PIL and apply base transform
        aug_images = []
        for img_np in img_nps:
            img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            aug_images.append(self.base_transform(img_pil))
        
        # Normalize bboxes to [0, 1] based on current image size
        for frame_bboxes in aug_bboxes:
            for bbox in frame_bboxes:
                x, y, w, h = bbox['bbox']
                bbox['bbox'] = [x / img_w, y / img_h, w / img_w, h / img_h]
        
        return aug_images, aug_bboxes
    
    def _apply_temporal_mosaic(self, img_nps: list, all_bboxes: list, img_w: int, img_h: int):
        """
        Apply Temporal Mosaic augmentation for videos
        
        Combines frames from different time points to create a richer training sample.
        """
        num_frames = len(img_nps)
        if num_frames < 4:
            return img_nps, all_bboxes
        
        # Select 4 frames at different temporal positions
        frame_indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
        selected_frames = [img_nps[i] for i in frame_indices]
        selected_bboxes = [all_bboxes[i] for i in frame_indices]
        
        output_h, output_w = img_h, img_w
        mosaic_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        
        center_x, center_y = output_w // 2, output_h // 2
        positions = [
            (0, 0, 0),
            (center_x, 0, 1),
            (0, center_y, 2),
            (center_x, center_y, 3)
        ]
        
        mosaic_bboxes_list = [[] for _ in range(num_frames)]
        
        for x_off, y_off, idx in positions:
            quad_img = selected_frames[idx]
            quad_bboxes = selected_bboxes[idx]
            quad_h, quad_w = quad_img.shape[:2]
            
            place_w = min(quad_w, output_w - x_off)
            place_h = min(quad_h, output_h - y_off)
            
            mosaic_img[y_off:y_off+place_h, x_off:x_off+place_w] =                 cv2.resize(quad_img, (place_w, place_h))
            
            for bbox in quad_bboxes:
                x, y, w, h = bbox['bbox']
                abs_x = x * quad_w + x_off
                abs_y = y * quad_h + y_off
                abs_w = w * quad_w
                abs_h = h * quad_h
                
                if (abs_x >= 0 and abs_x + abs_w <= output_w and
                    abs_y >= 0 and abs_y + abs_h <= output_h):
                    norm_x = abs_x / output_w
                    norm_y = abs_y / output_h
                    norm_w = abs_w / output_w
                    norm_h = abs_h / output_h
                    
                    mosaic_bboxes_list[0].append({
                        'category_id': bbox['category_id'],
                        'bbox': [norm_x, norm_y, norm_w, norm_h]
                    })
        
        result_imgs = [mosaic_img] + img_nps[1:]
        result_bboxes = [mosaic_bboxes_list[0]] + all_bboxes[1:]
        
        return result_imgs, result_bboxes
    
    def _apply_copy_paste_motion(self, img_nps: list, all_bboxes: list, img_w: int, img_h: int):
        """
        Copy-Paste augmentation with motion awareness for videos
        """
        small_objects = []
        
        for bbox in all_bboxes[0]:
            x, y, w, h = bbox['bbox']
            area = w * h
            if area < self.small_object_threshold:
                small_objects.append(bbox)
        
        if not small_objects:
            return img_nps, all_bboxes
        
        result_imgs = [img.copy() for img in img_nps]
        result_bboxes = [frame_bboxes.copy() for frame_bboxes in all_bboxes]
        
        num_pastes = min(len(small_objects), 3)
        
        for _ in range(num_pastes):
            src_bbox = random.choice(small_objects)
            x, y, w, h = src_bbox['bbox']
            
            x_px = x * img_w
            y_px = y * img_h
            w_px = w * img_w
            h_px = h * img_h
            
            padding = 2
            x1 = max(0, int(x_px - padding))
            y1 = max(0, int(y_px - padding))
            x2 = min(img_w, int(x_px + w_px + padding))
            y2 = min(img_h, int(y_px + h_px + padding))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            obj_patch = result_imgs[0][y1:y2, x1:x2].copy()
            
            dx, dy = 0, 0
            if len(all_bboxes) > 1:
                for bbox1 in all_bboxes[0]:
                    for bbox2 in all_bboxes[1]:
                        if bbox1['category_id'] == bbox2['category_id']:
                            x1_norm, y1_norm = bbox1['bbox'][0], bbox1['bbox'][1]
                            x2_norm, y2_norm = bbox2['bbox'][0], bbox2['bbox'][1]
                            dx += (x2_norm - x1_norm) * img_w
                            dy += (y2_norm - y1_norm) * img_h
            
            for frame_idx in range(min(3, len(result_imgs))):
                new_x = int(x_px + dx * frame_idx + random.randint(-10, 10))
                new_y = int(y_px + dy * frame_idx + random.randint(-10, 10))
                
                new_x = max(0, min(new_x, img_w - (x2 - x1)))
                new_y = max(0, min(new_y, img_h - (y2 - y1)))
                
                paste_h = min(y2 - y1, img_h - new_y)
                paste_w = min(x2 - x1, img_w - new_x)
                
                if paste_h > 0 and paste_w > 0:
                    result_imgs[frame_idx][new_y:new_y+paste_h, new_x:new_x+paste_w] =                         obj_patch[:paste_h, :paste_w]
                    
                    new_cx = (new_x + paste_w / 2) / img_w
                    new_cy = (new_y + paste_h / 2) / img_h
                    new_w_norm = paste_w / img_w
                    new_h_norm = paste_h / img_h
                    
                    result_bboxes[frame_idx].append({
                        'category_id': src_bbox['category_id'],
                        'bbox': [new_cx, new_cy, new_w_norm, new_h_norm]
                    })
        
        return result_imgs, result_bboxes
    
    def _apply_frame_dropping(self, img_nps: list, all_bboxes: list):
        """
        Frame dropping and interpolation
        """
        num_frames = len(img_nps)
        if num_frames <= 4:
            return img_nps, all_bboxes
        
        num_to_drop = random.randint(1, min(2, num_frames - 3))
        drop_indices = random.sample(range(1, num_frames - 1), num_to_drop)
        
        kept_indices = [i for i in range(num_frames) if i not in drop_indices]
        result_imgs = [img_nps[i] for i in kept_indices]
        result_bboxes = [all_bboxes[i] for i in kept_indices]
        
        while len(result_imgs) < num_frames:
            max_gap = 0
            insert_pos = 0
            for i in range(len(kept_indices) - 1):
                gap = kept_indices[i + 1] - kept_indices[i]
                if gap > max_gap:
                    max_gap = gap
                    insert_pos = i + 1
            
            if max_gap <= 1:
                break
            
            idx1 = kept_indices[insert_pos - 1]
            idx2 = kept_indices[insert_pos]
            
            alpha = 0.5
            interpolated_img = cv2.addWeighted(img_nps[idx1], alpha, img_nps[idx2], 1 - alpha, 0)
            
            interpolated_bboxes = []
            bboxes1_dict = {b['category_id']: b for b in all_bboxes[idx1]}
            bboxes2_dict = {b['category_id']: b for b in all_bboxes[idx2]}
            
            for cat_id in set(bboxes1_dict.keys()) | set(bboxes2_dict.keys()):
                if cat_id in bboxes1_dict and cat_id in bboxes2_dict:
                    b1 = bboxes1_dict[cat_id]['bbox']
                    b2 = bboxes2_dict[cat_id]['bbox']
                    avg_bbox = [(b1[i] + b2[i]) / 2 for i in range(4)]
                    interpolated_bboxes.append({
                        'category_id': cat_id,
                        'bbox': avg_bbox
                    })
            
            result_imgs.insert(insert_pos, interpolated_img)
            result_bboxes.insert(insert_pos, interpolated_bboxes)
            kept_indices.insert(insert_pos, kept_indices[insert_pos - 1] + 1)
        
        return result_imgs, result_bboxes
    
    def _apply_consistent_color_jitter(self, img_nps: list):
        """
        Apply consistent color jitter across all frames
        """
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)
        
        result_imgs = []
        for img in img_nps:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            hsv[:, :, 0] += hue * 180
            hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
            img_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            img_jittered = cv2.convertScaleAbs(img_jittered, alpha=contrast, beta=brightness * 10)
            result_imgs.append(img_jittered)
        
        return result_imgs
    
    def _apply_motion_blur(self, img_nps: list):
        """
        Apply motion blur to simulate fast-moving objects
        """
        angle = random.uniform(0, 360)
        kernel_size = random.choice([3, 5, 7])
        
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        dx = int(center * np.cos(np.radians(angle)))
        dy = int(center * np.sin(np.radians(angle)))
        
        cv2.line(kernel, (center - dx, center - dy), 
                (center + dx, center + dy), 1, 1)
        kernel /= np.sum(kernel)
        
        result_imgs = []
        for img in img_nps:
            blurred = cv2.filter2D(img, -1, kernel)
            result_imgs.append(blurred)
        
        return result_imgs


        """
        Apply Mosaic augmentation: combine 4 images into one
        
        This increases small object density and provides context variation.
        """
        # Create output image
        output_h, output_w = img.shape[:2]
        mosaic_img = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        mosaic_bboxes = []
        
        # Define 4 quadrants
        center_x, center_y = output_w // 2, output_h // 2
        
        # We need 3 more images - for simplicity, use the same image with different crops
        # In practice, you would load 3 additional random images from the dataset
        images_and_bboxes = [(img, bboxes)]
        
        for _ in range(3):
            # Create a variant of the original image with random crop
            crop_w = random.randint(int(img_w * 0.5), img_w)
            crop_h = random.randint(int(img_h * 0.5), img_h)
            x_start = random.randint(0, max(0, img_w - crop_w))
            y_start = random.randint(0, max(0, img_h - crop_h))
            
            cropped_img = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
            
            # Adjust bboxes for crop
            cropped_bboxes = []
            for bbox in bboxes:
                cx, cy, w, h = bbox['bbox']
                cx_px, cy_px = cx * img_w, cy * img_h
                w_px, h_px = w * img_w, h * img_h
                
                # Check if bbox is in crop region
                x1, y1 = cx_px - w_px/2, cy_px - h_px/2
                x2, y2 = cx_px + w_px/2, cy_px + h_px/2
                
                if x1 < x_start or x2 > x_start+crop_w or y1 < y_start or y2 > y_start+crop_h:
                    continue  # Skip bboxes outside crop
                
                # Adjust to crop coordinates
                new_cx = (cx_px - x_start) / crop_w
                new_cy = (cy_px - y_start) / crop_h
                new_w = w_px / crop_w
                new_h = h_px / crop_h
                
                cropped_bboxes.append({
                    'category_id': bbox['category_id'],
                    'bbox': [new_cx, new_cy, new_w, new_h]
                })
            
            images_and_bboxes.append((cropped_img, cropped_bboxes))
        
        # Place 4 images in quadrants
        positions = [
            (0, 0, images_and_bboxes[0]),      # top-left
            (center_x, 0, images_and_bboxes[1]),  # top-right
            (0, center_y, images_and_bboxes[2]),  # bottom-left
            (center_x, center_y, images_and_bboxes[3])  # bottom-right
        ]
        
        for x_off, y_off, (quad_img, quad_bboxes) in positions:
            quad_h, quad_w = quad_img.shape[:2]
            
            # Calculate actual placement size
            place_w = min(quad_w, output_w - x_off)
            place_h = min(quad_h, output_h - y_off)
            
            mosaic_img[y_off:y_off+place_h, x_off:x_off+place_w] = \
                cv2.resize(quad_img, (place_w, place_h))
            
            # Adjust bboxes for quadrant position
            for bbox in quad_bboxes:
                cx, cy, w, h = bbox['bbox']
                
                # Convert to absolute coordinates in quadrant
                abs_cx = cx * quad_w + x_off
                abs_cy = cy * quad_h + y_off
                abs_w = w * quad_w
                abs_h = h * quad_h
                
                # Check if bbox is within final image bounds
                if (abs_cx - abs_w/2 >= 0 and abs_cx + abs_w/2 <= output_w and
                    abs_cy - abs_h/2 >= 0 and abs_cy + abs_h/2 <= output_h):
                    
                    # Normalize to output image size
                    norm_cx = abs_cx / output_w
                    norm_cy = abs_cy / output_h
                    norm_w = abs_w / output_w
                    norm_h = abs_h / output_h
                    
                    mosaic_bboxes.append({
                        'category_id': bbox['category_id'],
                        'bbox': [norm_cx, norm_cy, norm_w, norm_h]
                    })
        
        return mosaic_img, mosaic_bboxes
    
    def _apply_copy_paste(self, img: np.ndarray, bboxes: List[Dict], img_w: int, img_h: int):
        """
        Copy-Paste augmentation for small objects
        
        Identify small objects and paste copies at random locations.
        """
        img_h, img_w = img.shape[:2]
        small_objects = []
        
        # Identify small objects
        for bbox in bboxes:
            cx, cy, w, h = bbox['bbox']
            area = w * h
            if area < self.small_object_threshold:
                small_objects.append(bbox)
        
        if not small_objects:
            return img, bboxes
        
        aug_bboxes = bboxes.copy()
        aug_img = img.copy()
        
        # Paste up to 5 copies of small objects
        num_pastes = min(len(small_objects), 5)
        
        for _ in range(num_pastes):
            # Select a random small object
            src_bbox = random.choice(small_objects)
            cx, cy, w, h = src_bbox['bbox']
            
            # Convert to pixel coordinates
            cx_px = cx * img_w
            cy_px = cy * img_h
            w_px = w * img_w
            h_px = h * img_h
            
            # Extract object patch (with some padding)
            padding = 2
            x1 = max(0, int(cx_px - w_px/2 - padding))
            y1 = max(0, int(cy_px - h_px/2 - padding))
            x2 = min(img_w, int(cx_px + w_px/2 + padding))
            y2 = min(img_h, int(cy_px + h_px/2 + padding))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            obj_patch = aug_img[y1:y2, x1:x2].copy()
            
            # Find a random location to paste
            max_attempts = 10
            for _ in range(max_attempts):
                new_x = random.randint(0, img_w - (x2 - x1))
                new_y = random.randint(0, img_h - (y2 - y1))
                
                # Check if paste location overlaps significantly with existing objects
                overlap = False
                new_cx = (new_x + (x2 - x1) / 2) / img_w
                new_cy = (new_y + (y2 - y1) / 2) / img_h
                new_w = (x2 - x1) / img_w
                new_h = (y2 - y1) / img_h
                
                for existing_bbox in aug_bboxes:
                    ex_cx, ex_cy, ex_w, ex_h = existing_bbox['bbox']
                    # Simple overlap check
                    if (abs(new_cx - ex_cx) < (new_w + ex_w) / 2 and
                        abs(new_cy - ex_cy) < (new_h + ex_h) / 2):
                        overlap = True
                        break
                
                if not overlap:
                    # Paste the object
                    aug_img[new_y:new_y+(y2-y1), new_x:new_x+(x2-x1)] = obj_patch
                    
                    # Add new bbox
                    aug_bboxes.append({
                        'category_id': src_bbox['category_id'],
                        'bbox': [new_cx, new_cy, new_w, new_h]
                    })
                    break
        
        return aug_img, aug_bboxes
    
    def _random_crop_preserve_small(self, img: np.ndarray, bboxes: List[Dict]):
        """
        Random crop that preserves small objects
        
        Ensures that at least one small object remains in the cropped region.
        """
        img_h, img_w = img.shape[:2]
        
        # Identify small objects
        small_objects = []
        for bbox in bboxes:
            cx, cy, w, h = bbox['bbox']
            area = w * h
            if area < self.small_object_threshold:
                small_objects.append((cx * img_w, cy * img_h, w * img_w, h * img_h))
        
        # If no small objects, do standard random crop
        if not small_objects:
            return self._standard_random_crop(img, bboxes)
        
        # Select a random small object to preserve
        preserve_obj = random.choice(small_objects)
        obj_cx, obj_cy, obj_w, obj_h = preserve_obj
        
        # Determine crop region that includes this object
        min_crop_w = int(obj_w * 3)  # At least 3x the object width
        min_crop_h = int(obj_h * 3)
        
        crop_w = random.randint(max(min_crop_w, int(img_w * 0.5)), img_w)
        crop_h = random.randint(max(min_crop_h, int(img_h * 0.5)), img_h)
        
        # Ensure crop includes the object
        max_x_start = min(obj_cx - obj_w/2, img_w - crop_w)
        min_x_start = max(0, obj_cx + obj_w/2 - crop_w)
        max_y_start = min(obj_cy - obj_h/2, img_h - crop_h)
        min_y_start = max(0, obj_cy + obj_h/2 - crop_h)
        
        if min_x_start > max_x_start:
            min_x_start = max_x_start
        if min_y_start > max_y_start:
            min_y_start = max_y_start
        
        x_start = random.randint(int(min_x_start), int(max_x_start))
        y_start = random.randint(int(min_y_start), int(max_y_start))
        
        # Crop image
        cropped_img = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
        
        # Adjust bboxes
        cropped_bboxes = []
        for bbox in bboxes:
            cx, cy, w, h = bbox['bbox']
            cx_px, cy_px = cx * img_w, cy * img_h
            w_px, h_px = w * img_w, h * img_h
            
            # Check if bbox center is in crop region
            if (x_start <= cx_px <= x_start + crop_w and
                y_start <= cy_px <= y_start + crop_h):
                
                # Adjust to crop coordinates and normalize
                new_cx = (cx_px - x_start) / crop_w
                new_cy = (cy_px - y_start) / crop_h
                new_w = w_px / crop_w
                new_h = h_px / crop_h
                
                # Ensure bbox is fully within crop (with some tolerance)
                if (new_cx - new_w/2 >= -0.1 and new_cx + new_w/2 <= 1.1 and
                    new_cy - new_h/2 >= -0.1 and new_cy + new_h/2 <= 1.1):
                    cropped_bboxes.append({
                        'category_id': bbox['category_id'],
                        'bbox': [new_cx, new_cy, new_w, new_h]
                    })
        
        # If no bboxes remain, fall back to original
        if not cropped_bboxes:
            return img, bboxes
        
        return cropped_img, cropped_bboxes
    
    def _standard_random_crop(self, img: np.ndarray, bboxes: List[Dict]):
        """Standard random crop without small object preservation"""
        img_h, img_w = img.shape[:2]
        
        # Random crop size (at least 50% of original)
        crop_w = random.randint(int(img_w * 0.5), img_w)
        crop_h = random.randint(int(img_h * 0.5), img_h)
        
        x_start = random.randint(0, max(0, img_w - crop_w))
        y_start = random.randint(0, max(0, img_h - crop_h))
        
        cropped_img = img[y_start:y_start+crop_h, x_start:x_start+crop_w]
        
        # Adjust bboxes
        cropped_bboxes = []
        for bbox in bboxes:
            cx, cy, w, h = bbox['bbox']
            cx_px, cy_px = cx * img_w, cy * img_h
            
            # Check if bbox center is in crop region
            if (x_start <= cx_px <= x_start + crop_w and
                y_start <= cy_px <= y_start + crop_h):
                
                new_cx = (cx_px - x_start) / crop_w
                new_cy = (cy_px - y_start) / crop_h
                new_w = w * img_w / crop_w
                new_h = h * img_h / crop_h
                
                cropped_bboxes.append({
                    'category_id': bbox['category_id'],
                    'bbox': [new_cx, new_cy, new_w, new_h]
                })
        
        if not cropped_bboxes:
            return img, bboxes
        
        return cropped_img, cropped_bboxes
    
    def _apply_color_jitter(self, img: np.ndarray):
        """Apply color jitter to image"""
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Random brightness
        brightness_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        
        # Random saturation
        saturation_factor = random.uniform(0.7, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        
        # Random hue
        hue_shift = random.uniform(-10, 10)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 180)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _apply_blur(self, img: np.ndarray):
        """Apply Gaussian blur to image"""
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

class FSVODDataset(Dataset):
    """
    Few-Shot Video Object Detection Dataset
    
    This dataset implements N-way K-shot learning for video object detection.
    Each episode consists of:
    - Support set: N classes with K examples each
    - Query set: A video sequence with frames containing objects of the N classes
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        n_way: int = 5,
        k_shot: int = 5,
        max_frames: int = 16,
        frame_stride: int = 1,
        transform=None,
        target_size: Tuple[int, int] = (560, 560),
        episode_length: int = 100,
        seed: int = 42,
        use_small_object_aug: bool = True  # NEW: Enable small object augmentation
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            max_frames: Maximum number of frames per video
            frame_stride: Stride for sampling frames
            transform: Image transformations
            target_size: Target size for resizing images (H, W)
            episode_length: Number of episodes in an epoch
            seed: Random seed for reproducibility
            use_small_object_aug: Whether to use small object specific augmentations (for training)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.n_way = n_way
        self.k_shot = k_shot
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.target_size = target_size
        self.episode_length = episode_length
        self.use_small_object_aug = use_small_object_aug and (split == 'train')
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Initialize video small object augmentation for training
        if self.use_small_object_aug:
            self.small_obj_aug = VideoSmallObjectAugmentation(
                target_size=target_size,
                multi_scale_range=(480, 800),
                temporal_mosaic_prob=0.3,
                copy_paste_prob=0.2,
                frame_drop_prob=0.2,
                color_jitter_prob=0.4,
                motion_blur_prob=0.3,
                temporal_flip_prob=0.2,
                max_frames=max_frames
            )
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load annotation files
        self.annotations = self._load_annotations()
        
        # Get all available classes
        self.all_classes = self._get_all_classes()
        
        # Split classes into base and novel based on split
        self.base_classes, self.novel_classes = self._split_classes()
        
        # For test and val, we use fixed episodes
        if split in ['test', 'val']:
            self.episodes = self._generate_episodes()
            # self.episodes = None
        else:
            self.episodes = None
    
    def _load_annotations(self):
        """Load dataset annotations"""
        annotation_file = self.root_dir / f"{self.split}_annotations.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file {annotation_file} not found")
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def _get_all_classes(self):
        """Get all available classes in the dataset"""
        classes = set()
        
        for video_id, video_info in self.annotations.items():
            for frame_id, frame_info in video_info['frames'].items():
                for obj in frame_info['objects']:
                    classes.add(obj['category_id'])
        
        return sorted(list(classes))
    
    def _split_classes(self):
        """Split classes into base and novel based on split"""
        # For simplicity, we use a deterministic split based on class IDs
        # In practice, this should be defined by the dataset
        all_classes = self.all_classes
        n_classes = len(all_classes)
        
        # 60% for base classes, 20% for validation, 20% for testing
        n_base = int(n_classes * 0.6)
        n_val = int(n_classes * 0.2)
        
        if self.split == 'train':
            # Base classes for training
            return all_classes[:n_base], []
        elif self.split == 'val':
            # Novel classes for validation
            return all_classes[:n_base], all_classes[n_base:n_base+n_val]
        else:  # test
            # Novel classes for testing
            return all_classes[:n_base], all_classes[n_base+n_val:]
    
    def _get_support_examples(self, class_id):
        """Get support examples for a specific class"""
        examples = []
        
        # Find all objects of this class in the dataset
        for video_id, video_info in self.annotations.items():
            for frame_id, frame_info in video_info['frames'].items():
                for obj in frame_info['objects']:
                    if obj['category_id'] == class_id:
                        # Convert bbox from [x, y, w, h] to [cx, cy, w, h]
                        x, y, w, h = obj['bbox']
                        cx, cy = x + w / 2, y + h / 2
                        examples.append({
                            'video_id': video_id,
                            'frame_id': frame_id,
                            'bbox': [cx, cy, w, h]  # [cx, cy, w, h] format
                        })
        
        # Randomly select k_shot examples
        if len(examples) >= self.k_shot:
            return random.sample(examples, self.k_shot)
        else:
            # If not enough examples, sample with replacement
            return random.choices(examples, k=self.k_shot)
    
    def _get_query_video(self, class_ids):
        """Get a query video containing objects of the selected classes"""
        valid_videos = []
        
        # Find videos containing all selected classes
        for video_id, video_info in self.annotations.items():
            video_classes = set()
            
            for frame_info in video_info['frames'].values():
                for obj in frame_info['objects']:
                    video_classes.add(obj['category_id'])
            
            # Check if this video contains all selected classes
            if all(cls_id in video_classes for cls_id in class_ids):
                valid_videos.append(video_id)
        
        # If no valid videos, relax constraint to contain at least one class
        if not valid_videos:
            valid_videos = []
            for video_id, video_info in self.annotations.items():
                video_classes = set()
                
                for frame_info in video_info['frames'].values():
                    for obj in frame_info['objects']:
                        video_classes.add(obj['category_id'])
                
                # Check if this video contains at least one selected class
                if any(cls_id in video_classes for cls_id in class_ids):
                    valid_videos.append(video_id)
        
        # Randomly select a video
        if valid_videos:
            selected_video_id = random.choice(valid_videos)
            return self._prepare_query_video(selected_video_id, class_ids)
        else:
            # Fallback: just select any video
            selected_video_id = random.choice(list(self.annotations.keys()))
            return self._prepare_query_video(selected_video_id, class_ids)
    
    def _prepare_query_video(self, video_id, class_ids):
        """Prepare query video by selecting appropriate frames"""
        video_info = self.annotations[video_id]
        
        # Sort frames by ID
        frame_ids = sorted(video_info['frames'].keys())
        
        # Select frames with stride
        selected_frames = frame_ids[::self.frame_stride]
        
        # Limit to max_frames
        if len(selected_frames) > self.max_frames:
            selected_frames = selected_frames[:self.max_frames]
        
        # Prepare frame data
        frames_data = []
        for frame_id in selected_frames:
            frame_info = video_info['frames'][frame_id]
            
            # Filter objects by selected classes
            objects = [obj for obj in frame_info['objects'] if obj['category_id'] in class_ids]
            
            frames_data.append({
                'frame_id': frame_id,
                'image_path': os.path.join(self.root_dir, 'videos', video_id, f"{frame_id}.jpg"),
                'objects': objects
            })
        
        return {
            'video_id': video_id,
            'frames': frames_data
        }
    
    def _generate_episodes(self):
        """Generate fixed episodes for validation and testing"""
        episodes = []
        
        # Available classes for this split
        available_classes = self.novel_classes if self.novel_classes else self.base_classes
        
        for _ in range(self.episode_length):
            # Randomly select n_way classes
            if len(available_classes) >= self.n_way:
                episode_classes = random.sample(available_classes, self.n_way)
            else:
                # If not enough classes, sample with replacement
                episode_classes = random.choices(available_classes, k=self.n_way)
            
            # Get support examples for each class
            support_set = {}
            for cls_id in episode_classes:
                support_set[cls_id] = self._get_support_examples(cls_id)
            
            # Get query video
            query_video = self._get_query_video(episode_classes)
            
            episodes.append({
                'classes': episode_classes,
                'support_set': support_set,
                'query_video': query_video
            })
        
        return episodes
    
    def _load_image(self, image_path, bbox=None):
        """Load and preprocess an image, optionally cropping to bbox
        
        Args:
            image_path: Path to the image
            bbox: Bounding box in [cx, cy, w, h] format (normalized) for cropping support image
            query_bboxes: List of bounding boxes in [cx, cy, w, h] format (pixel coordinates) 
                         for query image that need to be scaled after resize
        
        Returns:
            image: Transformed image tensor
            scaled_query_bboxes: List of scaled bounding boxes (only if query_bboxes is provided)
        """
        try:
            image = Image.open(image_path).convert('RGB')
            orig_w, orig_h = image.size
            
            if bbox is not None:
                # Crop to bounding box
                # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2] for PIL crop
                cx, cy, w, h = bbox  # normalized coordinates
                # Denormalize to pixel coordinates
                cx_px, cy_px = cx * orig_w, cy * orig_h
                w_px, h_px = w * orig_w, h * orig_h
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                image = image.crop((x1, y1, x2, y2))
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return empty tensor as fallback
            return torch.zeros((3, *self.target_size))
    
    def _prepare_episode(self, idx):
        """Prepare an episode (support set + query video)"""
        if self.split in ['test', 'val'] and self.episodes is not None:
            # Use pre-generated episodes for test/val
            episode = self.episodes[idx % len(self.episodes)]
        else:
            # Generate dynamic episode for training
            # Available classes for this split
            available_classes = self.novel_classes if self.novel_classes else self.base_classes
            
            # Randomly select n_way classes
            if len(available_classes) >= self.n_way:
                episode_classes = random.sample(available_classes, self.n_way)
            else:
                # If not enough classes, sample with replacement
                episode_classes = random.choices(available_classes, k=self.n_way)
            
            # Get support examples for each class
            support_set = {}
            for cls_id in episode_classes:
                support_set[cls_id] = self._get_support_examples(cls_id)
            
            # Get query video
            query_video = self._get_query_video(episode_classes)
            
            episode = {
                'classes': episode_classes,
                'support_set': support_set,
                'query_video': query_video
            }
        
        # Load support images
        support_images = {}
        for cls_id, examples in episode['support_set'].items():
            cls_images = []
            for example in examples:
                image_path = os.path.join(
                    self.root_dir, 'videos', example['video_id'], f"{example['frame_id']}.jpg"
                )
                # Load image and crop to object
                image = self._load_image(image_path, example['bbox'])
                cls_images.append(image)
            support_images[cls_id] = torch.stack(cls_images)
        
        # Load query video frames with small object augmentation (for training)
        query_frames = []
        frame_annotations = []
        
        for frame_data in episode['query_video']['frames']:
            # Get original image size before any processing
            with Image.open(frame_data['image_path']) as img:
                orig_w, orig_h = img.size

            # Prepare bboxes in normalized format for augmentation
            bboxes_for_aug = []
            for obj in frame_data['objects']:
                if obj['category_id'] in episode['classes']:
                    x, y, w, h = obj['bbox']
                    cx, cy = x + w / 2, y + h / 2
                    # Normalize to [0, 1]
                    bboxes_for_aug.append({
                        'category_id': obj['category_id'],
                        'bbox': [cx / orig_w, cy / orig_h, w / orig_w, h / orig_h]
                    })
            
            # Load and augment frame image
            image = Image.open(frame_data['image_path']).convert('RGB')
            
            if self.use_small_object_aug and bboxes_for_aug:
                # Apply small object augmentation
                frame, augmented_bboxes = self.small_obj_aug(image, bboxes_for_aug, is_training=True)
                
                # Convert augmented bboxes back to absolute coordinates in target_size
                objects = []
                for bbox_dict in augmented_bboxes:
                    cx_norm, cy_norm, w_norm, h_norm = bbox_dict['bbox']
                    cx = cx_norm * self.target_size[1]  # W
                    cy = cy_norm * self.target_size[0]  # H
                    w = w_norm * self.target_size[1]
                    h = h_norm * self.target_size[0]
                    objects.append({
                        'category_id': bbox_dict['category_id'],
                        'bbox': [cx, cy, w, h]
                    })
            else:
                # Standard loading without augmentation (for val/test or no bboxes)
                frame = self._load_image(frame_data['image_path'])
                
                # Prepare query bboxes for this frame (in target_size coordinates)
                objects = []
                for obj in frame_data['objects']:
                    if obj['category_id'] in episode['classes']:
                        x, y, w, h = obj['bbox']
                        cx, cy = x + w / 2, y + h / 2
                        # Scale to target_size
                        scale_x = self.target_size[1] / orig_w
                        scale_y = self.target_size[0] / orig_h
                        objects.append({
                            'category_id': obj['category_id'],
                            'bbox': [cx * scale_x, cy * scale_y, w * scale_x, h * scale_y],
                        })
            
            query_frames.append(frame)
            
            frame_annotations.append({
                'frame_id': frame_data['frame_id'],
                'objects': objects,
                'original_size': (orig_h, orig_w)  # (H, W) format for coordinate mapping
            })
        
        query_frames = torch.stack(query_frames) if query_frames else torch.zeros((0, 3, *self.target_size))
        
        return {
            'support_images': support_images,  # Dict[class_id -> Tensor[K, 3, H, W]]
            'query_frames': query_frames,      # Tensor[T, 3, H, W]
            'frame_annotations': frame_annotations,  # List[Dict] with original bboxes and sizes
            'classes': episode['classes']      # List of class IDs in this episode
        }
    
    def __len__(self):
        """Return the number of episodes in the dataset"""
        return self.episode_length
    
    def __getitem__(self, idx):
        """Get an episode"""
        return self._prepare_episode(idx)


def collate_episodes(batch):
    """
    Custom collate function for episode batches
    
    Args:
        batch: List of episodes from dataset __getitem__
        
    Returns:
        Batched episode data
    """
    # Since each element in batch is already a complete episode,
    # and we process one episode at a time, simply return the first item
    return batch[0]


def get_fsvod_loaders(
    root_dir: str,
    n_way: int = 5,
    k_shot: int = 5,
    max_frames: int = 16,
    batch_size: int = 1,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (640, 640),  # Changed default to 640 for better small object detection
    use_small_object_aug: bool = True  # NEW: Enable small object augmentation for training
):
    """
    Get dataloaders for all splits
    
    Args:
        root_dir: Root directory of the dataset
        n_way: Number of classes per episode
        k_shot: Number of support examples per class
        max_frames: Maximum number of frames per video
        batch_size: Batch size (typically 1 for episodes)
        num_workers: Number of workers for dataloader
        target_size: Target size for resizing images (H, W)
        use_small_object_aug: Whether to use small object specific augmentations
        
    Returns:
        Dict containing dataloaders for train, val, and test splits
    """
    # Create datasets with small object augmentation enabled for training
    train_dataset = FSVODDataset(
        root_dir=root_dir,
        split='train',
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        transform=None,  # Transform is handled internally by SmallObjectAugmentation
        target_size=target_size,
        use_small_object_aug=use_small_object_aug
    )
    
    # val_dataset = FSVODDataset(
    #     root_dir=root_dir,
    #     split='val',
    #     n_way=n_way,
    #     k_shot=k_shot,
    #     max_frames=max_frames,
    #     transform=None,
    #     target_size=target_size,
    #     use_small_object_aug=False  # No augmentation for validation
    # )
    
    test_dataset = FSVODDataset(
        root_dir=root_dir,
        split='test',
        n_way=n_way,
        k_shot=k_shot,
        max_frames=max_frames,
        transform=None,  # Use internal default transform
        target_size=target_size,
        episode_length=100,
        use_small_object_aug=False  # No augmentation for testing
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     collate_fn=collate_episodes,
    #     pin_memory=True
    # )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': None,
        'test': test_loader
    }