from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_F
def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def get_foreground_bbox(array_2d):
    """
    Get bounding box of foreground (non-zero) area in a 2D array.

    Args:
        array_2d: numpy array of shape (24, 24) or any 2D array

    Returns:
        tuple: (min_row, min_col, max_row, max_col) or None if no foreground
               - min_row, min_col: top-left corner of bounding box
               - max_row, max_col: bottom-right corner of bounding box (inclusive)
    """
    # Find all non-zero positions
    fg_positions = np.where(array_2d > 0)
    full_bbox = (0, 0, array_2d.shape[0] - 1, array_2d.shape[1] - 1)
    # Check if there are any foreground pixels
    if len(fg_positions[0]) == 0:
        return full_bbox  # return full image as bbox if no foreground

    # Get bounding box coordinates
    min_row = np.min(fg_positions[0])
    max_row = np.max(fg_positions[0])
    min_col = np.min(fg_positions[1])
    max_col = np.max(fg_positions[1])

    return (min_row, min_col, max_row, max_col)


def get_foreground_mask(
    original_image_size: tuple,
    target_image_size: int,
    min_dimension_size: int = 1,
    return_bbox: bool = False,
) -> np.ndarray:
    """
    Calculates a boolean mask indicating the location of the original image
    content after it has been resized and padded into a square.

    This function mimics the logic of resizing an image to fit within a square
    of `target_image_size` while maintaining aspect ratio, and then padding it
    to become a full square.

    Args:
        original_image_size (tuple): A tuple (width, height) representing the
        target_image_size (int): The side length of the final square image (e.g., 336 for a 336x336 output).
        min_dimension_size (int): The minimum size for any dimension after the initial resize, defaults to 1.

    Returns:
        np.ndarray: A 2D boolean NumPy array of shape
                    (target_image_size, target_image_size) where `True`
                    indicates a foreground pixel and `False` indicates a
                    background (padded) pixel.
    """
    # 1. Get original dimensions
    original_width, original_height = original_image_size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Original image dimensions must be positive.")

    # 2. Calculate the size after aspect-ratio-preserving resize
    max_dim = max(original_width, original_height)

    new_height = max(
        int(original_height / max_dim * target_image_size),
        min_dimension_size
    )
    new_width = max(
        int(original_width / max_dim * target_image_size),
        min_dimension_size
    )

    # 3. Determine the offsets based on the `expand2square` padding logic.
    if new_width > new_height:
        x1 = 0
        y1 = (target_image_size - new_height) // 2
        x2 = new_width
        y2 = y1 + new_height
    elif new_height > new_width:
        x1 = (target_image_size - new_width) // 2
        y1 = 0
        x2 = x1 + new_width
        y2 = new_height
    else:
        x1, y1 = 0, 0
        x2, y2 = target_image_size, target_image_size

    # Ensure the coordinates do not exceed the canvas dimensions due to rounding.
    x2 = min(x2, target_image_size)
    y2 = min(y2, target_image_size)

    # 4. Create the boolean mask
    mask = np.zeros((target_image_size, target_image_size), dtype=bool)
    # Note: NumPy slicing is [row_start:row_end, col_start:col_end], which corresponds to [y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = True
    if return_bbox:
        return mask, (y1, x1, y2-1, x2-1)
    return mask

def create_pseudo_labels(sink_attn, grounding_attn_o2i, sink_thresh=1e-3, binary_coff=0.2, max_ratio_limit=0.5, bg_coff=0.1):
    """
    Create pseudo labels for foreground, background, and ignore tokens.

    Args:
        sink_attn: array of shape (576,), layer 2 attention
        grounding_attn_o2i: array of shape (576,), original grounding attention
        sink_thresh: float, threshold to identify sink tokens
        binary_coff: float, coefficient to threshold grounding attention
        K: int, number of background tokens to select
        max_ratio_limit: float, maximum ratio of foreground, exceeding this will set the sample as ignore

    Returns:
        dict: {
            'fg_mask': binary mask for foreground tokens,
            'bg_mask': binary mask for background tokens,
            'ignore_mask': binary mask for ignore tokens,
            'labels': combined labels (0=bg, 1=fg, -1=ignore),
            'fg_bbox': foreground bounding box,
            'stats': statistics about the labeling
        }
    """
    # Ensure all inputs are 1D
    sink_attn = sink_attn.flatten()

    grounding_attn_o2i = grounding_attn_o2i.flatten()
    grounding_attn_o2i_ori = grounding_attn_o2i.copy()
    grounding_attn = grounding_attn_o2i * (sink_attn < sink_thresh).astype(float)  # remove sink tokens

    binary_mask = (grounding_attn > grounding_attn.max() * binary_coff).astype(float)
    grounding_attn *= binary_mask

    # 1. Identify sink tokens
    sink_token_mask = (sink_attn >= sink_thresh).astype(bool)

    # 2. Identify foreground tokens (where grounding_attn > 0)
    fg_mask = (grounding_attn > 0).astype(bool)

    # 3. Get foreground bounding box from grounding_attn
    grounding_attn_2d = grounding_attn.reshape(24, 24)
    fg_bbox = get_foreground_bbox(grounding_attn_2d)

    # 4. Create mask for tokens inside fg bounding box
    min_row, min_col, max_row, max_col = fg_bbox

    # Convert 1D indices to 2D coordinates
    row_indices, col_indices = np.divmod(np.arange(576), 24)

    # Check if each token is inside the bounding box
    in_fg_box_mask = (
        (row_indices >= min_row) & (row_indices <= max_row) &
        (col_indices >= min_col) & (col_indices <= max_col)
    )


    # 5. Find candidate background tokens
    # Tokens that are: NOT in fg box and set sink token score to 0
    bg_candidate_mask = (~in_fg_box_mask)  #& (~sink_token_mask)
    grounding_attn_o2i = grounding_attn_o2i * (sink_attn < sink_thresh).astype(float)  # remove sink tokens

    # 6. Select K background tokens from candidates based on grounding_attn_o2i values
    bg_mask = np.zeros(576, dtype=bool)

    if bg_candidate_mask.sum() > 0:
        # Get candidate positions and their grounding_attn_o2i values
        candidate_indices = np.where(bg_candidate_mask)[0]
        # candidate_values = grounding_attn_o2i[candidate_indices]

        candidate_grounding_values = grounding_attn_o2i[candidate_indices]
        selected_candidates_by_threshold_mask = (candidate_grounding_values < grounding_attn.max()*bg_coff)

        # Get the original indices (from bg_candidate_mask) of these selected candidates
        selected_bg_indices = candidate_indices[selected_candidates_by_threshold_mask]

        bg_mask[selected_bg_indices] = True

    # 7. All other tokens are ignored
    ignore_mask = ~(fg_mask | bg_mask)

    # 8. Create combined labels (-1=ignore, 0=bg, 1=fg)
    labels = np.full(576, -100, dtype=int)  # Start with all ignore
    if (max_col-min_col + 1) * (max_row-min_row + 1) < max_ratio_limit * 576:
        labels[bg_mask] = 0  # Background
        labels[fg_mask] = 1  # Foreground
    else:
        #print(f"Warning: Foreground bounding box too large, setting all as ignore. Ratio: {(max_col-min_col + 1) * (max_row-min_row + 1) / 576:.2f}")
        bg_mask = np.zeros(576, dtype=bool)
        fg_mask = np.zeros(576, dtype=bool)

    # 9. Collect statistics
    stats = {
        'num_fg': fg_mask.sum(),
        'num_bg': bg_mask.sum(),
        'num_ignore': ignore_mask.sum(),
        'num_sink': sink_token_mask.sum(),
        'num_candidates': bg_candidate_mask.sum(),
        'fg_bbox': fg_bbox
    }

    return {
        'fg_mask': fg_mask,
        'bg_mask': bg_mask,
        'ignore_mask': ignore_mask,
        'labels': labels,
        'fg_bbox': fg_bbox,
        'stats': stats
    }



# Helper function for batched bounding box calculation (PyTorch version)
def get_foreground_bbox_torch(attn_map_2d_batch: torch.Tensor, threshold: float = 0.0):
    """
    Calculates foreground bounding boxes for a batch of 2D attention maps.

    Args:
        attn_map_2d_batch (torch.Tensor): Batch of 2D attention maps.
                                          Shape: (B, H, W).
        threshold (float): Threshold to consider a pixel as foreground.

    Returns:
        torch.Tensor: Bounding boxes for each sample.
                      Shape: (B, 4) -> [min_row, min_col, max_row, max_col].
    """
    B, H, W = attn_map_2d_batch.shape
    device = attn_map_2d_batch.device

    bboxes = torch.zeros((B, 4), dtype=torch.long, device=device)

    for i in range(B):
        # Find coordinates of foreground pixels for the current sample
        fg_pixels = (attn_map_2d_batch[i] > threshold).nonzero(as_tuple=False)  # (N_fg_pixels, 2) -> [row, col]

        if fg_pixels.numel() == 0:  # No foreground pixels found
            bboxes[i] = torch.tensor([0, 0, H - 1, W - 1], device=device)  # Default to full image
        else:
            rows = fg_pixels[:, 0]
            cols = fg_pixels[:, 1]
            # bboxes[i] = torch.tensor([
            #     torch.min(rows), torch.min(cols),
            #     torch.max(rows), torch.max(cols)
            # ], device=device)
            bboxes[i] = torch.tensor([
                torch.min(cols), torch.min(rows),
                torch.max(cols)+1, torch.max(rows)+1
            ])
    return bboxes


def create_pseudo_labels_torch(
    sink_attn_batch: torch.Tensor,  # Shape: (B, 576)
    grounding_attn_o2i_batch: torch.Tensor,  # Shape: (B, 576)
    sink_thresh: float = 1e-3,
    binary_coff: float = 0.2,
    # K: int = 100, # K is not used in the provided threshold-based BG selection
    max_ratio_limit: float = 0.5,
    bg_coff: float = 0.1,
    grid_dim: int = 24  # Assuming a 24x24 grid for 576 tokens
):
    """
    Create pseudo labels for foreground, background, and ignore tokens (PyTorch batched version).

    Args:
        sink_attn_batch: Batch of layer 2 attentions, shape (B, 576)
        grounding_attn_o2i_batch: Batch of original grounding attentions, shape (B, 576)
        sink_thresh: Threshold to identify sink tokens
        binary_coff: Coefficient to threshold grounding attention for FG
        max_ratio_limit: Maximum ratio of foreground area, exceeding this sets sample to ignore
        bg_coff: Coefficient to threshold grounding_attn_o2i for BG selection, relative to FG's max attention
        grid_dim: Dimension of the square grid (e.g., 24 for a 24x24 grid)

    Returns:
        dict: {
            'fg_mask': Batched binary mask for foreground (B, 576),
            'bg_mask': Batched binary mask for background (B, 576),
            'ignore_mask': Batched binary mask for ignore (B, 576),
            'labels': Batched combined labels (0=bg, 1=fg, -100=ignore) (B, 576),
            'fg_bbox': Batched foreground bounding boxes (B, 4) [min_r, min_c, max_r, max_c],
            'stats': Dict of batched statistics (e.g., 'num_fg': (B,))
        }
    """
    B, num_tokens = sink_attn_batch.shape
    if num_tokens != grid_dim * grid_dim:
        raise ValueError(f"num_tokens ({num_tokens}) must match grid_dim*grid_dim ({grid_dim * grid_dim}).")
    device = sink_attn_batch.device

    # Ensure inputs are float for calculations involving them
    sink_attn_batch = sink_attn_batch.float()
    grounding_attn_o2i_batch = grounding_attn_o2i_batch.float()

    # 1. Calculate processed grounding_attn for Foreground (FG) identification
    # Remove sink tokens from grounding_attn_o2i influence
    sink_influence_mask = (sink_attn_batch < sink_thresh).float()
    grounding_attn_for_fg = grounding_attn_o2i_batch * sink_influence_mask

    # Binarize grounding_attn_for_fg
    # Get max per sample, keeping batch dim for broadcasting: (B, 1)
    grounding_attn_for_fg_max = torch.max(grounding_attn_for_fg, dim=1, keepdim=True)[0]
    # Avoid division by zero or issues if max is 0 by adding a small epsilon or handling zero max
    # For simplicity, if max is 0, all binary_mask will be 0.
    binary_threshold_fg = grounding_attn_for_fg_max * binary_coff
    binary_mask_fg = (grounding_attn_for_fg > binary_threshold_fg).float()

    grounding_attn_processed_fg = grounding_attn_for_fg * binary_mask_fg

    # 2. Identify sink tokens
    sink_token_mask_batch = (sink_attn_batch >= sink_thresh)  # (B, 576) boolean

    # 3. Identify foreground tokens
    fg_mask_batch = (grounding_attn_processed_fg > 0)  # (B, 576) boolean

    # 4. Get foreground bounding box from grounding_attn_processed_fg
    grounding_attn_2d_batch = grounding_attn_processed_fg.view(B, grid_dim, grid_dim)
    fg_bboxes_batch = get_foreground_bbox_torch(grounding_attn_2d_batch, threshold=1e-9)  # (B, 4)

    # 5. Create mask for tokens inside FG bounding box
    grid_indices_1d = torch.arange(num_tokens, device=device).unsqueeze(0)  # (1, 576)
    row_indices_grid = grid_indices_1d // grid_dim  # (1, 576)
    col_indices_grid = grid_indices_1d % grid_dim  # (1, 576)

    # Expand bbox dims for broadcasting: (B, 1)
    min_row_b = fg_bboxes_batch[:, 0].unsqueeze(1)
    min_col_b = fg_bboxes_batch[:, 1].unsqueeze(1)
    max_row_b = fg_bboxes_batch[:, 2].unsqueeze(1)
    max_col_b = fg_bboxes_batch[:, 3].unsqueeze(1)

    in_fg_box_mask_batch = (
        (row_indices_grid >= min_row_b) & (row_indices_grid <= max_row_b) &
        (col_indices_grid >= min_col_b) & (col_indices_grid <= max_col_b)
    )  # (B, 576) boolean

    # 6. Find candidate background (BG) tokens and select them
    # Tokens that are NOT in fg box. Sink token influence is handled in grounding_attn_o2i_for_bg_selection.
    bg_candidate_mask_batch = (~in_fg_box_mask_batch)  # (B, 576) boolean

    # Prepare grounding_attn_o2i for BG selection (original values, but with sinks zeroed out)
    grounding_attn_o2i_for_bg_selection = grounding_attn_o2i_batch * sink_influence_mask  # (B, 576)

    bg_mask_batch = torch.zeros_like(fg_mask_batch, dtype=torch.bool)  # (B, 576)

    # Threshold for BG selection is relative to the max of *processed* FG attention
    bg_selection_threshold_val = grounding_attn_for_fg_max * bg_coff  # (B, 1)

    for i in range(B):
        sample_bg_candidates_mask = bg_candidate_mask_batch[i]  # (576,)
        if sample_bg_candidates_mask.sum() > 0:
            candidate_indices_sample = torch.where(sample_bg_candidates_mask)[0]  # 1D tensor of indices

            candidate_grounding_values_sample = grounding_attn_o2i_for_bg_selection[
                i, candidate_indices_sample]  # Values for candidates

            # Select BG tokens if their original (sink-masked) grounding value is below threshold
            selected_by_thresh_mask_sample = (candidate_grounding_values_sample < bg_selection_threshold_val[i])

            selected_bg_indices_sample = candidate_indices_sample[selected_by_thresh_mask_sample]

            if selected_bg_indices_sample.numel() > 0:
                bg_mask_batch[i, selected_bg_indices_sample] = True

    # 7. Handle max_ratio_limit: if bbox too large, mark sample's FG and BG as empty
    fg_bbox_areas_batch = (fg_bboxes_batch[:, 2] - fg_bboxes_batch[:, 0] + 1) * \
                          (fg_bboxes_batch[:, 3] - fg_bboxes_batch[:, 1] + 1)  # (B,)

    invalid_ratio_batch_mask = (fg_bbox_areas_batch >= max_ratio_limit * num_tokens)  # (B,) boolean

    if invalid_ratio_batch_mask.any():
        # For samples with too large bbox, set their fg_mask and bg_mask to all False
        fg_mask_batch[invalid_ratio_batch_mask, :] = False
        bg_mask_batch[invalid_ratio_batch_mask, :] = False

    # 8. Create combined labels (-100=ignore, 0=bg, 1=fg)
    # Initialize with ignore value
    labels_batch = torch.full((B, num_tokens), -100, dtype=torch.long, device=device)
    labels_batch[bg_mask_batch] = 0  # Background (uses potentially modified bg_mask_batch)
    labels_batch[fg_mask_batch] = 1  # Foreground (uses potentially modified fg_mask_batch)

    # 9. All other tokens are ignored
    ignore_mask_batch = ~(fg_mask_batch | bg_mask_batch)  # (B, 576) boolean

    # 10. Collect statistics (batched tensors)
    stats_batch = {
        'num_fg': fg_mask_batch.sum(dim=1),
        'num_bg': bg_mask_batch.sum(dim=1),
        'num_ignore': ignore_mask_batch.sum(dim=1),
        'num_sink': sink_token_mask_batch.sum(dim=1),
        'num_bg_candidates_initially': bg_candidate_mask_batch.sum(dim=1),  # Before thresholding BG based on values
        'fg_bbox_area': fg_bbox_areas_batch,
        'failed_max_ratio_limit': invalid_ratio_batch_mask.long()  # 0 or 1
    }

    return {
        'fg_mask': fg_mask_batch,
        'bg_mask': bg_mask_batch,
        'ignore_mask': ignore_mask_batch,
        'labels': labels_batch,
        'fg_bbox': fg_bboxes_batch,  # (B, 4)
        'stats': stats_batch
    }

def get_singleturn_query_text_hs(
    hidden_states: torch.Tensor,
    labels: torch.Tensor
):
    is_response_token = (labels != -100)  # Shape: (batch_size, sequence_length)
    first_response_token_indices_all_samples = torch.argmax(is_response_token.int(), dim=1)
    query_indices = torch.max(
        torch.tensor(0, device=labels.device),
        first_response_token_indices_all_samples - 1
    )
    # Gather the hidden states for the selected query tokens
    batch_size_hs = hidden_states.size(0)
    hidden_dim = hidden_states.size(2)
    idx_expanded = query_indices.view(batch_size_hs, 1, 1).expand(batch_size_hs, 1, hidden_dim)
    query_hidden_states = torch.gather(hidden_states, 1, idx_expanded) #
    return query_hidden_states
def get_multiturn_query_text_hs_and_expanded_visuals(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    visual_tokens: torch.Tensor,
    lazy_sink_attention: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Selects the hidden state of the text token immediately preceding each answer turn
    in a multi-round conversation for every sample in a batch.
    For each selected text query token, it also provides the corresponding set of visual tokens,
    expanded (repeated) to match the number of query turns for that sample.

    Args:
        hidden_states (torch.Tensor): Shape (batch_size, sequence_length, hidden_size).
                                      These are the hidden states from the language model.
        labels (torch.Tensor): Shape (batch_size, sequence_length).
                               Prompt/human turns are -100, assistant/model answer turns have token IDs.
        visual_tokens (torch.Tensor): Shape (batch_size, num_visual_tokens, hidden_size).
                                      Let num_visual_tokens be N_v, hidden_size be D.
                                      These are the visual tokens for each sample in the batch.
        lazy_sink_attention (torch.Tensor, optional): If provided, used to filter out sink tokens.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]:
            - query_text_hs_list (list[torch.Tensor]):
                A list of length batch_size. Each tensor at index `b` corresponds to
                the b-th batch sample and has shape (M_b, D), where M_b is the
                number of identified answer turns for sample b. Contains the
                hidden states of the text tokens preceding each answer.
            - expanded_visual_tokens_list (list[torch.Tensor]):
                A list of length batch_size. Each tensor at index `b` corresponds to
                the b-th batch sample and has shape (M_b, N_v, D).
                For each of the M_b turns, it contains a copy of visual_tokens[b].
            - expanded_lazy_sink_attention (list[torch.Tensor], optional):
                A list of length batch_size. Each tensor at index `b` corresponds to
                the b-th batch sample and has shape (M_b, N_v).
                If lazy_sink_attention is provided, it contains the expanded lazy sink attention
                for each query turn.
    """
    batch_size, _, hidden_size_d = hidden_states.shape

    if visual_tokens.shape[0] != batch_size:
        raise ValueError("Batch size mismatch between hidden_states and visual_tokens.")
    if visual_tokens.shape[2] != hidden_size_d:
        raise ValueError("Hidden size mismatch between hidden_states/query tokens and visual_tokens.")

    num_visual_tokens_nv = visual_tokens.shape[1]

    all_query_text_hs_for_batch = []
    all_expanded_visual_tokens_for_batch = []
    all_expanded_lazy_sink_attention_for_batch = []

    for i in range(batch_size):
        sample_labels = labels[i]  # Shape: (sequence_length,)
        sample_hidden_states = hidden_states[i]  # Shape: (sequence_length, D)

        # Get the single set of visual tokens for this specific sample
        sample_visual_tokens_set = visual_tokens[i]  # Shape: (N_v, D)
        sample_sink_attention = lazy_sink_attention[i] # Shape: (N_v,)


        # --- Logic to find indices of text tokens preceding each answer turn ---
        # Shift labels to find where a prompt token (-100) is followed by a response token (not -100)
        prev_labels = torch.roll(sample_labels, shifts=1, dims=0)
        prev_labels[0] = -100  # Assume start of sequence is like a -100 predecessor

        is_current_token_response = (sample_labels != -100)
        was_prev_token_prompt_or_padding = (prev_labels == -100)

        # A new answer starts where the previous token was -100 and current is not -100
        start_of_answer_mask = is_current_token_response & was_prev_token_prompt_or_padding
        indices_of_first_answer_tokens = torch.where(start_of_answer_mask)[0]

        # The query token is the one *before* the first token of an answer.
        potential_query_indices = indices_of_first_answer_tokens - 1
        valid_query_indices_mask = (potential_query_indices >= 0)  # Ensure indices are not negative
        actual_query_indices = potential_query_indices[valid_query_indices_mask]

        num_query_turns_for_sample = actual_query_indices.numel()
        # --- End of index finding logic ---

        if num_query_turns_for_sample > 0:
            # Select the query text hidden states
            query_text_hs_for_sample = sample_hidden_states[actual_query_indices]  # Shape: (M_i, D)

            # Expand/repeat the single set of visual tokens for each query turn
            # sample_visual_tokens_set is (N_v, D)
            # We want to create a tensor of shape (M_i, N_v, D)
            expanded_visuals_for_sample = sample_visual_tokens_set.unsqueeze(0).expand(
                num_query_turns_for_sample, num_visual_tokens_nv, hidden_size_d
            )
            expanded_sink_attention_for_sample = sample_sink_attention.unsqueeze(0).expand(
                num_query_turns_for_sample, num_visual_tokens_nv
            ) # Shape: (M_i, N_v)
        else:
            # No valid query tokens found for this sample based on the turn detection logic
            query_text_hs_for_sample = torch.empty((0, hidden_size_d),
                                                   device=hidden_states.device,
                                                   dtype=hidden_states.dtype)
            expanded_visuals_for_sample = torch.empty((0, num_visual_tokens_nv, hidden_size_d),
                                                      device=visual_tokens.device,
                                                      dtype=visual_tokens.dtype)
            expanded_sink_attention_for_sample = torch.empty((0, num_visual_tokens_nv),
                                                              device=lazy_sink_attention.device,
                                                              dtype=lazy_sink_attention.dtype)
        all_query_text_hs_for_batch.append(query_text_hs_for_sample)
        all_expanded_visual_tokens_for_batch.append(expanded_visuals_for_sample)
        all_expanded_lazy_sink_attention_for_batch.append(expanded_sink_attention_for_sample)
    return all_query_text_hs_for_batch, all_expanded_visual_tokens_for_batch, all_expanded_lazy_sink_attention_for_batch


def aggregate_text2visual_attention_per_turn(
    labels: torch.Tensor,
    text_self_attention_weights: torch.Tensor,
    visual_token_mask_in_sequence: torch.Tensor
) -> list[torch.Tensor]:
    """
    Aggregates precomputed text self-attention weights, focusing on attention
    from LLM response tokens to visual tokens embedded within the same sequence.
    The aggregation (averaging) is done per QA turn.

    Args:
        labels (torch.Tensor): Labels with -100 for non-response parts.
                               Shape: (batch_size, seq_length).
        text_self_attention_weights (torch.Tensor):
                               Precomputed self-attention weights within the text/hybrid sequence.
                               Shape: (batch_size, num_attn_heads, seq_length, seq_length).
        visual_token_mask_in_sequence (torch.Tensor):
                               Boolean mask indicating positions of visual tokens in the sequence.
                               Shape: (batch_size, seq_length).

    Returns:
        list[torch.Tensor]: A list of tensors, one for each batch sample.
                            Each tensor has shape:
                            (num_qa_rounds_for_sample, num_attn_heads, num_visual_tokens_in_sample).
                            It contains the attention weights from response tokens (averaged)
                            to visual tokens (identified by the mask) for each head and QA turn.
                            Note: num_visual_tokens_in_sample can vary if not padded.
    """
    batch_size, full_seq_length = labels.shape
    _, num_attn_heads, attn_seq_length_q, attn_seq_length_kv = text_self_attention_weights.shape

    if not (full_seq_length == attn_seq_length_q == attn_seq_length_kv):
        raise ValueError(
            f"Sequence length mismatch: labels ({full_seq_length}), "
            f"attn_q ({attn_seq_length_q}), attn_kv ({attn_seq_length_kv}). Must all be equal."
        )
    if visual_token_mask_in_sequence.shape != (batch_size, full_seq_length):
        raise ValueError(
            f"Shape mismatch for visual_token_mask_in_sequence. Expected {(batch_size, full_seq_length)}, "
            f"got {visual_token_mask_in_sequence.shape}"
        )

    output_list_for_batch = []

    for b_idx in range(batch_size):
        sample_labels = labels[b_idx]  # (seq_length,)
        # Attentions for this sample: (num_attn_heads, seq_length, seq_length)
        sample_self_attentions = text_self_attention_weights[b_idx]
        # Visual token mask for this sample: (seq_length,)
        sample_visual_token_mask = visual_token_mask_in_sequence[b_idx]

        # Get the indices of visual tokens (these are the columns we care about in the attention matrix)
        visual_token_indices_in_seq = torch.where(sample_visual_token_mask)[0]
        num_visual_tokens_in_this_sample = visual_token_indices_in_seq.numel()

        if num_visual_tokens_in_this_sample == 0:  # No visual tokens identified for this sample
            # Append empty tensor with expected number of dimensions if no visual tokens
            # Or handle as an error, or skip sample, depending on desired behavior.
            output_list_for_batch.append(
                torch.empty((0, num_attn_heads, 0),  # 0 visual tokens
                            device=text_self_attention_weights.device,
                            dtype=text_self_attention_weights.dtype)
            )
            continue

        # --- Identify response segments (QA turns) using labels ---
        is_response_token = (sample_labels != -100)
        is_response_padded = torch.cat([
            torch.tensor([False], device=is_response_token.device), is_response_token,
            torch.tensor([False], device=is_response_token.device)
        ])
        diffs = is_response_padded.to(torch.int8).diff()
        turn_start_indices = torch.where(diffs == 1)[0]
        turn_end_indices = torch.where(diffs == -1)[0]

        if turn_start_indices.numel() == 0:  # No response turns in this sample
            output_list_for_batch.append(
                torch.empty((0, num_attn_heads, num_visual_tokens_in_this_sample),
                            device=text_self_attention_weights.device,
                            dtype=text_self_attention_weights.dtype)
            )
            continue

        all_turn_avg_attns_for_sample = []
        for turn_idx in range(turn_start_indices.numel()):
            start_resp_seq_idx = turn_start_indices[turn_idx]
            if sample_labels[turn_end_indices[turn_idx]-1] == 2:  # EOS token
                start_resp_seq_idx = start_resp_seq_idx
                end_resp_seq_idx = turn_end_indices[turn_idx] - 1 #remove EOS, important!!
            else:
                end_resp_seq_idx = turn_end_indices[turn_idx] #condition of truncated sequence over max limit

            if start_resp_seq_idx >= end_resp_seq_idx: continue

            # Get attention weights *from* the tokens in the current response turn
            # sample_self_attentions shape: (num_attn_heads, seq_length, seq_length)
            # We want rows corresponding to response tokens:
            attns_from_current_turn_tokens = sample_self_attentions[:, start_resp_seq_idx:end_resp_seq_idx, :]
            # Shape: (num_attn_heads, len_of_current_response_turn, full_seq_length_kv)

            # Now, from these attentions, select only the weights *to* the visual tokens
            # We index the last dimension (keys/values) using visual_token_indices_in_seq
            attns_from_resp_to_visual = attns_from_current_turn_tokens[:, :, visual_token_indices_in_seq]
            # Shape: (num_attn_heads, len_of_current_response_turn, num_visual_tokens_in_this_sample)

            len_of_current_response_turn = attns_from_resp_to_visual.shape[1]
            if len_of_current_response_turn == 0: continue

            # Average attention weights across the response sequence length (dim=1)
            avg_turn_attn_weights = attns_from_resp_to_visual.mean(dim=1)
            # Shape: (num_attn_heads, num_visual_tokens_in_this_sample)

            all_turn_avg_attns_for_sample.append(avg_turn_attn_weights)

        if not all_turn_avg_attns_for_sample:
            output_list_for_batch.append(
                torch.empty((0, num_attn_heads, num_visual_tokens_in_this_sample),
                            device=text_self_attention_weights.device,
                            dtype=text_self_attention_weights.dtype)
            )
        else:
            sample_output = torch.stack(all_turn_avg_attns_for_sample, dim=0)
            # Shape: (num_qa_rounds_for_sample, num_attn_heads, num_visual_tokens_in_this_sample)
            output_list_for_batch.append(sample_output)

    return output_list_for_batch


from llava.ana_utils import get_bbox4src, get_bbox_from_noisy_map, plot_bboxes_on_img
import torchvision.transforms.functional as torchvision_F
from llava.ana_utils import plot_attention_analysis
def get_batched_sub_images(pred_roi, src_imgs, image_processor, image_encoder, conf_thresh=0.15,
    upscale_method = "bbox",
):
    batch_image_emd = []
    sub_img_nums, sub_img_bboxes, sub_img_pos_id = [], [], []
    default_config = {'image_aspect_ratio': 'pad'}
    encoder_img_size = (336, 336)
    img_downsample_ratio = 14
    mask_bbox_num_thresh = 2
    blurred_map = None
    for batch_id in range(pred_roi.shape[0]):
        image = src_imgs[batch_id]

        if conf_thresh<0:
            #print("Using negative conf_thresh to indicate noisy map")
            bbox_feat = get_bbox_from_noisy_map(pred_roi[batch_id].detach(), conf_thresh=conf_thresh, min_area=4, max_area=15 * 15)
        else:
            bbox_feat = get_bbox_from_noisy_map(pred_roi[batch_id].sigmoid().detach(), conf_thresh=conf_thresh, min_area=4,max_area=15 * 15)

        if upscale_method == 'bbox' or mask_bbox_num_thresh>bbox_feat.shape[0]:
            bbox_roi = get_bbox4src(bbox_feat, image, feat_size=(24, 24), img_size=encoder_img_size)
            if bbox_roi.shape[0] == 0 or bbox_roi.shape[0]>2:
                sub_img_nums.append(0)
                continue
            sub_img_bboxes.append(bbox_feat)
            bbox_roi_imgs = []
            for bbox_id in range(bbox_roi.shape[0]):
                bbox = bbox_roi[bbox_id]
                bbox_img = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                bbox_img = bbox_img.resize(encoder_img_size, Image.BILINEAR)
                bbox_roi_imgs.append(bbox_img)

            image_tensor = process_images(bbox_roi_imgs, image_processor, default_config).type_as(pred_roi)
            sub_img_nums.append(len(bbox_roi_imgs))
            img_feat = image_encoder(image_tensor)
            batch_image_emd.append(img_feat)

        elif upscale_method == 'mask':
            roi_mask = torchvision_F.gaussian_blur(pred_roi[batch_id].sigmoid().unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1.0).squeeze().squeeze()
            known_fg_mask, fg_bbox = get_foreground_mask(image.size, 24, min_dimension_size=1, return_bbox=True)
            roi_mask = (roi_mask > conf_thresh).float()
            if roi_mask.sum() == 0:
                sub_img_nums.append(0)
                continue

            # get the bbox that enclose the fg part of the roi_mask
            surrounding_bbox = get_foreground_bbox_torch(roi_mask.unsqueeze(0), threshold=0.5)
            bbox_roi = get_bbox4src(surrounding_bbox, image, feat_size=(24, 24), img_size=encoder_img_size)[0]
            bbox_img = image.crop((int(bbox_roi[0]), int(bbox_roi[1]), int(bbox_roi[2]), int(bbox_roi[3])))
            roi_mask = roi_mask[surrounding_bbox[0,1]:surrounding_bbox[0,3], surrounding_bbox[0,0]:surrounding_bbox[0,2]]

            bbox_img_longer = 336  # (max(bbox_img.size) * limited_upscale_ratio // 16 * 16)
            bbox_img = bbox_img.resize((bbox_img_longer, bbox_img_longer), Image.BILINEAR)
            valid_roi_mask_size = bbox_img_longer // img_downsample_ratio
            valid_roi_mask = F.interpolate(roi_mask.unsqueeze(0).unsqueeze(0), size=(valid_roi_mask_size, valid_roi_mask_size), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            valid_roi_mask = valid_roi_mask > 0.5
            image_tensor = process_images([bbox_img], image_processor, default_config).type_as(pred_roi)
            images_embeds = image_encoder(image_tensor)
            images_embeds = images_embeds[:, valid_roi_mask.flatten()]
            sub_img_nums.append(1)
            sub_img_bboxes.append(torch.tensor([fg_bbox[0], fg_bbox[1], fg_bbox[2], fg_bbox[3]], device=pred_roi.device).unsqueeze(0))
            batch_image_emd.append(images_embeds)

        blurred_map = torchvision_F.gaussian_blur(pred_roi[0].unsqueeze(0).unsqueeze(0).sigmoid(), kernel_size=3, sigma=1.0).squeeze()

    if len(batch_image_emd) == 0:
        return None, sub_img_nums, sub_img_bboxes, sub_img_pos_id, bbox_roi, blurred_map
    batch_image_embed = torch.cat(batch_image_emd, dim=0)
    return batch_image_embed, sub_img_nums, sub_img_bboxes, sub_img_pos_id, bbox_roi, blurred_map


def interplot_img_feat(sub_img_feat, sub_img_bboxes, max_ratio=2.0, upscale_method = 'bbox', sub_img_pos_id=[]):
    """
    Interpolate image features to match the bounding boxes of sub-images.

    This function takes a batch of feature maps and corresponding bounding boxes.
    It expands each bounding box by `max_ratio`, clips it to the feature map
    boundaries, and then uses RoIAlign to extract and resize the feature
    region within the new box back to the original feature map's dimensions.

    Args:
        sub_img_feat (torch.Tensor): Image features of shape (batch_size, num_features, height, width).
                                     If features are flattened (B, C, H*W), they will be reshaped.
        sub_img_bboxes (list of torch.Tensor or torch.Tensor): A list of bounding boxes for each feature map,
                                                               or a single tensor of shape (batch_size, 4).
                                                               Bboxes are in [x1, y1, x2, y2] format.
        max_ratio (float): The factor by which to expand the bounding boxes.

    Returns:
        torch.Tensor: Interpolated image features of the same shape as the input feature map
                      (batch_size, num_features, height, width).
    """
    # --- Input Validation and Reshaping ---
    if sub_img_feat.dim() == 3:
        # Handle flattened features like (B, C, 24*24)
        batch_size, num_features, dim = sub_img_feat.shape
        # Assume a square feature map if it's flattened
        height = width = int(math.sqrt(num_features))
        if height * width != num_features and upscale_method=='bbox':
            raise ValueError(f"Cannot reshape flattened features of dim {num_features} to a square map.")
        if upscale_method=='bbox':
            sub_img_feat = sub_img_feat.view(batch_size, height, width, dim).permute(0, 3, 1, 2)
    elif sub_img_feat.dim() != 4:
        raise ValueError(f"sub_img_feat must have 3 or 4 dimensions, but got {sub_img_feat.dim()}")
    if upscale_method=='bbox': batch_size, dim, height, width = sub_img_feat.shape
    else: batch_size, seq_len, dim = sub_img_feat.shape
    # --- Step 1: Process and Validate Bounding Boxes ---
    bboxes = torch.cat(sub_img_bboxes, dim=0) if isinstance(sub_img_bboxes, list) else sub_img_bboxes

    if bboxes.shape[0] != batch_size:
        raise ValueError(f"Number of bboxes ({bboxes.shape[0]}) must match batch size ({batch_size}).")
    if bboxes.dim() != 2 or bboxes.shape[1] != 4:
        raise ValueError("Bboxes tensor must have shape (batch_size, 4).")

    # Ensure bboxes are on the correct device and are float type for calculations
    bboxes = bboxes.to(sub_img_feat.device)

    # --- Process each feature map individually ---
    output_features, output_sub_img_pos = [], []
    for i in range(batch_size):
        # Isolate the i-th feature map and its bbox
        feature_map_i = sub_img_feat[i:i + 1]  # Keep batch dim for interpolate
        if sub_img_pos_id: pos_id_i = sub_img_pos_id[i]

        if max_ratio != -1 and upscale_method=='bbox':
            bbox_i = bboxes[i]

            # 1. Calculate original bbox dimensions
            bbox_w = bbox_i[2] - bbox_i[0]
            bbox_h = bbox_i[3] - bbox_i[1]

            # 2. Calculate the initial target size by expanding the bbox
            target_w = bbox_w * max_ratio
            target_h = bbox_h * max_ratio

            # 3. Apply the constraint: if expanded size is larger than the base feature map, scale it down.
            longest_target_side = max(target_w, target_h)
            longest_feat_side = max(height, width)

            if longest_target_side > longest_feat_side:
                scale_down_ratio = longest_feat_side / longest_target_side
                target_w *= scale_down_ratio
                target_h *= scale_down_ratio

            # 4. Perform the resizing using interpolation
            # Target size must be integers
            final_target_size = (int(round(target_h.item())), int(round(target_w.item())))

            # Ensure the target size is at least 1x1
            final_target_size = (max(1, final_target_size[0]), max(1, final_target_size[1]))

            resized_feat = F.interpolate(
                feature_map_i,  # Add a temporary batch dimension
                size=final_target_size,
                mode='bilinear',
                align_corners=False
            ) # 1, C, final_target_size[0], final_target_size[1]

            # Squeeze the temporary batch dimension before appending
            resized_feat = resized_feat.squeeze(0).flatten(1).transpose(0, 1)
            assert resized_feat.shape[0]<=576, f"Resized feature shape {resized_feat.shape} exceeds 576 tokens limit."
            if sub_img_pos_id:
                pos_id_i = F.interpolate(
                    pos_id_i.unsqueeze(0).unsqueeze(0).float(),
                    size=final_target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
        else:
            if upscale_method == 'bbox':
                resized_feat = feature_map_i.squeeze(0).flatten(1).transpose(0, 1)
            else:
                resized_feat = feature_map_i.squeeze(0)
        output_features.append(resized_feat)  # H*W, C

        if sub_img_pos_id:
            output_sub_img_pos.append(pos_id_i.flatten().long())


    # --- NEW: Padding Logic ---
    # 1. Find the maximum sequence length in this batch
    max_len = max(feat.shape[0] for feat in output_features) #hard-code to avoid multi-gpu sync problem #max(feat.shape[0] for feat in output_features)

    # 2. Create the padded output tensor and the attention mask
    padded_features = torch.full(
        (batch_size, max_len, dim),
        fill_value=0,
        dtype=sub_img_feat.dtype,
        device=sub_img_feat.device
    )
    padding_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
        device=sub_img_feat.device
    )

    # 3. Copy the data from the unpadded list into the new padded tensor
    for i, feat in enumerate(output_features):
        seq_len = feat.shape[0]
        padded_features[i, :seq_len, :] = feat
        padding_mask[i, :seq_len] = True


    return padded_features, padding_mask, output_sub_img_pos

def insert_sub_feat(ori_feat, sub_feat, sub_padding, sub_img_num, visual_token_num, sys_token_num,
                    new_labels, attention_mask, position_ids, padding_side = 'right', IGNORE_INDEX=-100, sub_img_pos_id=[]):
    #TODO: deal with rope pos emd
    batch_size = ori_feat.shape[0]
    token_num_rec, dst_feat, dst_label = [], [], []
    sub_img_id = 0
    sub_feat_list = []
    for tmp_id in range(sub_feat.shape[0]):
        tmp_feat = sub_feat[tmp_id][sub_padding[tmp_id]]  # L_i, C
        sub_feat_list.append(tmp_feat)
    sub_feat = sub_feat_list

    for i in range(batch_size):
        ori_feat_i = ori_feat[i] #L, C
        if sub_img_num[i] == 0:
            dst_feat.append(ori_feat_i)
            token_num_rec.append(ori_feat_i.shape[0])
            dst_label.append(new_labels[i] if new_labels is not None else None)
            continue
        sub_feat_i = sub_feat[sub_img_id:sub_img_id + sub_img_num[i]]
        sub_feat_i = torch.cat(sub_feat_i, dim=0) #L_i, C
        dst_feat_i = torch.cat(
            [ori_feat_i[:sys_token_num+visual_token_num], sub_feat_i, ori_feat_i[sys_token_num+visual_token_num:]],
            dim=0
        )
        token_num_rec.append(dst_feat_i.shape[0])
        dst_feat.append(dst_feat_i)
        sub_img_id += sub_img_num[i]

    max_len = max(token_num_rec)

    try:
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    except:
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=position_ids.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = dst_feat
    if new_labels is None:
        #inference mode, return other values
        attention_mask = attention_mask + attention_mask.new_ones((batch_size, max_len), dtype=torch.bool)
        #FIXME: only work for bs=1
        if sub_img_pos_id:
            sub_img_pos_id = torch.cat(sub_img_pos_id, dim=0) if isinstance(sub_img_pos_id, list) else sub_img_pos_id
            position_ids = torch.cat([position_ids[0][:sys_token_num+visual_token_num], sub_img_pos_id,position_ids[0][sys_token_num+visual_token_num:]], dim=0).unsqueeze(0)
        else:
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = position_ids + torch.arange(0, max_len, dtype=position_ids.dtype, device=position_ids.device)
        return torch.stack(new_input_embeds, dim=0), None, attention_mask, position_ids

