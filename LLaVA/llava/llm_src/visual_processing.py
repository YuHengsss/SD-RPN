import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt


#src from https://github.com/seilk/VisAttnSink/blob/98e62d6ca74cf721daede708146d73d400ef501a/src/logic/logic.py#L50
DIM_SINK = {
    "llama-v2-7b": torch.tensor([2533, 1415]),
    "llama-v2-13b": torch.tensor([2100, 4743]),
}
def process_hidden_states_and_find_indices(hs, dim_sink=torch.tensor([2533, 1415]), tau=20, eps=1e-6, sink=False):
    """
    Performs RMS normalization on hidden states and then identifies token indices
    based on the normalized values, specific dimensions (dim_sink), and a threshold (tau).

    Args:
        hs (torch.Tensor): The input hidden states. Expected shape [bsz, tok, dim].
        dim_sink: A list or tuple of dimension indices to
                                         monitor within the hidden states after normalization.
        tau (float): The threshold value. Tokens where the max RMS value across
                     dim_sink dimensions exceeds this tau are selected.
        eps (float, optional): Epsilon value for RMS normalization to prevent
                               division by zero. Defaults to 1e-6.

    Returns:
        tuple: A tuple containing:
            - normalized_hidden_states (torch.Tensor): The hidden states after RMS normalization.
            - token_indices (torch.Tensor): A 1D tensor of token indices that met the criteria.
                                            Shape: [num_selected_tokens].
    """
    # 1. RMS Normalization (derived from the original rmsnorm method)
    hidden_states_float32 = hs.to(torch.float32)
    variance = hidden_states_float32.pow(2).mean(-1, keepdim=True)
    normalized_hidden_states = hidden_states_float32 * torch.rsqrt(variance + eps)

    # 2. Logic to find indices (derived from the original run_logic method)
    abs_normalized_hs = torch.abs(normalized_hidden_states)  # Expected shape: [bsz, tok, dim]
    token_indices = torch.tensor([], dtype=torch.long, device=hs.device) # Default to empty

    if dim_sink is None:
        # If dim_sink is empty, no dimensions to check.
        pass # token_indices remains empty
    elif abs_normalized_hs.ndim != 3:
        print(f"Warning: Expected hidden_states to be 3D [bsz, tok, dim], but got {abs_normalized_hs.ndim}D. "
              "Slicing for dim_sink might behave unexpectedly or error.")
        # token_indices remains empty or raise error
    else:
        try:
            # Select values from the specified dim_sink dimensions
            # abs_normalized_hs shape: [bsz, tok, dim]
            # dim_sink is a list of indices for the last dimension
            # rms_values_selected will have shape: [bsz, tok, len(dim_sink)]
            rms_values_selected = abs_normalized_hs[:, :, dim_sink]
        except IndexError as e:
            raise IndexError(
                f"Error slicing hidden states with dim_sink. Ensure all indices in dim_sink "
                f"(values: {dim_sink}) are valid for the last dimension of hidden_states "
                f"(size: {abs_normalized_hs.shape[2]}). Original error: {e}"
            )

        if rms_values_selected.numel() > 0:
            max_rms_values_across_sinks = torch.max(rms_values_selected, dim=-1)[0]  # Shape: [bsz, tok]

            # Find coordinates (batch_idx, token_idx) where condition is met
            if sink:
                condition_met_coords = torch.nonzero(max_rms_values_across_sinks > tau)
            else:
                condition_met_coords = torch.nonzero(max_rms_values_across_sinks < tau)

            if condition_met_coords.numel() > 0:
                # Extract only the token indices (the second column of the coordinates)
                token_indices = condition_met_coords[:, 1]
        # else: if rms_values_selected is empty (e.g. if len(dim_sink) was 0 after all), token_indices remains empty

    return token_indices


def find_indices_based_on_attn(attn_mat, average_head=True, threshold=1e-3, sink=False):
    """
    Find token indices based on attention matrix values.

    Args:
        attn_mat (torch.Tensor): The attention matrix. Expected shape [bsz, num_heads, tok, tok].
        average_head (bool): Whether to average across heads.
        threshold (float): The threshold value for selecting token indices.
        sink (bool): If True, select tokens with values below the threshold.

    Returns:
        torch.Tensor: A 1D tensor of token indices that met the criteria.
    """
    # Compute the mean across heads if required
    if average_head:
        attn_mat = attn_mat.mean(dim=1).mean(dim=1)  # Shape: [bsz, tok]

    # Apply thresholding
    if sink:
        attn_mat = attn_mat > threshold
    else:
        attn_mat = attn_mat < threshold

    # Find indices where condition is met
    token_indices = torch.nonzero(attn_mat)[:, 1]

    return token_indices

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        alpha: Weighting factor for the positive class (class 1).
               If positive:negative is 1:10, and positive is minority,
               you might want alpha to be higher for positive class, e.g., around 10/11 = 0.9.
               Or, if alpha is the weight for the class being focused on (positive class),
               and it's rare, a common setting from papers is alpha=0.25 for the rare class
               if the loss formulation is -alpha*(1-pt)^gamma*log(pt), and alpha weights the
               contribution of that class. Let's assume alpha weights the positive class.
               Given your 1:10 positive:negative ratio, if positive is class 1,
               setting alpha to ~0.9 might be appropriate to upweight the positive class.
               Let's make alpha tunable, defaulting to a common paper value, but you should adjust.
        gamma: Focusing parameter.
        reduction: 'none', 'mean', 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Raw, unnormalized scores (before sigmoid)
        targets: Binary target labels (0 or 1)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Calculate p_t
        p = torch.sigmoid(logits)
        # p_t = p if targets == 1 else 1 - p
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate focal loss modulator: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Calculate alpha_t: alpha if targets == 1 else 1 - alpha
        # This assumes self.alpha is the weight for the positive class (targets == 1)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * modulating_factor * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
