import  torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image
import torch.nn.functional as F
from skimage.measure import label
from scipy.ndimage import binary_closing

def register_hooks(model):
    """Register hooks at all stages of processing."""
    activations = {}
    hooks = []

    for i in range(len(model.model.layers)):
        layer = model.model.layers[i]
        def make_attn_hook(layer_idx):
            def hook(module, inputs, outputs):
                # This assumes outputs has attention weights as second element in tuple
                # Check output structure to confirm this works with your model
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attn_weights = outputs[1]  # Typically the attention weights

                    attn_key = f'layer{layer_idx}_attention_weights'
                    if attn_key not in activations:
                        activations[attn_key] = []
                    if attn_weights is None:
                        return outputs
                    activations[attn_key].append(attn_weights.detach())

                    norm_key = f'layer{layer_idx}_feature_norm'
                    #L2 normalization value of the input features

                    feature_norm = outputs[0].pow(2).max(-1, keepdim=False)[0]
                    if norm_key not in activations:
                        activations[norm_key] = []
                    activations[norm_key].append(feature_norm.detach())

                return outputs

            return hook

        h = layer.self_attn.register_forward_hook(make_attn_hook(i))
        hooks.append(h)

    if hasattr(model.model, 'enable_twig') and model.model.enable_twig:
        for i in range(len(model.model.twig_layers)):
            layer = model.model.twig_layers[i]
            def make_twig_attn_hook(layer_idx):
                def hook(module, inputs, outputs):
                    if isinstance(outputs, tuple) and len(outputs) > 1:
                        attn_weights = outputs[1]
                        attn_key = f'layer{layer_idx+model.model.twig_K}_attention_weights'
                        if attn_key not in activations:
                            activations[attn_key] = []
                        activations[attn_key].append(attn_weights.detach())
                    return outputs
                return hook
            h = layer.self_attn.register_forward_hook(make_twig_attn_hook(i))
            hooks.append(h)
    return activations, hooks


def get_attn_weights_map(activations, layers=32, reduction='mean', required_layers = None):
    """
    Extract attention weights from activations for visualization.

    Args:
        activations: Dictionary containing model activations
        layers: Number of layers to analyze

    Returns:
        Dictionary containing attention weights between output token and prompt tokens for each layer
    """
    attn_weights = {}
    visual_mask = activations.get('visual_mask', None)
    #find the first non-zero in viusal_mask
    system_end_idx = visual_mask.nonzero(as_tuple=True)[1][0].item()
    image_end_idx = system_end_idx + visual_mask[0].sum().item()
    ins_end_idx = visual_mask.shape[1]
    output_start_idx = visual_mask.shape[1]

    # Process each layer
    for i in range(layers):
        if required_layers is None:pass
        elif i not in required_layers: continue
        layer_key = f'layer{i}_attention_weights'
        tmp_attn_weights = activations.get(layer_key, None)

        if tmp_attn_weights is None or not tmp_attn_weights:
            continue

        # Initialize layer dictionary
        attn_weights[f'layer{i}'] = {
            'output2system': [],
            'output2image': [],
            'output2instruction': [],
            'output2output': [],
            'image2system': [],
            'image2image': [],
            'imagetokens2system': [],
            'imagetokens2systems': [],
            'output2images': [],
            'question2images': []
        }

        # Process each chunk of attention weights (for prefill and generation steps)
        for j in range(len(tmp_attn_weights)):
            weights = tmp_attn_weights[j].to(torch.float16)
            batch_size, num_heads, _, seq_len = weights.shape

            if j==0:
                #compute attn weights between image to others
                image2system = weights[:, :, system_end_idx:image_end_idx, 0:system_end_idx].sum(dim=(0, 3))#.mean(dim=-1)  # [num_heads]
                image2systems = weights[:, :, system_end_idx:image_end_idx, 0:system_end_idx].sum(dim=(0)) # [num_heads, imge_tokens, system_tokens]
                attn_weights[f'layer{i}']['imagetokens2systems'].append(image2systems.mean(dim=0).cpu().numpy()) #[num_heads, system_tokens]
                attn_weights[f'layer{i}']['image2system'].append(image2system.mean(dim=-1).cpu().numpy())
                attn_weights[f'layer{i}']['imagetokens2system'].append(image2system.mean(dim=0).cpu().numpy())
                image2image = weights[:, :, system_end_idx:image_end_idx, system_end_idx:image_end_idx].sum(dim=(0, 3)).mean(dim=-1)  # [num_heads]
                attn_weights[f'layer{i}']['image2image'].append(image2image.cpu().numpy())
                question2images = weights[:, :, image_end_idx:, system_end_idx:image_end_idx].sum(dim=(0))  # [num_heads, question_tokens, image_tokens]
                attn_weights[f'layer{i}']['question2images'].append(question2images[:,-1].cpu().numpy()) # [:,-1].cpu().numpy()  .mean(dim=1).cpu().numpy() # [num_heads, image_tokens]
                continue
            # Output to system prompt
            output2system = weights[:, :, 0, :system_end_idx].sum(dim=(0, 2))  # [num_heads]
            attn_weights[f'layer{i}']['output2system'].append(output2system.cpu().numpy())

            # Output to image tokens
            output2image = weights[:, :, 0, system_end_idx:image_end_idx].sum(dim=(0, 2))  # [num_heads]
            attn_weights[f'layer{i}']['output2image'].append(output2image.cpu().numpy())
            output2images = weights[:, :, 0, system_end_idx:image_end_idx].sum(dim=(0))  # [num_heads, image_tokens]
            attn_weights[f'layer{i}']['output2images'].append(output2images.cpu().numpy())

            # Output to instruction tokens
            output2ins = weights[:, :, 0, image_end_idx:ins_end_idx].sum(dim=(0, 2))  # [num_heads]
            attn_weights[f'layer{i}']['output2instruction'].append(output2ins.cpu().numpy())

            # Output to previous output tokens
            output2output = weights[:, :, 0, output_start_idx:].sum(dim=(0, 2))  # [num_heads]
            attn_weights[f'layer{i}']['output2output'].append(output2output.cpu().numpy())

    for i in range(layers):
        if reduction == 'mean':
            if f'layer{i}' in attn_weights:
                attn_weights[f'layer{i}']['output2images'] = np.mean(np.array(attn_weights[f'layer{i}']['output2images']), axis=0) #attn_weights[f'layer{i}']['output2images'][-1] #
                attn_weights[f'layer{i}']['output2image']  = [np.mean(np.array(attn_weights[f'layer{i}']['output2image']), axis=0)]
        elif reduction == 'max':
            if f'layer{i}' in attn_weights:
                attn_weights[f'layer{i}']['output2images'] = np.max(np.array(attn_weights[f'layer{i}']['output2images']), axis=0)
                attn_weights[f'layer{i}']['output2image'] = [np.max(np.array(attn_weights[f'layer{i}']['output2image']), axis=0)]
        elif reduction == 'none':
            if f'layer{i}' in attn_weights:
                attn_weights[f'layer{i}']['output2images'] = np.array(attn_weights[f'layer{i}']['output2images'])
                attn_weights[f'layer{i}']['output2image'] = np.array(attn_weights[f'layer{i}']['output2image'])
    return attn_weights


def get_visual_feat_norm(activations, layers=32, required_layers=None):
    """
    Extract feature norms of visual features from activations for visualization.

    Args:
        activations: Dictionary containing model activations
        layers: Number of layers to analyze

    Returns:
        Dictionary containing feature norms for each layer
    """
    feat_norms = {}
    visual_mask = activations.get('visual_mask', None)
    # find the first non-zero in viusal_mask
    system_end_idx = visual_mask.nonzero(as_tuple=True)[1][0].item()
    image_end_idx = system_end_idx + visual_mask[0].sum().item()
    ins_end_idx = visual_mask.shape[1]
    output_start_idx = visual_mask.shape[1]
    # Process each layer
    for i in range(layers):
        if required_layers is None: pass
        elif i not in required_layers: continue
        layer_key = f'layer{i}_feature_norm'
        tmp_feat_norms = activations.get(layer_key, None)

        if tmp_feat_norms is None or not tmp_feat_norms:
            continue

        # Initialize layer dictionary
        feat_norms[f'layer{i}'] = {
            'image': []
        }

        # Process each chunk of feature norms (for prefill and generation steps)
        for j in range(len(tmp_feat_norms)):
            feat_norm = tmp_feat_norms[j].to(torch.float16)

            if j == 0:
                # compute feature norms for image tokens
                image_feat_norm = feat_norm[:, system_end_idx:image_end_idx]
                feat_norms[f'layer{i}']['image'].append(image_feat_norm.cpu().numpy())
    return feat_norms


def visualize_norm_values(feat_norms, layers=32, save_path=None, title_prefix="", attn_size=None):
    """
    Visualize L2 norm of visual features for each token as 2D maps, layer by layer.

    Args:
        feat_norms (dict): Dictionary from get_visual_feat_norm.
                           Expected structure: feat_norms[f'layer{i}']['image']
                           contains a list with a numpy array of shape [batch_size, num_image_tokens].
        layers (int): Total number of layers to visualize.
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
        title_prefix (str, optional): Prefix for plot titles.
        attn_size (tuple, optional): The (height, width) of the feature map. If None, it's inferred.
    """
    norms_found = False
    num_image_tokens = 0

    # Find the first valid layer to get dimensions
    for i in range(layers):
        layer_key = f'layer{i}'
        if layer_key in feat_norms and 'image' in feat_norms[layer_key] and feat_norms[layer_key]['image']:
            potential_norms = feat_norms[layer_key]['image'][0]
            if isinstance(potential_norms, np.ndarray) and potential_norms.ndim == 2:
                if potential_norms.shape[1] > 0:
                    num_image_tokens = potential_norms.shape[1]
                    norms_found = True
                    break

    if not norms_found:
        print(f"Warning: No valid feature norms found for visualization in layers 0-{layers-1}.")
        return

    # Calculate grid size for the 2D map
    if attn_size is not None:
        grid_size = attn_size
    else:
        grid_size_f = math.sqrt(num_image_tokens)
        if grid_size_f != int(grid_size_f):
            raise ValueError(f"Number of image tokens ({num_image_tokens}) is not a perfect square. "
                             f"Cannot reshape into a square 2D map.")
        grid_size = (int(grid_size_f), int(grid_size_f))

    # Determine subplot layout
    cols = int(math.ceil(math.sqrt(layers)))
    rows = int(math.ceil(layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    axes = axes.flatten()

    # Find global min/max for consistent color scaling
    min_val, max_val = float('inf'), float('-inf')
    processed_norms = {}
    valid_layer_indices = []

    for i in range(layers):
        layer_key = f'layer{i}'
        if layer_key in feat_norms and 'image' in feat_norms[layer_key] and feat_norms[layer_key]['image']:
            raw_norms = feat_norms[layer_key]['image'][0] # Take first item, assuming prefill
            
            if not (isinstance(raw_norms, np.ndarray) and raw_norms.ndim == 2 and raw_norms.shape[1] == num_image_tokens):
                print(f"Warning: Skipping layer {i} due to incompatible norm format.")
                continue

            avg_norms = raw_norms.mean(axis=0) # Shape: [num_image_tokens]
            
            if avg_norms.size == 0:
                continue
                
            processed_norms[i] = avg_norms
            min_val = min(min_val, avg_norms.min())
            max_val = max(max_val, avg_norms.max())
            valid_layer_indices.append(i)

    if not processed_norms:
        print("Error: No valid data to plot after processing norms.")
        plt.close(fig)
        return

    # --- Plotting Loop ---
    im = None
    for i in range(layers):
        ax = axes[i]
        
        if i in valid_layer_indices:
            map_2d = processed_norms[i].reshape(grid_size)
            
            current_vmin = min_val
            current_vmax = max(max_val, min_val + 1e-9)
            
            im = ax.imshow(map_2d, cmap='viridis', vmin=current_vmin, vmax=current_vmax, aspect='equal')
            ax.set_title(f"Layer {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    # Remove unused subplots
    for i in range(layers, len(axes)):
         fig.delaxes(axes[i])

    # Adjust layout and add colorbar
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='L2 Norm')
        
    fig.suptitle(f"{title_prefix}Visual Feature L2 Norm Across Layers ({grid_size[0]}x{grid_size[1]} map)", fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def plot_attention_across_layers(attn_weights, layers=32, save_path=None, title_prefix="", avg_heads=True):
    """
    Plot attention weights distribution across layers.

    Args:
        attn_weights: Dictionary from get_attn_weights_map containing attention weights
        layers: Number of layers to analyze
        save_path: Path to save the visualization (if None, will display)
        title_prefix: Prefix for plot titles
        avg_heads: Whether to average across attention heads

    Returns:
        None (displays or saves plots)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # Define custom colormap
    colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c", "#a50034"]
    cmap = LinearSegmentedColormap.from_list("attention_cmap", colors)
    seg_keys = ['output2system', 'output2image', 'output2instruction', 'output2output', 'image2system', 'image2image']
    valid_layers = [i for i in range(layers) if f'layer{i}' in attn_weights]
    if not valid_layers:
        print("No valid attention weights found")
        return

    # Get number of heads from the data
    for layer_idx in valid_layers:
        for seg_key in seg_keys:
            if (attn_weights[f'layer{layer_idx}'][seg_key] and
                len(attn_weights[f'layer{layer_idx}'][seg_key]) > 0):
                num_heads = len(attn_weights[f'layer{layer_idx}'][seg_key][0])
                break

    # Create figures
    fig1, ax1 = plt.subplots(figsize=(12, 8))  # Aggregated attention by segment
    #fig2, ax2 = plt.subplots(figsize=(14, 10))  # Heatmap across layers
    # 1. AGGREGATED ATTENTION TO EACH SEGMENT
    # Initialize data structures
    segments = ['System', 'Image', 'Instruction', 'Output', 'I2System', 'I2Image']
    segment_keys = seg_keys
    segment_avgs = {segment: np.zeros(len(valid_layers)) for segment in segments}
    segment_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#ff7f0e', '#8c564b']

    # Compute average attention weights for each segment across layers
    for i, layer_idx in enumerate(valid_layers):
        layer_key = f'layer{layer_idx}'

        for segment, key in zip(segments, segment_keys):
            # Average across all generation steps and heads
            if (attn_weights[layer_key][key] and len(attn_weights[layer_key][key]) > 0):
                if avg_heads:
                    # Average across heads first, then across generation steps
                    segment_avgs[segment][i] = np.mean([np.mean(weights) for weights in attn_weights[layer_key][key]])
                else:
                    # Stack without averaging heads
                    all_weights = np.vstack([weights for weights in attn_weights[layer_key][key]])
                    segment_avgs[segment][i] = np.mean(all_weights, axis=0)  # Average across generation steps

    # Plot attention distribution
    for segment, color in zip(segments, segment_colors):
        ax1.plot(valid_layers, segment_avgs[segment], marker='o', color=color,
                 linewidth=2.5, markersize=8, label=segment)

    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('Average Attention Weight', fontsize=14)
    ax1.set_title(f'{title_prefix}Attention Distribution Across Layers', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_xticks(valid_layers)

    segments = segments[:-2]  # Exclude the last two segments (I2System and I2Image)
    segment_keys = segment_keys[:-2]  # Exclude the last two keys
    segment_avgs = {segment: segment_avgs[segment] for segment in segments}
    segment_colors = segment_colors[:-2]  # Exclude the last two colors

    imagetokens2system = np.zeros((len(valid_layers), 576))
    for i, layer_idx in enumerate(valid_layers):
        layer_key = f'layer{layer_idx}'
        all_weights = np.stack([weights for weights in attn_weights[layer_key]['imagetokens2system']]) #(samples, 576)
        imagetokens2system[layer_idx] = np.mean(all_weights, axis=0)  #576
    #plot imagetokens2system
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.imshow(imagetokens2system, cmap=cmap, aspect='auto', vmin=0, vmax=np.max(imagetokens2system) * 1.1)
    ax4.set_xlabel('Image Token', fontsize=14)
    ax4.set_ylabel('Layer', fontsize=14)
    ax4.set_title(f'{title_prefix}Image Tokens Attention to System Prompt', fontsize=16)
    ax4.set_yticks(range(len(valid_layers)))
    ax4.set_yticklabels(valid_layers, fontsize=10)
    ax4.set_xticks(range(576))
    ax4.set_xticklabels([f'{i}' for i in range(576)], fontsize=10)
    # tight_layout()
    fig4.tight_layout()

    if save_path:
        fig1.savefig(f"{save_path}_distribution.png", dpi=100, bbox_inches='tight')
        fig2.savefig(f"{save_path}_heatmap.png", dpi=100, bbox_inches='tight')
        fig3.savefig(f"{save_path}_heads.png", dpi=100, bbox_inches='tight')
        plt.close('all')
        print(f"Plots saved to {save_path}_*.png")
    else:
        plt.show()


def visualize_token_wise_attention(attn_weights, layers=32, save_path=None, title_prefix="",
                                   avg_heads=True, qk_type='output2images', layer_idx=None,
                                   rgb_image=None, grounding_attn=None, attn_size=None):
    """
    Visualize attention weights for specific tokens as 2D maps.

    Mode 1 (Default): Plots attention averaged over heads for all layers in a grid.
                      Triggered when `avg_heads` is True or `layer_idx` is None.
    Mode 2 (Single Layer, All Heads): Plots attention for each head of a specific layer.
                                      Triggered when `avg_heads` is False and `layer_idx` is an integer.

    Args:
        attn_weights: Dictionary from get_attn_weights_map containing attention weights.
                      Expected structure: attn_weights[f'layer{i}'][qk_type]
                      where the value is a numpy array of shape [num_heads, num_image_tokens].
        layers (int): Total number of layers potentially available in attn_weights.
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
        title_prefix (str, optional): Prefix for plot titles.
        avg_heads (bool): If True, averages across attention heads (Mode 1).
                          If False and layer_idx is set, plots each head separately (Mode 2).
        qk_type (str): Type of attention weights to visualize (e.g., 'output2images', 'question2images').
                       Must map to weights of shape [num_heads, num_image_tokens].
        layer_idx (int, optional): The specific layer index to visualize when avg_heads is False.

    Returns:
        None (displays or saves plots)
    """
    if qk_type not in ['output2images', 'question2images']:
        raise ValueError(f"This visualization function currently only supports "
                         f"qk_type='output2images' or 'question2images' for 2D mapping. Got: {qk_type}")

    # --- Mode 2: Plot all heads for a single specified layer ---
    if avg_heads is False and isinstance(layer_idx, int):
        layer_key = f'layer{layer_idx}'
        if layer_idx < 0 or layer_idx >= layers:
             print(f"Error: Specified layer_idx {layer_idx} is out of range (0-{layers-1}).")
             return
        if layer_key not in attn_weights or qk_type not in attn_weights[layer_key]:
            print(f"Error: No attention weights found for layer {layer_idx}, qk_type '{qk_type}'.")
            return

        weights = attn_weights[layer_key][qk_type]
        # Handle potential list wrapper
        if isinstance(weights, list):
            weights = weights[0] if weights else None

        if weights is None or not isinstance(weights, np.ndarray):
             print(f"Error: Weights for layer {layer_idx}, qk_type '{qk_type}' are not a valid numpy array.")
             return
        if weights.ndim != 2:
             print(f"Error: Expected weights for layer {layer_idx}, qk_type '{qk_type}' to have 2 dimensions (num_heads, num_tokens), but got {weights.ndim}.")
             return

        num_heads, num_image_tokens = weights.shape

        if num_image_tokens == 0:
             print(f"Error: No image tokens found in weights for layer {layer_idx}, qk_type '{qk_type}'.")
             return

        # Calculate grid size for the 2D map (per head)
        if attn_size is not None:
            grid_size = attn_size
        else:
            grid_size_f = math.sqrt(num_image_tokens)
            if grid_size_f != int(grid_size_f):
                raise ValueError(f"Number of image tokens ({num_image_tokens}) is not a perfect square. "
                                 f"Cannot reshape into a square 2D map.")
            grid_size = (int(grid_size_f), int(grid_size_f))

        # Determine subplot layout for heads
        cols = int(math.ceil(math.sqrt(num_heads)))
        rows = int(math.ceil(num_heads / cols))

        if rgb_image is not None:
            fig, axes_all = plt.subplots(rows, cols + 1, figsize=((cols + 1) * 2, rows * 2), squeeze=False)
            # Add image to first subplot
            ax_img = axes_all[0, 0]
            ax_img.imshow(np.array(rgb_image))
            ax_img.set_title("Input Image", fontsize=10)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            # Start plotting heads in second subplot
            axes = axes_all.flatten()[1:]
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
            axes = axes.flatten()

        # Find min/max across all heads for this layer
        min_val = weights.min()
        max_val = weights.max()
        # Ensure max_val > min_val for color mapping
        current_vmin = min_val
        current_vmax = max(max_val, min_val + 1e-9)

        im = None # To store the last image map for colorbar
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            head_weights = weights[head_idx, :]
            map_2d = head_weights.reshape(grid_size)

            im = ax.imshow(map_2d, cmap='viridis', vmin=current_vmin, vmax=current_vmax, aspect='equal')
            ax.set_title(f"Head {head_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        if grounding_attn is not None:
            ax = axes[num_heads]
            if ax is not None:
                map_2d = grounding_attn.reshape(grid_size)
                im = ax.imshow(map_2d, cmap='viridis', vmin=current_vmin, vmax=current_vmax, aspect='equal')
                ax.set_title(f"Grounding Attention", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            # Remove unused subplots
            for i in range(num_heads+1, len(axes)):
                fig.delaxes(axes[i])
        else:
            # Remove unused subplots
            for i in range(num_heads, len(axes)):
                fig.delaxes(axes[i])

        # Adjust layout and add colorbar/title
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.93]) # Adjust right/top margins

        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
            fig.colorbar(im, cax=cbar_ax, label='Attention Weight')

        title_str = f"{title_prefix} Layer {layer_idx} Heads Attention: {qk_type.replace('output2', 'Output -> ').replace('question2', 'Question -> ')} ({grid_size[0]}x{grid_size[1]} map)"
        fig.suptitle(title_str, fontsize=14, y=0.98)

        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Visualization for Layer {layer_idx} heads saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show(dpi=100)

        # --- End of Mode 2 ---
        return # Exit function after plotting single layer heads

    # --- Mode 1: Plot average attention for all layers (Original Logic) ---
    print("Mode: Plotting average attention across all layers.") # Inform user
    weights_found = False
    num_heads = 0
    num_image_tokens = 0
    first_layer_weights = None

    # Find the first valid layer to get dimensions (for Mode 1)
    for i in range(layers):
        layer_key = f'layer{i}'
        if layer_key in attn_weights and qk_type in attn_weights[layer_key]:
            potential_weights = attn_weights[layer_key][qk_type]
            if isinstance(potential_weights, list): potential_weights = potential_weights[0] if potential_weights else None

            if isinstance(potential_weights, np.ndarray) and potential_weights.ndim == 2:
                 num_heads, num_image_tokens = potential_weights.shape
                 if num_image_tokens > 0: # Ensure we actually have image tokens
                    weights_found = True
                    break

    if not weights_found:
        print(f"Warning (Mode 1): No valid attention weights found for qk_type='{qk_type}' "
              f"with >0 image tokens in the provided attn_weights dictionary for layers 0-{layers-1}.")
        return

    # Calculate grid size for the 2D map (per layer average)
    if attn_size is not None:
        grid_size = attn_size
    else:
        grid_size_f = math.sqrt(num_image_tokens)
        if grid_size_f != int(grid_size_f):
            raise ValueError(f"Number of image tokens ({num_image_tokens}) is not a perfect square. "
                             f"Cannot reshape into a square 2D map.")
        grid_size = (int(grid_size_f), int(grid_size_f))

    # Determine subplot layout for layers with space for the image
    cols = int(math.ceil(math.sqrt(layers)))
    rows = int(math.ceil(layers / cols))
    # Add extra slot for image if provided
    if rgb_image is not None:
        fig, axes = plt.subplots(rows, cols + 1, figsize=((cols + 1) * 2, rows * 2), squeeze=False)
        # Add image to first subplot
        ax_img = axes[0, 0]
        ax_img.imshow(np.array(rgb_image))
        ax_img.set_title("Input Image", fontsize=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        # Rest of plotting continues with shifted indices
        axes = axes.flatten()[1:]  # Skip the image subplot
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        axes = axes.flatten()

    # Find global min/max for consistent color scaling (average weights)
    min_val, max_val = float('inf'), float('-inf')
    valid_layer_indices = []
    processed_weights = {}
    plotted_something = False # Flag to track if any plot was actually made

    for i in range(layers):
        layer_key = f'layer{i}'
        if layer_key in attn_weights and qk_type in attn_weights[layer_key]:
             weights = attn_weights[layer_key][qk_type]
             # Handle potential list wrapper
             if isinstance(weights, list):
                 weights = weights[0] if weights else None

             # Check compatibility for averaging
             if weights is None or not isinstance(weights, np.ndarray) or weights.ndim != 2 or weights.shape[1] != num_image_tokens:
                 print(f"Warning (Mode 1): Skipping layer {i} due to missing, incompatible, or token count mismatch for {qk_type}.")
                 continue

             # Always average for Mode 1
             processed = weights.mean(axis=0) # Shape: [num_image_tokens]

             if processed.size == 0: # Check if processed array is empty
                 print(f"Warning (Mode 1): Skipping layer {i} because processed average weights are empty.")
                 continue

             processed_weights[i] = processed # Store for plotting loop
             min_val = min(min_val, processed.min())
             max_val = max(max_val, processed.max())
             plotted_something = True # Mark that we will plot at least one layer
             valid_layer_indices.append(i)


    if not plotted_something:
        print(f"Error (Mode 1): No valid data to plot after processing average weights for {qk_type}.")
        plt.close(fig) # Close the empty figure
        return

    # --- Plotting Loop (Mode 1) ---
    im = None # Initialize im to None
    plot_idx = 0
    for i in range(layers):
        ax = axes[plot_idx] if plot_idx < len(axes) else None # Get axis safely
        if ax is None: continue

        if i in valid_layer_indices and i in processed_weights:
            weights_to_plot = processed_weights[i]
            map_2d = weights_to_plot.reshape(grid_size)

            # Use vmin/vmax ensuring max_val > min_val
            current_vmin = min_val
            current_vmax = max(max_val, min_val + 1e-9)

            im = ax.imshow(map_2d, cmap='viridis', vmin=current_vmin, vmax=current_vmax, aspect='equal')
            ax.set_title(f"Layer {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
        else:
            # Keep empty subplots for missing layers to maintain grid structure
            ax.axis('off')
            plot_idx += 1 # Still advance plot index

    if grounding_attn is not None:
        ax = axes[plot_idx] if plot_idx < len(axes) else None # Get axis safely
        if ax is not None:
            map_2d = grounding_attn.reshape(grid_size)
            im = ax.imshow(map_2d, cmap='viridis', vmin=current_vmin, vmax=current_vmax, aspect='equal')
            ax.set_title(f"Grounding Attention", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plot_idx += 1
        else:
            # Keep empty subplots for missing layers to maintain grid structure
            ax.axis('off')
            plot_idx += 1
            print(f"Warning (Mode 1): No axis available for grounding attention. Skipping plot for it.")
    # --- Post-Plotting Adjustments (Mode 1) ---
    # Remove unused subplot axes more robustly
    for i in range(plot_idx, len(axes)):
         fig.delaxes(axes[i])

    # Adjust layout BEFORE adding colorbar
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Add a single colorbar in a dedicated axis
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Attention Weight (Avg Heads)')
    else:
        print("Warning (Mode 1): Cannot add colorbar as no images were plotted.")

    # Add the main title AFTER layout adjustments
    qk_title = qk_type.replace('output2', 'Output -> ').replace('question2', 'Question -> ')
    fig.suptitle(f"{title_prefix} Attention (Avg Heads): {qk_title} ({grid_size[0]}x{grid_size[1]} map)", fontsize=14, y=0.98)

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Visualization (Avg Layers) saved to: {save_path}")
    else:
        plt.show()
    # --- End of Mode 1 ---


def create_combined_visualization(all_results, layers=32):
    """
    Create a combined visualization of key metrics across all analyses.

    Args:
        all_results: Dictionary containing all analysis results
        layers: Number of layers to visualize
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(15, 10))

    layer_indices = range(layers)
    valid_layers = [i for i in layer_indices
                    if f'layer{i}' in all_results['cosine_similarity']
                    and f'layer{i}' in all_results['l2_norm']
                    and f'layer{i}' in all_results['relative']]

    # Extract key metrics
    visual_cosine_sims = [all_results['cosine_similarity'][f'layer{i}']['visual_residual_similarity'] for i in
                          valid_layers]
    text_cosine_sims = [all_results['cosine_similarity'][f'layer{i}']['text_residual_similarity'] for i in valid_layers]

    visual_ffn_pcts = [all_results['relative'][f'layer{i}']['visual_ffn_percent'] for i in valid_layers]
    text_ffn_pcts = [all_results['relative'][f'layer{i}']['text_ffn_percent'] for i in valid_layers]

    visual_l2_norms = [all_results['l2_norm'][f'layer{i}']['visual_ffn_l2_norm'] for i in valid_layers]
    text_l2_norms = [all_results['l2_norm'][f'layer{i}']['text_ffn_l2_norm'] for i in valid_layers]

    # Create a 2x2 subplot grid
    plt.subplot(2, 2, 1)
    plt.plot(valid_layers, visual_cosine_sims, 'b-', marker='o', label='Visual')
    plt.plot(valid_layers, text_cosine_sims, 'r-', marker='x', label='Text')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Residual Feature Preservation')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(valid_layers, visual_ffn_pcts, 'b-', marker='o', label='Visual')
    plt.plot(valid_layers, text_ffn_pcts, 'r-', marker='x', label='Text')
    plt.xlabel('Layer')
    plt.ylabel('FFN Contribution (%)')
    plt.title('FFN Relative Contribution')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(valid_layers, visual_l2_norms, 'b-', marker='o', label='Visual')
    plt.plot(valid_layers, text_l2_norms, 'r-', marker='x', label='Text')
    plt.xlabel('Layer')
    plt.ylabel('L2 Norm')
    plt.title('FFN Contribution Magnitude')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    # Calculate the difference between visual and text metrics
    cosine_diffs = np.array(visual_cosine_sims) - np.array(text_cosine_sims)
    ffn_pct_diffs = np.array(visual_ffn_pcts) - np.array(text_ffn_pcts)
    l2_norm_ratios = np.array(visual_l2_norms) / np.array(text_l2_norms)

    plt.plot(valid_layers, cosine_diffs, 'g-', marker='o', label='Cosine Sim Diff')
    plt.plot(valid_layers, ffn_pct_diffs, 'm-', marker='x', label='FFN % Diff')
    plt.plot(valid_layers, l2_norm_ratios, 'c-', marker='^', label='L2 Norm Ratio')
    plt.xlabel('Layer')
    plt.ylabel('Difference/Ratio')
    plt.title('Visual vs Text Differences')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle=':')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.savefig('ffn_analysis_combined.png')
    # plt.close()
    plt.show()



def plot_tensor_2d(tensor, title=None, cmap='viridis', figsize=(6, 6), show_colorbar=True):
    """
    Plot a 2D PyTorch tensor as a heatmap.

    Args:
        tensor: 2D PyTorch tensor
        title: Optional title for the plot
        cmap: Colormap to use (default: 'viridis')
        figsize: Figure size as (width, height) in inches
        show_colorbar: Whether to display a colorbar
    """
    # Ensure the tensor is 2D
    if len(tensor.shape) != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tensor.shape}")

    # Convert to numpy for plotting
    if tensor.requires_grad:
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor.cpu().numpy()

    # Create the plot
    plt.figure(figsize=figsize)
    im = plt.imshow(array, cmap=cmap)

    # Add title if provided
    if title:
        plt.title(title)

    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im)

    plt.tight_layout()
    plt.show()


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

def plot_attention_analysis(image, grounding_attn, save_path=None, figsize=(8, 4), blend_attn_mask=False):
    """
    Plot image and attention maps.

    Args:
        image_path: str, path to the image file
        layer2_attn: ndarray (576,), layer 2 attention weights
        grounding_attn_o2i: ndarray (576,), grounding attention object to image
        grounding_attn: ndarray (576,), grounding attention
        save_path: str, optional path to save the plot
        figsize: tuple, figure size
    """
    # Create figure with subplots
    ncol = 2 if grounding_attn is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=figsize)
    if type(image) == str:
        img = Image.open(image)
    elif isinstance(image, torch.Tensor):
        unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        image = unnorm(image[0].to(torch.float16)).cpu()
        img = image.permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8) #[336,336,3]
    elif isinstance(image, np.ndarray):
        img = image
    else: raise ValueError(f"Unsupported image type: {type(image)}. Expected str or torch.Tensor.")

    if grounding_attn is None:
        axes = [axes]
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Reshape attention arrays to 24x24
    if grounding_attn is not None:
        grounding_attn_2d = grounding_attn
        if blend_attn_mask:
            # --- Parameters for visualization ---
            blend_attn_mask = True  # Assuming this is True to enter the block
            gamma_value = 0.7  # Affects the contrast of the attention's influence
            min_visibility_factor = 0.0  # (e.g., 0.1 means even 0-attention areas are 10% bright)
            # Tune this: 0 means black, 1 means original image unchanged

            raw_2d_attention_map = grounding_attn_2d  # Assuming this is the raw attention map
            img_original_uint8 = img  # Assuming img is already in uint8 format [0, 255]
            # 1. Normalize the raw 2D attention map to [0, 1]
            max_attn_val = raw_2d_attention_map.max()
            if max_attn_val > 0:
                normalized_attention_np = raw_2d_attention_map / max_attn_val
            else:
                normalized_attention_np = np.zeros_like(raw_2d_attention_map, dtype=np.float32)

            normalized_attention_np = np.clip(normalized_attention_np.astype(np.float32), 0.0, 1.0)

            # 2. Get image dimensions and prepare attention tensor for PyTorch interpolation
            H, W = img_original_uint8.shape[:2]
            # Reshape for PyTorch: (h_attn, w_attn) -> (1, 1, h_attn, w_attn)
            attention_tensor = torch.from_numpy(normalized_attention_np).unsqueeze(0).unsqueeze(0)

            # 3. Resize attention map to image dimensions using 'nearest' to preserve sharp edges if desired
            #    For smoother attention, 'bilinear' could be used: mode='bilinear', align_corners=False
            resized_attention_tensor = F.interpolate(attention_tensor, size=(H, W), mode='nearest')

            # 4. Apply Gamma Correction to the resized attention tensor (still in PyTorch)
            gamma_corrected_attention_tensor = torch.pow(resized_attention_tensor, gamma_value)
            gamma_corrected_attention_tensor = torch.clamp(gamma_corrected_attention_tensor, 0.0, 1.0)

            # 5. Convert gamma-corrected attention back to NumPy array
            #    Shape: (1, 1, H, W) -> (H, W)
            processed_attention_np = gamma_corrected_attention_tensor.squeeze().cpu().numpy()
            processed_attention_np = np.clip(processed_attention_np, 0.0, 1.0)  # Final clip for safety

            # 6. Apply Minimum Brightness logic to the processed attention
            #    This scales the attention so that 0 becomes min_visibility_factor, and 1 remains 1.
            final_brightness_modulator_np = min_visibility_factor + (1.0 - min_visibility_factor) * processed_attention_np
            final_brightness_modulator_np = np.clip(final_brightness_modulator_np, 0.0, 1.0)  # Ensure it's in [0,1]

            # 7. Prepare for broadcasting with the image: (H, W) -> (H, W, 1)
            brightness_modulator_for_broadcast = final_brightness_modulator_np[:, :, np.newaxis]

            # 8. Normalize original image to [0, 1] float
            img_float_0_1 = img_original_uint8.astype(np.float32) / 255.0

            # 9. Modulate image brightness using the final modulator
            blended_img_float_0_1 = img_float_0_1 * brightness_modulator_for_broadcast

            # 10. Convert the blended image back to uint8 [0, 255]
            # This is the image you want to display.
            output_visualization_uint8 = (np.clip(blended_img_float_0_1, 0.0, 1.0) * 255).astype(np.uint8)

            grounding_attn_2d = output_visualization_uint8
        # Plot 2: Grounding Attention Q2I
        im2 = axes[1].imshow(grounding_attn_2d, cmap='inferno', aspect='auto')
        axes[1].set_title('Predicted RoI')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.6)

    # # Plot 2: Grounding Attention Q2I after sigmoid
    # im3 = axes[2].imshow(torch.tensor(grounding_attn_2d).sigmoid().numpy(), cmap='inferno', aspect='auto')
    # axes[2].set_title('Predicted RoI')
    # axes[2].axis('off')
    # plt.colorbar(im3, ax=axes[2], shrink=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.show()
    pass


def plot_image_with_heatmaps(image, attention_maps, save_path=None, figsize=None, dpi=100):
    """
    Plots an image alongside one or more of its attention maps (heatmaps).

    Args:
        image (str, torch.Tensor, np.ndarray): The source image.
        attention_maps (list of dict): A list of attention maps to plot.
        save_path (str, optional): Path to save the final plot.
        figsize (tuple, optional): The figure size in inches.
        dpi (int, optional): The resolution in dots per inch for both display and saving.
                             Defaults to 100.
    """
    # --- 1. Set up the figure and subplots ---
    num_maps = len(attention_maps)
    if num_maps == 0:
        print("Warning: No attention maps provided to plot.")
        return

    ncol = 1 + num_maps
    if figsize is None:
        figsize = (4 * ncol, 4)

    # Pass dpi at figure creation to affect both display and save resolution
    fig, axes = plt.subplots(1, ncol, figsize=figsize, dpi=dpi)

    if ncol == 1:
        axes = [axes]

    # --- 2. Process and plot the original image ---
    if isinstance(image, str):
        img = Image.open(image)
        img = np.array(img)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        image = unnorm(image.to(torch.float32)).cpu()
        img = image.permute(1, 2, 0).numpy()
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}. Expected str, torch.Tensor, or np.ndarray.")

    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # --- 3. Loop through and plot each attention map ---
    for i, map_info in enumerate(attention_maps):
        ax = axes[i + 1]

        raw_attn_map = map_info['map']
        title = map_info.get('title', f'Map {i + 1}')
        blend_mask = map_info.get('blend', False)
        cmap = map_info.get('cmap', 'inferno')

        if isinstance(raw_attn_map, torch.Tensor):
            raw_attn_map = raw_attn_map.float().detach().cpu().numpy()

        if blend_mask:
            # ... (blending logic remains the same) ...
            gamma_value = map_info.get('gamma', 0.7)
            min_visibility_factor = map_info.get('min_bright', 0.0)
            max_val = raw_attn_map.max()
            norm_map = raw_attn_map / max_val if max_val > 0 else np.zeros_like(raw_attn_map)
            H, W = img.shape[:2]
            attn_tensor = torch.from_numpy(norm_map).unsqueeze(0).unsqueeze(0)
            resized_attn = F.interpolate(attn_tensor, size=(H, W), mode='bilinear', align_corners=False)
            gamma_corrected_attn = torch.pow(resized_attn, gamma_value).squeeze().cpu().numpy()
            brightness_modulator = min_visibility_factor + (1.0 - min_visibility_factor) * gamma_corrected_attn
            brightness_modulator = np.clip(brightness_modulator, 0.0, 1.0)[:, :, np.newaxis]
            img_float = img.astype(np.float32) / 255.0
            blended_img_float = img_float * brightness_modulator
            display_data = (np.clip(blended_img_float, 0.0, 1.0) * 255).astype(np.uint8)
            ax.imshow(display_data)
        else:
            display_data = raw_attn_map
            im = ax.imshow(display_data, cmap=cmap)
            fig.colorbar(im, ax=ax, shrink=0.7)

        ax.set_title(title)
        ax.axis('off')

    # --- 4. Finalize and show/save the plot ---
    plt.tight_layout()
    if save_path:
        # The figure already has the correct DPI, so you don't need to specify it again.
        # It will use the DPI set during figure creation by default.
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

import matplotlib.patches as patches

def plot_bboxes_on_img(heatmap, bboxes, ax=None, cmap='inferno', title="Heatmap with Bounding Boxes"):
    """
    Plots bounding boxes on a heatmap.

    Args:
        heatmap : A 2D tensor representing the heatmap or a PIL image.
        bboxes (torch.tensor): A tensor of shape (N, 4) with bboxes [x_min, y_min, x_max, y_max].
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
        title (str): The title for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1)
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    # Move tensor to CPU and convert to numpy for plotting
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.detach().cpu().numpy()
    elif isinstance(heatmap, np.ndarray):
        heatmap_np = heatmap

    # Display the heatmap
    ax.imshow(heatmap_np, cmap=cmap)
    ax.set_title(title)

    # Add bounding boxes
    if bboxes is not None and len(bboxes) > 0:
        bboxes_np = bboxes.cpu().numpy()
        for bbox in bboxes_np:
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='lime', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

    if show_plot:
        plt.tight_layout()
        plt.show()


def get_bbox4src(noisy_bbox, src_img, feat_size=(24, 24), img_size=(336, 336)):
    '''
    Converts bounding boxes from feature map coordinates to original source image coordinates.
    This assumes the source image was resized to fit within img_size, maintaining aspect ratio,
    and then center-padded.

    Args:
    noisy_bbox: torch.Tensor, shape (N, 4), [x_min, y_min, x_max, y_max] derived from a noisy feature map.
    src_img: PIL.Image or torch.Tensor, the source image.
    feat_size: tuple (width, height), the size of the feature map.
    img_size: tuple (width, height), the target size of the processed image (after resizing and padding).

    Returns:
        torch.Tensor: A tensor of shape (N, 4) containing the bounding boxes in the original image coordinates,
                      clamped to the image boundaries.
    '''
    if noisy_bbox.numel() == 0:
        return torch.empty((0, 4), dtype=torch.long, device=noisy_bbox.device)

    # 1. Get original image size
    if isinstance(src_img, Image.Image):
        src_w, src_h = src_img.size
    elif isinstance(src_img, torch.Tensor):
        if src_img.dim() >= 2:  # Covers (H, W), (C, H, W), etc.
            src_h, src_w = src_img.shape[-2:]
        else:
            raise ValueError(f"Unsupported torch.Tensor shape: {src_img.shape}")
    else:
        raise TypeError(f"Unsupported type for src_img: {type(src_img)}")

    feat_h, feat_w = feat_size
    img_h, img_w = img_size

    # 2. Scale bounding boxes from feature map coordinates to padded image coordinates
    scale_x = img_w / feat_w
    scale_y = img_h / feat_h

    bboxes_padded = noisy_bbox.float().clone()
    bboxes_padded[:, 0::2] *= scale_x  # Scale all x coordinates (min and max)
    bboxes_padded[:, 1::2] *= scale_y  # Scale all y coordinates (min and max)


    resize_x, resize_y = src_w / img_w, src_h / img_h
    src_bboxes = torch.zeros_like(bboxes_padded, dtype=torch.float32)

    src_bboxes[:, 0::2] = bboxes_padded[:, 0::2] * resize_x
    src_bboxes[:, 1::2] = bboxes_padded[:, 1::2] * resize_y

    src_bboxes[:, 0::2] = torch.clamp(src_bboxes[:, 0::2], min=0, max=src_w)
    src_bboxes[:, 1::2] = torch.clamp(src_bboxes[:, 1::2], min=0, max=src_h)

    return src_bboxes.long()

import torchvision.transforms.functional as torchvision_F
def get_bbox_from_noisy_map(noisy_map, conf_thresh, min_area=4, max_area=None):
    """
    Get N bounding boxes from a noisy map based on confidence threshold and area constraints.

    Args:
        noisy_map (torch.tensor): A 2D tensor of shape (H, W) representing the noisy map with confidence scores.
        conf_thresh (float): Confidence threshold to filter out low-confidence areas.
        min_area (int): Minimum area of the bounding box to consider.
        max_area (int, optional): Maximum area of the bounding box to consider. If None, no limit is applied.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) where N is the number of found bounding boxes,
                      and each row is [x_min, y_min, x_max, y_max]. Returns an empty tensor if no bbox is found.
    """
    # draft steps: gaussian blur, threshold, split connect regions, merge small regions, filter over-small regions,  get bounding boxes

    if noisy_map.dim() != 2:
        raise ValueError(f"Input noisy_map is expected to be a 2D tensor, but got {noisy_map.dim()} dimensions.")

    H,W = noisy_map.size()
    # Step 1: Gaussian blur to smooth the map.
    # Add batch and channel dimensions for gaussian_blur, then remove them.
    blurred_map = torchvision_F.gaussian_blur(noisy_map.unsqueeze(0).unsqueeze(0), kernel_size=3, sigma=1.0).squeeze()
    #plot_tensor_2d(blurred_map, title='Blurred Map', cmap='inferno')

    # Step 2: Threshold the map to get a binary map.
    if conf_thresh<0:
        binary_map = (blurred_map > blurred_map.max() * (-conf_thresh)).cpu().numpy()
    else:
        binary_map = (blurred_map > conf_thresh).cpu().numpy()
    #plot_tensor_2d(torch.tensor(blurred_map), title='Blurred Map', cmap='inferno')

    # Step 3: Find connected components (regions).
    # `connectivity=2` indicates 8-way connectivity.
    labeled_map, num_labels = label(binary_map, connectivity=2, return_num=True)

    if num_labels == 0:
        return torch.empty((0, 4), dtype=torch.long, device=noisy_map.device)

    bboxes = []
    for i in range(1, num_labels + 1):
        # Find all pixels belonging to the current label
        coords = np.argwhere(labeled_map == i)

        # Step 5: Get bounding box for the current region.
        # np.argwhere returns coordinates in (row, col) i.e., (y, x) format.
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_max, x_max = y_max + 1, x_max + 1  # Make inclusive
        # Step 6: Filter regions based on area.
        area = len(coords)
        if area == 1: continue #remove sink and noise
        if area < min_area:
            # expand to at least 3*3
            xc, yc = (x_min + x_max) / 2, (y_min + y_max) / 2
            w_min = max(3, x_max - x_min + 1)
            h_min = max(3, y_max - y_min + 1)
            x_min = max(0, int(xc - w_min / 2))
            y_min = max(0, int(yc - h_min / 2))
            x_max = min(W - 1, int(xc + w_min / 2))
            y_max = min(H - 1, int(yc + h_min / 2))

        if max_area is not None and area > max_area:
            continue

        bboxes.append([x_min, y_min, x_max, y_max])

    if not bboxes:
        return torch.empty((0, 4), dtype=torch.long, device=noisy_map.device)

    return torch.tensor(bboxes, dtype=torch.long, device=noisy_map.device)

