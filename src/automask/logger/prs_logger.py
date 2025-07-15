import re
import math

import torch
import torch.nn as nn


def _get_head(scores, head_index: int):
    if head_index == -1:
        return scores.mean(dim=0)
    elif head_index == -2:
        return scores.reshape(-1, scores.size(-1))
    else:
        try:
            return scores[head_index]  # shape: [H*W, H*W]
        except IndexError as e:
            raise IndexError(
                f"Head index {head_index} is out of range for tensor with shape {scores.shape}"
            ) from e


def _do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x

    x = x.permute(0, 3, 1, 2)
    x = pool(x)

    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


def _scaled_dot_product_attention_scores(query, key, scale=None) -> torch.Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    return torch.softmax(attn_weight, dim=-1)


class PRSLogger:
    """
    PRSLogger is a utility class for capturing and analyzing attention scores from specific layers
    of a model. It provides functionality to attach hooks to the model, retrieve attention scores,
    and manage resources efficiently.

    Attributes:
        model (torch.nn.Module): The model from which attention scores are captured.
        layer_index (int): The index of the layer to monitor for attention scores.
        head_index (int): The index of the attention head to analyze.
        spatial (bool): Whether to process spatial attention scores. Defaults to True.
        attentions (dict): A dictionary to store attention inputs keyed by layer names.
        hooks (list): A list of registered hooks for capturing attention inputs.

    Methods:
        __init__(model, layer_index, head_index, spatial=True):
            Initializes the PRSLogger with the given model, layer index, and head index.

        get_attention_scores(device="cuda"):
            Computes and retrieves the attention scores for the specified layer and head.

        get_hook(name):
            Returns a hook function that captures attention inputs for a specific layer.

        attach_hooks():
            Attaches forward hooks to the model layers that handle attention.

        detach_hooks():
            Removes all registered hooks to free memory.

        reset(reattach_hooks=True):
            Clears all stored data, detaches hooks, and optionally reattaches hooks.
    """

    def __init__(self, model, layer_index, head_index, spatial=True):
        self.attentions = {}
        self.spatial = spatial
        self.index = layer_index
        self.head_index = head_index
        self.model = model
        self.hooks = []

    @torch.no_grad()
    def get_attention_scores(self, device="cuda"):
        """
        Re computes on the fly attention scores for the specified layer and head.
        """
        layer = f"image_encoder.trunk.blocks.{self.index}.attn.qkv"

        x = self.attentions[layer].to(device)

        B, H, W, _ = x.shape
        num_heads = self.model.image_encoder.trunk.blocks[self.index].attn.num_heads

        qkv = (
            self.model.image_encoder.trunk.blocks[self.index]
            .attn.qkv(x)
            .reshape(B, H * W, 3, num_heads, -1)
        )

        q, k, _ = torch.unbind(qkv, 2)

        q_pool = self.model.image_encoder.trunk.blocks[self.index].attn.q_pool

        if q_pool:
            q = _do_pool(q.reshape(B, H, W, -1), q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, num_heads, -1)

        scores = _scaled_dot_product_attention_scores(
            q.transpose(1, 2),
            k.transpose(1, 2),
        )

        return _get_head(scores[0], self.head_index)  # [H*W, H*W]
    
    def get_attention_embeddings(self, device="cuda"): 
        layer = f"image_encoder.trunk.blocks.{self.index}.attn.qkv"

        return self.attentions[layer]
    
    def get_mlp_embeddings(self, device="cuda"):
        """
        Re computes on the fly MLP embeddings for the specified layer.
        """
        layer = f"image_encoder.trunk.blocks.{self.index}.mlp.layers.1"

        return  self.attentions[layer].to(device)
        

    def get_hook(self, name):
        """Returns a function that captures attention inputs with the corresponding layer name."""

        def store_attention_scores(module, input, output):
            if input:
                attn_input = input[0].detach()
                self.attentions[name] = attn_input

        return store_attention_scores

    def attach_hook(self):
        """
        Attaches a forward hook only to the qkv module of the specified layer.
        """
        layer_name = f"image_encoder.trunk.blocks.{self.index}.attn.qkv"

        modules_dict = dict(self.model.named_modules())
        module = modules_dict.get(layer_name, None)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in the model.")

        hook = module.register_forward_hook(self.get_hook(layer_name))
        self.hooks.append(hook)
        
    def attach_hook_mlp(self):
        """
        Attaches a forward hook to the MLP module of the specified layer.
        """
        layer_name = f"image_encoder.trunk.blocks.{self.index}.mlp.layers.1"
        
        modules_dict = dict(self.model.named_modules())
        module = modules_dict.get(layer_name, None)
        if module is None:
            raise ValueError(f"Module '{layer_name}' not found in the model.")

        hook = module.register_forward_hook(self.get_hook(layer_name))
        self.hooks.append(hook)

    def attach_hooks(self):
        """Attaches hooks to model layers that handle attention."""
        pattern = r"image_encoder\.trunk\.blocks\.\d+\.attn\.qkv"

        for name, module in self.model.named_modules():
            if re.search(pattern, name):
                hook = module.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)

    def detach_hooks(self):
        """Removes all registered hooks to free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_attentions(self):
        """Clears all stored attention values."""
        self.attentions.clear()

    def reset_logger(self, reattach_hooks=True):
        """Clears all stored data except for the model."""
        self.detach_hooks()

        self.attentions.clear()

        torch.cuda.empty_cache()

        if reattach_hooks:
            self.attach_hook()
