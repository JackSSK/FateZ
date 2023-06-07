#!/usr/bin/env python3
"""
Classifier but using low rank adaption model.

author: jy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import fatez.model.transformer as transformer
import fatez.model.position_embedder as pe



class Classifier(nn.Module):
    """
    Classification model.
    """
    def __init__(self,
        rep_embedder = pe.Skip(),
        encoder:transformer.Encoder = None,
        adaptor = None,
        classifier = None,
        **kwargs
        ):
        """
        :param rep_embedder: = position_embedder.Skip
            Positional embedding method for GNN-encoded representations.

        :param encoder:transformer.Encoder = None
            The Encoder to build fine-tune model with.

        :param classifier = None
            The classification model for making predictions.
        """
        super(Classifier, self).__init__()
        self.rep_embedder = rep_embedder
        self.encoder = encoder
        self.adaptor = adaptor
        self.classifier = classifier

    def forward(self,
            src: torch.Tensor,
            mask: torch.Tensor = None,
            src_key_padding_mask: torch.Tensor = None,
            is_causal: bool = None,
            ) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as mask (optional)
                and ignores attn_mask for computing scaled dot product attention.
                Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = self.rep_embedder(src)
        convert_to_nested = False
        first_layer = self.encoder.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.encoder.layers[0]"
        if not isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.encoder.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.encoder.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = torch.triu(
                    torch.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if torch.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.encoder.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers
            )

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.encoder.norm is not None:
            output = self.encoder.norm(output)

        output = self.classifier(output)
        return output
