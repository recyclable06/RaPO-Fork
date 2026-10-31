"""Whole-model LoRA + directly trained vectors, and non-mutating weight export.

The supported matrix layers are the ones used by Qwen2-VL. Keep this separate
from vLLM's language-only adapter protocol: rollout and Anchor consume ordinary
model weights.
"""

import weakref
from collections import Counter

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch import nn


class TiedLoraHead(nn.Module):
    """Use the input embedding's factors without registering duplicate parameters.

    Both embedding and head must stay in the root FSDP unit. A transpose wrapped
    in a second nn.Parameter would create independent optimizer/FSDP states.
    """

    def __init__(self, base_layer: nn.Linear, embedding: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self._embedding = weakref.ref(embedding)

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, hidden_states):
        embedding = self._embedding()
        a = embedding.lora_embedding_A["default"]
        b = embedding.lora_embedding_B["default"]
        result = self.base_layer(hidden_states)
        delta = (hidden_states.to(b.dtype) @ b) @ a
        return result + (delta * embedding.scaling["default"]).to(result.dtype)


def apply_full_lora(model, rank: int, alpha: int):
    """Cover every original trainable parameter, or report the unsupported names."""
    original = {id(p): (name, p) for name, p in model.named_parameters() if p.requires_grad}
    covered = set()
    targets = []
    counts = Counter()
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = head is not None and head.weight is embedding.weight

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding, nn.Conv3d)):
            continue
        if id(module.weight) not in original:
            continue
        if isinstance(module, nn.Conv3d) and module.groups != 1:
            raise ValueError(f"Whole-model LoRA only supports groups=1 Conv3d: {name}")
        if not (tied and module is head):
            targets.append(name)
        if id(module.weight) not in covered:
            kind = "embedding" if isinstance(module, nn.Embedding) else (
                "conv3d" if isinstance(module, nn.Conv3d) else "linear"
            )
            counts[kind] += module.weight.numel()
            covered.add(id(module.weight))

    vectors = []
    for param_id, (name, param) in original.items():
        if param.ndim == 1:
            vectors.append(param)
            counts["vectors"] += param.numel()
            covered.add(param_id)
    missing = [name for param_id, (name, _) in original.items() if param_id not in covered]
    if missing:
        raise ValueError(f"Original parameters without an update path: {missing}")

    # Do not create a second adapter for the tied head. Reuse the actual input
    # embedding factors in its forward, including their combined gradients.
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        target_modules=targets, bias="none",
    ))
    for param in vectors:
        param.requires_grad_(True)
    if tied:
        model.get_base_model().set_output_embeddings(TiedLoraHead(head, model.get_input_embeddings()))
    model._rapo_full_lora = True
    counts["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, dict(counts)


def _clean_name(name):
    return name.replace("_fsdp_wrapped_module.", "").removeprefix("base_model.model.")


@torch.no_grad()
def effective_lora_state_dict(model: PeftModel, state_dict: dict) -> dict:
    """Merge a FULL gathered state dict without changing Actor or its optimizer.

    FSDP collection is done by the caller. Metadata comes from the live layers;
    all numerical operations use gathered tensors, never local FSDP shards.
    """
    state = {_clean_name(name): value for name, value in state_dict.items()}
    result = {
        name.replace(".base_layer.", "."): value
        for name, value in state.items() if ".lora_" not in name
    }
    embedding_name = None
    tied_heads = []
    for raw_name, layer in model.named_modules():
        name = _clean_name(raw_name)
        if isinstance(layer, TiedLoraHead):
            tied_heads.append(name)
            continue
        if not isinstance(layer, LoraLayer):
            continue
        base = layer.get_base_layer()
        weight = state[f"{name}.base_layer.weight"]
        if isinstance(base, nn.Embedding):
            a = state[f"{name}.lora_embedding_A.default"].float()
            b = state[f"{name}.lora_embedding_B.default"].float()
            delta = (b @ a).T
            if layer is model.get_input_embeddings():
                embedding_name = name
        else:
            a = state[f"{name}.lora_A.default.weight"].float()
            b = state[f"{name}.lora_B.default.weight"].float()
            if isinstance(base, nn.Conv3d):
                delta = (b.flatten(1) @ a.flatten(1)).reshape(weight.shape)
            elif isinstance(base, nn.Linear):
                delta = b @ a
            else:
                raise TypeError(f"Unsupported LoRA layer for effective export: {name}")
        result[f"{name}.weight"] = (weight.float() + delta * layer.scaling["default"]).to(weight.dtype)
    for name in tied_heads:
        if embedding_name is None:
            raise ValueError("Tied output head has no adapted input embedding")
        result[f"{name}.weight"] = result[f"{embedding_name}.weight"]
    return result


def gather_effective_lora_weights(module):
    """Collect once on every rank, using the same FSDP state-dict API as rollout."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    state = get_model_state_dict(module, options=StateDictOptions(full_state_dict=True))
    return effective_lora_state_dict(module._fsdp_wrapped_module, state)
