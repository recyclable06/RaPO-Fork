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
from torch.distributed.tensor import DTensor, Replicate


_MERGE_ROWS = 1024


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
    """Small-model reference formula for numerical checks, not production sync.

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


def cpu_sharded_state_dict(module):
    """Stage local state shards while vLLM sleeps; never collect a full model."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(module, options=StateDictOptions(full_state_dict=False, cpu_offload=True))


@torch.no_grad()
def _merge_weight_rows(weight, a, b, scaling, embedding):
    """Allocate one output matrix; FP32 workspace is limited to a row block."""
    left, right = (a.T, b.T) if embedding else (b.flatten(1), a.flatten(1))
    right = right.float()
    output = torch.empty_like(weight, memory_format=torch.contiguous_format)
    for start in range(0, weight.shape[0], _MERGE_ROWS):
        stop = min(start + _MERGE_ROWS, weight.shape[0])
        block = left[start:stop].float() @ right
        block.mul_(scaling)
        block.add_(weight[start:stop].reshape(stop - start, -1))
        output[start:stop].copy_(block.reshape(output[start:stop].shape))
        del block
    return output


def _full_weight(shard, device):
    # Offloaded DTensors still carry the CUDA mesh. Move the local shard back
    # before collective communication; NCCL cannot all-gather CPU tensors.
    tensor = shard.to(device)
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def iter_effective_lora_weights(model: PeftModel, cpu_state: dict, device):
    """Yield complete ordinary weights, retaining only the current GPU matrix.

    Every rank drives this iterator in the same order, independently of whether
    a receiver consumes a name. The caller must release each yielded weight
    before requesting the next one. Shapes come from the snapshot, not FSDP's
    live (possibly flattened/empty) parameter views.
    """
    state = {_clean_name(name): value for name, value in cpu_state.items()}
    sources = {name.replace(".base_layer.", "."): name for name in state if ".lora_" not in name}
    merges = {}
    embedding_name = None
    tied_heads = []
    for raw_name, layer in model.named_modules():
        name = _clean_name(raw_name)
        if isinstance(layer, TiedLoraHead):
            tied_heads.append(f"{name}.weight")
        elif isinstance(layer, LoraLayer):
            embedding = isinstance(layer.get_base_layer(), nn.Embedding)
            if embedding:
                a, b = f"{name}.lora_embedding_A.default", f"{name}.lora_embedding_B.default"
                if layer is model.get_input_embeddings():
                    embedding_name = f"{name}.weight"
            else:
                a, b = f"{name}.lora_A.default.weight", f"{name}.lora_B.default.weight"
            merges[f"{name}.weight"] = (a, b, layer.scaling["default"], embedding)
    if tied_heads and embedding_name is None:
        raise ValueError("Tied output head has no adapted input embedding")

    for name in sorted(sources):
        if name in tied_heads:
            continue  # Emitted with the source embedding, without a second merge.
        weight = _full_weight(state[sources[name]], device)
        if name in merges:
            a_name, b_name, scaling, embedding = merges[name]
            a = _full_weight(state[a_name], device)
            b = _full_weight(state[b_name], device)
            effective = _merge_weight_rows(weight, a, b, scaling, embedding)
            del weight, a, b
        else:
            effective = weight
            del weight
        yield name, effective
        if name == embedding_name:
            for head in sorted(tied_heads):
                yield head, effective
        del effective


@torch.no_grad()
def _anchor_shard(weight, template):
    if weight.shape != template.shape:
        raise ValueError(f"Anchor weight shape mismatch: {weight.shape} != {template.shape}")
    weight = weight.to(template.dtype)
    if isinstance(template, DTensor):
        # Each rank already computed the same complete matrix. Public DTensor
        # redistribution performs local chunking from Replicate to Shard, using
        # the target layout (including uneven/empty shards), not flat offsets.
        replicated = DTensor.from_local(
            weight, device_mesh=template.device_mesh,
            placements=[Replicate()] * template.device_mesh.ndim,
        )
        return replicated.redistribute(placements=template.placements).cpu()
    return weight.cpu()


@torch.no_grad()
def copy_effective_lora_to_anchor(actor, anchor):
    """Build local CPU target shards, then load Anchor once in sharded mode."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    actor_state = cpu_sharded_state_dict(actor)
    remaining = cpu_sharded_state_dict(anchor)
    target_names = {_clean_name(name): name for name in remaining}
    updates = {}
    device = torch.device("cuda", torch.cuda.current_device())
    for name, weight in iter_effective_lora_weights(actor._fsdp_wrapped_module, actor_state, device):
        target = target_names[name]
        template = remaining.pop(target)
        updates[target] = _anchor_shard(weight, template)
        del template, weight
    if remaining:
        raise ValueError(f"Anchor parameters missing from effective policy: {sorted(remaining)}")
    del actor_state
    set_model_state_dict(
        anchor, updates,
        options=StateDictOptions(full_state_dict=False, cpu_offload=True, strict=True),
    )
