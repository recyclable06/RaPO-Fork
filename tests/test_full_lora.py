"""CPU checks for the new parameterization, not a distributed training test.

Run: python -m unittest discover -s tests -p test_full_lora.py -v
"""

import copy
import io
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from verl.utils.full_lora import (
    _anchor_shard,
    apply_full_lora,
    cpu_sharded_state_dict,
    effective_lora_state_dict,
    iter_effective_lora_weights,
)


def tiny_model():
    config = Qwen2VLConfig(
        vocab_size=48, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        tie_word_embeddings=True, image_token_id=45, video_token_id=46,
        vision_start_token_id=44, vision_end_token_id=47,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        vision_config={
            "depth": 1, "embed_dim": 32, "hidden_size": 32,
            "num_heads": 4, "mlp_ratio": 2, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
        },
    )
    config._attn_implementation = "eager"
    return Qwen2VLForConditionalGeneration(config)


class FullLoraTest(unittest.TestCase):
    def test_blocked_sync_matches_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(13)
                plain = tiny_model().to(dtype).eval()
                anchor = copy.deepcopy(plain).requires_grad_(False)
                actor, _ = apply_full_lora(plain, rank=3, alpha=7)
                with torch.no_grad():
                    for param in actor.parameters():
                        if param.requires_grad:
                            param.add_(torch.randn_like(param) * 0.03)
                before = {name: value.clone() for name, value in actor.state_dict().items()}
                reference = effective_lora_state_dict(actor, before)
                snapshot = cpu_sharded_state_dict(actor)
                self.assertTrue(all(value.device.type == "cpu" for value in snapshot.values()))
                # Force non-divisible row blocks for linear, embedding, Conv3d.
                with patch("verl.utils.full_lora._MERGE_ROWS", 7):
                    weights = dict(iter_effective_lora_weights(actor, snapshot, torch.device("cpu")))
                self.assertEqual(weights.keys(), reference.keys())
                for name, value in weights.items():
                    self.assertEqual(value.shape, reference[name].shape)
                    self.assertEqual(value.dtype, dtype)
                    torch.testing.assert_close(value, reference[name])
                embedding_name = next(name for name in weights if name.endswith("embed_tokens.weight"))
                self.assertIs(weights[embedding_name], weights["lm_head.weight"])
                for name, value in actor.state_dict().items():
                    self.assertTrue(torch.equal(value, before[name]), name)

                # CPU tensors exercise the merge/name/shape/writeback path.
                # Actual FSDP DTensor layouts still require the cluster run.
                templates = cpu_sharded_state_dict(anchor)
                updates = {name: _anchor_shard(value, templates[name]) for name, value in weights.items()}
                set_model_state_dict(anchor, updates, options=StateDictOptions(full_state_dict=False, strict=True))
                for name, value in anchor.state_dict().items():
                    torch.testing.assert_close(value, reference[name])

    def test_update_merge_anchor_and_reload(self):
        torch.manual_seed(7)
        original = tiny_model().eval()
        ref = copy.deepcopy(original).requires_grad_(False)
        anchor = copy.deepcopy(ref)
        actor, counts = apply_full_lora(original, rank=2, alpha=4)
        actor.eval()
        inputs = {
            "input_ids": torch.tensor([[1, 44, 45, 47, 2, 3]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "pixel_values": torch.randn(4, 3 * 2 * 2 * 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "use_cache": False,
        }
        ref_state = {k: v.clone() for k, v in ref.state_dict().items()}
        frozen = {k: v.clone() for k, v in actor.named_parameters() if not v.requires_grad}
        self.assertTrue(all(counts[k] > 0 for k in ("linear", "conv3d", "embedding", "vectors")))
        torch.testing.assert_close(actor(**inputs).logits, ref(**inputs).logits)

        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            logits = actor(**inputs).logits
            logits.square().mean().backward()
            # Every trainable parameter participates in this multimodal graph.
            self.assertFalse([n for n, p in actor.named_parameters() if p.requires_grad and p.grad is None])
            optimizer.step()
        for name, param in actor.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(param, frozen[name]), name)

        before_export = {k: v.clone() for k, v in actor.state_dict().items()}
        effective = effective_lora_state_dict(actor, actor.state_dict())
        anchor.load_state_dict(effective, strict=True)
        torch.testing.assert_close(actor(**inputs).logits, anchor(**inputs).logits, rtol=1e-4, atol=1e-5)
        self.assertIs(anchor.get_input_embeddings().weight, anchor.get_output_embeddings().weight)
        for name, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, before_export[name]), name)

        # Model state includes the directly trained norm/bias values, unlike a
        # PEFT adapter-only export. Reload into the same parameterization.
        payload = io.BytesIO()
        torch.save(actor.state_dict(), payload)
        payload.seek(0)
        reloaded, _ = apply_full_lora(tiny_model(), rank=2, alpha=4)
        reloaded.load_state_dict(torch.load(payload, weights_only=True), strict=True)
        reloaded.eval()
        torch.testing.assert_close(actor(**inputs).logits, reloaded(**inputs).logits)

        # Simulate continuing the same Actor: neither frozen policy follows it.
        snapshot = anchor(**inputs).logits.detach().clone()
        optimizer.zero_grad()
        actor(**inputs).logits.square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(anchor(**inputs).logits, snapshot)
        for name, value in ref.state_dict().items():
            self.assertTrue(torch.equal(value, ref_state[name]), name)

    def test_uncovered_parameter_is_reported(self):
        model = tiny_model()
        model.extra_matrix = torch.nn.Parameter(torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "extra_matrix"):
            apply_full_lora(model, rank=2, alpha=4)


if __name__ == "__main__":
    unittest.main()
