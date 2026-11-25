# Comparison contract, v0.1.0

The component is the SDXL Base UNet at unit adapter strength. Run `lora-witness manifest` for the complete allowlist. Each record specifies a Kohya stem, native stem and full output × input dimensions. Only separate, unsliced linear attention projections count. A partial adapter may omit any projection on both sides.

The manifest contains 560 verified identities. The four-layer demonstration exercises these entries.

| Block | Transformer | Projection | Output × input |
|---|---:|---|---|
| down_blocks.1.attentions.0 | 0 | attn2.to_k | 640 × 2048 |
| mid_block.attentions.0 | 9 | attn1.to_q | 1280 × 1280 |
| mid_block.attentions.0 | 9 | attn1.to_v | 1280 × 1280 |
| up_blocks.0.attentions.1 | 3 | attn2.to_out.0 | 1280 × 1280 |

Kohya keys use the SGM block family and `.lora_down.weight`, `.lora_up.weight` and optional scalar `.alpha`. Native keys begin with `unet.` and end in `.lora_A.weight` or `.lora_B.weight`. Both shapes must match (rank,input) and (output,rank). Each file needs homogeneous rank across valid layers. The files may have different ranks. All supported source alphas must be absent or share one finite positive value. The caller declares the target to be a unit-scale native export. Headers may be empty or carry only `format=pt`. Sibling adapter_config.json, config.json, model_index.json, stem.json or filename.safetensors.json make configuration inconclusive. External or renamed configurations require exporter knowledge. Use the declaration only when you know the profile.

The native SDXL loader filters keys by component before choosing a UNet adapter configuration. A single UNet DoRA magnitude vector, LoKr factor, LoRA B bias or SAI Control LoRA marker makes the ordinary product assessment inconclusive for the entire UNet. This includes legacy up-bias keys that can convert to a B bias. These markers also apply on unlisted UNet paths. The report records the configuration issue, paired factors, any missing/extra keys and complete input denominators. Both files have zero numerically compared keys. A text-encoder marker is outside the UNet component and allows a numerical assessment of ordinary UNet pairs. [Pinned component filtering and tuner selection](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/loaders/peft.py) · [Pinned configuration inference](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/utils/peft_utils.py)

Compute the FP64 effect as `scale * (B @ A)`. Source scale is alpha/rank or 1 when all alphas are absent. Target scale is 1. Each layer must satisfy `E <= atol + rtol*M` independently. E is the greatest absolute difference at a tested coordinate. M is the largest absolute source effect. Relative Frobenius error divides the error norm by the source norm across all tested entries (null for zero denominator). The maximum criterion can mask relatively large differences on small entries. Use lower tolerances or exact zeros for a stricter check.

| Profile | Permitted factor storage | atol | rtol |
|---|---|---:|---:|
| fp32-v1 | FP32 | 1e-10 | 1e-5 |
| fp16-v1 (explicit opt-in) | FP32 or FP16 | 1e-9 | 2e-3 |

The profiles originate in the supplied three-projection research. The included synthetic fixture uses exactly representable values. The CLI does not evaluate BF16, runtime GPU kernels, images or end-to-end pipelines.

Sampled mode draws 32 standard normal FP64 columns using NumPy PCG64. SHA-256 of the UTF-8 decimal seed, slash and full native path sets the per-layer seed (first 16 digest bytes, little endian). Both sides use the same matrix. Its recorded hash covers little-endian FP64 C-order bytes. A worst coordinate is [output,probe] in sampled mode or [output,input] in exhaustive mode, zero based. A near-zero sampled source triggers exhaustive fallback. Below or at the resolution floor M ≤ atol/rtol, nonzero results within tolerance are inconclusive. Exact zero on both sides permits zero-effect agreement. With a pure absolute tolerance the resolution floor is atol. A finite sample can miss an update orthogonal to the probes.

## Report schema 1.0

Run `lora-witness schema` for the machine-readable JSON Schema. Each report carries the manifest hash, file hashes, settings, layer statistics and both file inventories. Each file lists every parsed key, dimensions and storage dtype. Matched keys are recognized by spelling. Compared keys are those included in a numerical check, including failed numerical checks. Coverage counts keys and stored scalar parameters on each side separately, including scalar alpha. Verified attachments list compared identities with fixture loader evidence. The CLI itself does no loader injection. Missing and extra keys are relative to the peer's recognized manifest paths. A projection can be present but lack a factor. Unpaired keys identify the surviving parts. Unsupported or ambiguous tensors remain unassessed. When parsing fails before a safe inventory exists, counts are null and errors explain why. Hashes are absent if the snapshot could not be read.

Exit 0 means sampled or exhaustive agreement on nonempty, fully covered inputs. Exit 1 means malformed input, a known projection-set difference or demonstrated numerical disagreement. Exit 2 means inconclusive coverage/configuration/precision or a command usage error. Numerical failure survives unrelated unsupported tensors. Likely swaps require unique reciprocal bitwise matches of full-effect hashes in a fully compared exhaustive report. Worst-coordinate ties use the first row-major maximum.

## Resource bounds

Each local regular file is copied into a private temporary directory before hashing and decoding. The limit per file is 64 MiB, 512 KiB of header, 4,096 tensor keys and 8 million stored scalar elements. Supported rank is 1–64. At most 560 recognized projection pairs are compared, with at most 1,280 output rows and 2,048 inputs. The element cap also bounds exhaustive work to under 6.4 billion multiply-adds per file: each rank contributes output × input operations and stores output + input elements. The default row block is 64, selectable from 1 to 256. Factors are decoded by layer and BLAS uses one thread during comparison. No network access or checkpoint is required. Mixed SGM/native names, legacy processor serialization, slices, fused QKV, convolutions, biases, DoRA, LyCORIS, text encoders, unknown metadata and unlisted projections prevent full agreement.
