# LoRA Witness

Compare the numerical effect of a Kohya SDXL Base UNet LoRA and its native Diffusers export, projection by projection, on a local CPU. A conversion can load cleanly and still change the model update. LoRA Witness identifies the affected projection and shows how much of each file it checked.

The v0.1 allowlist covers 560 separate linear attention projections. Each attachment path was exercised through both pinned loaders with synthetic weights. Files can cover a subset. The runtime needs NumPy, safetensors and threadpoolctl. Model checkpoints, GPUs and uploads are unnecessary.

## Install from the checkout

Run these commands in the repository root with Python 3.12. They build a source distribution and wheel, then install the wheel into a second, clean environment.

```sh
python3.12 -m venv .venv-build
.venv-build/bin/python -m pip install build==1.3.0
.venv-build/bin/python -m build
python3.12 -m venv .venv
.venv/bin/python -m pip install dist/lora_witness-0.1.0-py3-none-any.whl
. .venv/bin/activate
```

## Try the installed wheel

Generate the four small synthetic files in a fresh directory. The wheel contains the generator.

```sh
lora-witness fixture demo
lora-witness compare demo/kohya.safetensors demo/native.safetensors --native-unit-scale
lora-witness compare demo/kohya.safetensors demo/half.safetensors --native-unit-scale --mode exhaustive
```

The first comparison returns `sampled_agreement` (exit 0), with 12/12 Kohya keys and 8/8 native keys compared. The second returns `fail` (exit 1). Only `unet.mid_block.attentions.0.transformer_blocks.9.attn1.to_q` fails. Both files store FP32. Source scale is 0.5, target scale is 1, and the largest coefficient error is 3.5 at zero-based [1,2]. The reference peak is 7, tolerance is 0.0000700001, and relative Frobenius error is 0.5.

The active 2×3 corner of that source effect is `[[2,4,-2],[-1,0,7]]`. The equivalent target folds alpha/rank into its B matrix. Halving that B gives `[[1,2,-1],[-0.5,0,3.5]]`. All other entries in the projection are zero.

To see partial coverage, run the mutant with one additional text-encoder tensor. Its report retains the numerical failure, lists the unsupported key and shows 8/9 native keys compared.

```sh
lora-witness compare demo/kohya.safetensors demo/mixed.safetensors --native-unit-scale --mode exhaustive --json
```

## Check a converter export

Supply the original Kohya file first. Declare unit native scaling only when the exporter folded any source scaling into the factors and requires no other configuration. Run from the checkout to try the same command against a file exported by the pinned Diffusers/PEFT stack.

```sh
lora-witness compare examples/synthetic/kohya.safetensors examples/upstream/converted.safetensors --native-unit-scale --mode exhaustive
```

Sampled mode evaluates 32 seeded Gaussian input vectors per layer. Exhaustive mode visits every supported matrix entry in bounded row blocks. Both use FP64 arithmetic and apply tolerances independently per projection. Use `--profile fp16` for explicitly permitted FP16 storage rounding. Use `--atol`, `--rtol`, `--seed` and `--rows` to control the test. Use `--json` for CI, `lora-witness schema` for the report schema, and `lora-witness manifest` for exact projection identities and shapes.

Exit 0 indicates agreement under the reported mode and profile. Exit 1 indicates a demonstrated mismatch or invalid input. Exit 2 indicates an inconclusive result or CLI usage error. Unsupported keys, partial/mixed alpha, unknown configuration, old Diffusers spellings and empty coverage can make a comparison inconclusive. A UNet DoRA, LoKr, LoRA B bias or SAI Control LoRA marker leaves every UNet effect unassessed while preserving the file inventories. Input limits include 64 MiB and 8 million tensor elements per file. Agreement describes the declared adapter operator on covered projections. Finite probes can miss differences, and image quality and rendering are outside this check.

- [Sampled agreement report](docs/evidence/sampled.json) and [exhaustive mixed-input failure](docs/evidence/mixed.json)
- [File contract, tolerances and report fields](docs/contract.md)
- [Pinned loader reproduction and measurements](docs/loaders.md)
- [Contribution guide](CONTRIBUTING.md)
- [Release announcement and pilot plan](docs/launch.md)
- [MIT license](LICENSE) and [third-party notices](NOTICE)
