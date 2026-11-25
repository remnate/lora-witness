# Loader evidence

The release check used CPython 3.12.16 on Linux x86-64, CPU Torch 2.8.0+cpu, torchvision 0.23.0+cpu, Diffusers 0.43.0, PEFT 0.22.0, Transformers 5.24.1, Accelerate 1.16.0 and safetensors 0.8.0. These are inference-loader pins. The runtime comparator imports none of the model-loader stack.

## Reproduce from the checkout

Use a fresh environment. The first step downloads the CPU wheels, and the shell script downloads the pinned sd-scripts source archive and SDXL Base UNet config into temporary storage. Model construction and adapter injection then run offline. The experiment uses no base checkpoint, GPU or account.

```sh
python3.12 -m venv .venv-oracle
. .venv-oracle/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu
python -m pip install -c scripts/oracle-linux-py312.txt '.[oracle]'
sh scripts/check-loaders.sh oracle-work
lora-witness compare examples/synthetic/kohya.safetensors oracle-work/converted.safetensors --native-unit-scale --mode exhaustive
```

Choose a new output directory. The oracle builds Kohya's original SDXL UNet and a Diffusers UNet from Stability AI's SDXL Base config on the meta device. The ordinary LoRA controls evaluate individual CPU modules with zero base parameters. The DoRA controls use nonzero native base weights. Checks observe actual forward hooks, resulting weights and untouched neighbors. The source uses sd-scripts create_network_from_weights for inference, apply_to and strict state loading on the normal fixture. A separate absent-alpha control verifies the exact missing alpha buffers and their rank fallback. The target uses StableDiffusionXLLoraLoaderMixin.lora_state_dict and load_lora_into_unet.

The four-layer fixture covers down, middle and up blocks, both attention types and all four projection types. Each layer has a distinct rank-two signature. The two loaders agreed bit-for-bit on 32 seeded FP32 vectors per active layer. All three adjacent sentinel outputs stayed zero. The maximum difference against the independent FP64 corner calculation was 1.669e-6. The loadable mutant changed only mid-block Q and halved every output there. The largest loader-output difference was 8.191748 on those probes. This differs from the exhaustive matrix-entry error of 3.5 because the coordinates describe different measurements. [Four-layer results](evidence/loader-results.json)

## Adapter-wide configuration controls

One native magnitude vector on mid-block V enabled DoRA on all four attached projections. With a unit first coefficient in each base row, a collinear mid-block Q update normalized away completely. The target produced zero difference from the zero-update Kohya source on every input basis vector at each projection. The installed CLI returned exit 2 with four pairs identified per file and 0/12 source keys and 0/9 native keys numerically compared. [CLI report](evidence/dora.json)

Adding the same V magnitude vector to the half-scale export changed the down-block K and up-block output deltas by 1.3333333 and 4.5120497 relative to Kohya on the first input basis vector. A separate mixed LoKr file attached only a Kronecker update to V, and a single UNet B bias enabled B biases on all four ordinary modules. The SAI Control LoRA guard follows the pinned loader's component-scoped marker and has a lightweight file regression. [Native control measurements](evidence/loader-results.json)

With the ordinary half-scale export plus a text-encoder bias, the pinned UNet loader preserved all four ordinary attachments and only mid-block Q differed from the source. The installed CLI returned exit 1, maximum coefficient error 3.5 and 8/9 native keys compared. [Ordinary mixed-input CLI report](evidence/mixed.json)

The oracle writes the two DoRA inputs for this installed-CLI reproduction. The command returns exit 2.

```sh
lora-witness compare oracle-work/zero-source.safetensors oracle-work/collinear-dora.safetensors --native-unit-scale --mode exhaustive --json
```

## Upstream conversion and inventory

For the upstream conversion, Diffusers remaps the synthetic Kohya file and injects PEFT modules. The harness extracts PEFT's actual state, multiplies each B by its attached PEFT scaling to request a unit-scale export, then calls Diffusers save_lora_weights. A fresh UNet reloads the resulting file. Additional real-loader controls verify a square transposed update and a rank-three padded export. CLI exhaustive comparison agrees on all entries. The checkout includes the resulting synthetic adapter in `examples/upstream/`.

A second experiment inventories each original UNet independently, selects alternate entries in sorted Kohya module order, and gives each active projection its own rank-one integer signature. Each of two loads activates 280 projections and leaves the other 280 as sentinels. Diffusers remaps and exports each original source using upstream library code, with scale already equal to one. Actual source and reloaded target forward hooks visit all 560 paths in each load. Each active signature must appear on the exact manifest path, with every inactive output zero. The CLI also checks each exported cohort using 32 probes per projection. Both returned maximum error zero. This exercises all 560 attachment identities on synthetic inputs. [Inventory results and hashes](evidence/inventory-results.json)

Earlier research supplied with the project reported three active projections and an FP64-reference error of 4.25e-8 on different weights. Its FP16/BF16 rounding study informed the tolerance profiles. The numbers above come from this repository's own fixtures and loader scripts. Performance on real publisher pairs and image quality have yet to be measured.

[sd-scripts inference network](https://github.com/kohya-ss/sd-scripts/blob/88ff6b5391e89822d0463df8769591273d0d28b1/networks/lora.py) · [SDXL Base UNet config](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/unet/config.json) · [Diffusers loader](https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/loaders/lora_pipeline.py) · [PEFT injection](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/lora/layer.py)
