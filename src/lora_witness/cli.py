import argparse
import json
from importlib.resources import files

from .compare import compare
from .fixture import generate
from .manifest import MANIFEST, MANIFEST_HASH
from .numeric import PROFILES


def human(r):
    lines = [r["verdict"], f"{r['tolerance_profile']}  atol={r['atol']:g}  rtol={r['rtol']:g}"]
    for side, f in r["files"].items():
        c = f["coverage"]
        if c["total_keys"] is None:
            lines.append(f"{side}: inventory unavailable")
        else:
            lines.append(
                f"{side}: {c['compared_keys']}/{c['total_keys']} keys, "
                f"{c['compared_parameters']}/{c['total_parameters']} parameters"
            )
        lines.append(f"  sha256={f['sha256'] or 'unavailable'}")
        for category in (
            "missing_keys",
            "extra_keys",
            "unsupported_keys",
            "unpaired_keys",
            "ambiguous_keys",
            "configuration_issues",
            "errors",
        ):
            lines.extend(f"  {category}: {value!a}" for value in f[category])
    for path, layer in r["layers"].items():
        lines.append(
            f"{path}  {layer['mode']} {layer['status']}  E={layer['E']:.6g} "
            f"M={layer['M']:.6g}  worst={layer['worst_coordinate']} "
            f"relF={layer['relative_frobenius']}  limit={layer['threshold']:.6g}"
        )
        for side in ("kohya", "native"):
            lines.append(f"  {side} {layer[side]['dtypes']}  scale={layer[side]['scale']:g}")
    for a, b in r["likely_swaps"]:
        lines.append(f"Likely swap: {a} and {b}")
    if r["reasons"]:
        lines.append(", ".join(r["reasons"]))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare full-extent SDXL Base UNet LoRA projection effects on CPU."
    )
    parser.add_argument("--version", action="version", version="LoRA Witness 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="compare local safetensors files")
    c.add_argument("source", help="Kohya SDXL Base UNet adapter")
    c.add_argument("target", help="native Diffusers PEFT-key adapter")
    c.add_argument(
        "--native-unit-scale",
        dest="declared",
        action="store_true",
        help="declare a native export with unit effective scaling and no external configuration",
    )
    c.add_argument("--mode", choices=("sampled", "exhaustive"), default="sampled")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--rows", type=int, default=64, help="exhaustive block rows (1 to 256)")
    c.add_argument(
        "--profile", choices=PROFILES, default="fp32", help="permitted factor precision and tolerance"
    )
    c.add_argument("--atol", type=float, help="override absolute tolerance (0 to 1)")
    c.add_argument("--rtol", type=float, help="override relative tolerance (0 to 1)")
    c.add_argument("--json", action="store_true", help="emit report schema 1.0")
    sub.add_parser("manifest", help="print all supported identities and shapes")
    sub.add_parser("schema", help="print the JSON report schema")
    sub.add_parser("fixture", help="generate four synthetic local adapters").add_argument("directory")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "schema":
        print(files("lora_witness").joinpath("report.schema.json").read_text())
    elif command == "manifest":
        print(json.dumps({"sha256": MANIFEST_HASH, "projections": MANIFEST}, indent=2))
    elif command == "fixture":
        try:
            generate(args["directory"])
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    else:
        as_json = args.pop("json")
        try:
            r = compare(**args)
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(r, indent=2, allow_nan=False) if as_json else human(r))
        return {"fail": 1, "inconclusive": 2}.get(r["verdict"], 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
