"""Cut the Needle 3 base checkpoint down to an N-layer rung, saved as its own checkpoint.

`needle finetune` always trains on the full 20 layers, and `needle build --layers N` slices
afterwards. A LoRA learned with all 20 layers in place breaks once 16 of them are gone
(the 4- and 2-layer builds loop and score below the untuned base). Training on the rung
itself fixes that:

    python slice_base.py checkpoints/needle3.safetensors 4 base_4l.safetensors
    needle finetune data/train.jsonl --checkpoint base_4l.safetensors --out adapter_4l.safetensors
    needle build base_4l.safetensors --lora adapter_4l.safetensors --out duck_4l.cact
"""
import sys

from needle.model.checkpoints import read_checkpoint, write_checkpoint
from needle.model.finetune import rung
from needle.model.run import load_checkpoint


def main(src, layers, out):
    params, config, run = load_checkpoint(src, return_run=True)
    params, config = rung(params, config, int(layers))
    fmt = read_checkpoint(src)["format_version"]
    write_checkpoint(out, {"format_version": fmt, "config": dict(vars(config)), "params": params, "run": run})
    print(f"wrote {out}: {config.num_layers} layers")


if __name__ == "__main__":
    main(*sys.argv[1:4])
