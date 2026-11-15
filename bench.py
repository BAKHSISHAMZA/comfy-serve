"""Measure product_scene on your own GPU: send N jobs through comfy-serve, print render time and peak GPU memory.

Start ComfyUI and comfy-serve first, then in a second window:

    python bench.py                                   # a drawn stand-in product, 10 jobs
    python bench.py --image my_product.png            # your own product cut-out (PNG with transparency)
    python bench.py --checkpoint RealVisXL_V5.0_fp16.safetensors --n 10

The product is placed on a 1024 x 1024 transparent canvas, so the model paints the scene around it. Results are saved
to bench_out/. The last line is a row for the README table.

How to read the numbers:
- The first job also loads the checkpoint from disk, so it is reported on its own; the median is over the other jobs.
- GPU memory is what ComfyUI reports as in use on the whole GPU, so it includes the Windows desktop and any other app.
  The idle figure, taken before the first job, shows how much of the peak was already used by them.
"""
import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

SIZE = 1024


def stand_in_product():
    """A plain bottle shape, so the benchmark runs before you have product photos."""
    im = Image.new("RGBA", (400, 900), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((20, 260, 380, 900), radius=60, fill=(46, 92, 74, 255))      # body
    d.rectangle((140, 120, 260, 280), fill=(46, 92, 74, 255))                          # neck
    d.rounded_rectangle((120, 0, 280, 140), radius=18, fill=(218, 196, 150, 255))     # cap
    d.rectangle((60, 480, 340, 700), fill=(236, 232, 222, 255))                        # label
    return im


def on_canvas(path):
    if path:
        im = Image.open(path).convert("RGBA")
        alpha = im.getchannel("A")
        if alpha.getextrema()[0] == 255:
            sys.exit(f"{path} has no transparent pixels. Cut the product out first and save it as a PNG with transparency.")
        im = im.crop(alpha.getbbox())
    else:
        im = stand_in_product()
    scale = 0.6 * SIZE / max(im.size)                       # product fills 60% of the frame's longer side
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((SIZE - im.width) // 2, max(0, round(SIZE * 0.88) - im.height)))   # stands low in frame
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


def checkpoints(comfy):
    """Checkpoint files ComfyUI can see, or None if it does not say."""
    try:
        spec = httpx.get(f"{comfy}/object_info/CheckpointLoaderSimple", timeout=10).json()
        field = spec["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
        return field[0] if isinstance(field[0], list) else field[1]["options"]
    except Exception:
        return None


def gpu_used_mb(comfy):
    try:
        dev = httpx.get(f"{comfy}/system_stats", timeout=10).json()["devices"][0]
        return (dev["vram_total"] - dev["vram_free"]) / 2**20
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://127.0.0.1:8199", help="comfy-serve address")
    p.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI address (to list checkpoints)")
    p.add_argument("--image", help="product cut-out, PNG with transparency (default: a drawn stand-in)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--checkpoint", help="checkpoint file name (default: the workflow's default)")
    p.add_argument("--prompt", default="product on a marble kitchen counter, soft window light, "
                                       "professional product photography, photorealistic")
    a = p.parse_args()

    try:
        health = httpx.get(f"{a.api}/health", timeout=10).json()
    except httpx.TransportError:
        sys.exit(f"comfy-serve is not answering at {a.api}. Start it first: python -m comfy_serve")
    if health["comfyui"] != "up":
        sys.exit("comfy-serve is running but ComfyUI is not. Start ComfyUI (run_nvidia_gpu.bat) and try again.")

    ckpt = a.checkpoint or httpx.get(f"{a.api}/workflows").json()["product_scene"]["checkpoint"]["default"]
    have = checkpoints(a.comfy)
    if have is not None and ckpt not in have:
        sys.exit(f"ComfyUI has no checkpoint named {ckpt!r}. It has: {have or 'none'}\n"
                 f"Put one in ComfyUI\\models\\checkpoints, or pick one with --checkpoint <name>.")

    gpu = (health["gpu"] or "unknown GPU").split(" : ")[0]          # "cuda:0 NVIDIA GeForce RTX 5060 : cudaMallocAsync"
    gpu = gpu.split(" ", 1)[1] if gpu.startswith("cuda:") else gpu   # -> "NVIDIA GeForce RTX 5060"
    image = on_canvas(a.image)
    out = Path("bench_out")
    out.mkdir(exist_ok=True)
    (out / "input_canvas.png").write_bytes(image)
    idle = gpu_used_mb(a.comfy)
    print(f"GPU: {gpu}  |  checkpoint: {ckpt}  |  {a.n} jobs at {SIZE} x {SIZE}, {a.steps} steps")
    if idle is not None:
        print(f"GPU memory in use before the first job: {idle:.0f} MB")

    ids = []
    for seed in range(1, a.n + 1):
        inputs = {"prompt": a.prompt, "seed": seed, "steps": a.steps, "checkpoint": ckpt}
        r = httpx.post(f"{a.api}/jobs", data={"workflow": "product_scene", "inputs": json.dumps(inputs)},
                       files={"image": ("product.png", image, "image/png")}, timeout=60)
        if r.status_code != 202:
            sys.exit(f"job refused: {r.status_code} {r.text}")
        ids.append(r.json()["id"])

    jobs, t0 = {}, time.time()
    while len(jobs) < len(ids):
        for i, job_id in enumerate(ids, 1):
            if job_id in jobs:
                continue
            j = httpx.get(f"{a.api}/jobs/{job_id}", timeout=10).json()
            if j["status"] in ("done", "failed"):
                jobs[job_id] = j
                if j["status"] == "done":
                    (out / f"{i:02d}_seed{i}.png").write_bytes(httpx.get(a.api + j["outputs"][0], timeout=60).content)
                    print(f"job {i:2d}/{a.n}  done    {j['seconds']:6.1f} s   peak {j['peak_vram_mb'] or 0:5.0f} MB")
                else:
                    print(f"job {i:2d}/{a.n}  FAILED  {j['error']}")
        if time.time() - t0 > 900 * len(ids):
            sys.exit("timed out waiting for jobs")
        time.sleep(1)

    done = [jobs[i] for i in ids if jobs[i]["status"] == "done"]
    if not done:
        sys.exit("every job failed; the errors are above")
    first, rest = done[0], done[1:]
    warm = statistics.median(j["seconds"] for j in rest) if rest else first["seconds"]
    peak = max(j["peak_vram_mb"] or 0 for j in done)
    vram_gb = round(health["vram_total_mb"] / 1024) if health.get("vram_total_mb") else "?"

    print()
    print(f"done: {len(done)}/{a.n}   failed: {a.n - len(done)}")
    print(f"first job (also loads the checkpoint): {first['seconds']:.1f} s")
    print(f"median of the other jobs: {warm:.1f} s per image")
    print(f"peak GPU memory in use: {peak:.0f} MB" + (f" (idle before the first job: {idle:.0f} MB)" if idle else ""))
    print(f"images saved in {out.resolve()}")
    print()
    print("README row:")
    print(f"| {gpu} ({vram_gb} GB) | product_scene, {ckpt} | {SIZE} × {SIZE} | {a.steps} | "
          f"{warm:.1f} s (first job {first['seconds']:.1f} s, includes loading the model) | "
          f"{peak / 1024:.1f} GB (idle {idle / 1024:.1f} GB) |" if idle else
          f"| {gpu} ({vram_gb} GB) | product_scene, {ckpt} | {SIZE} × {SIZE} | {a.steps} | "
          f"{warm:.1f} s (first job {first['seconds']:.1f} s, includes loading the model) | {peak / 1024:.1f} GB |")


if __name__ == "__main__":
    main()
