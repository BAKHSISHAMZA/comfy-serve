# comfy-serve

Run a ComfyUI workflow as a job API: your app sends inputs over HTTP, gets a job id back at once, and downloads the result when it is done. Jobs live in a small database, so a crash or restart loses nothing and never renders the same job twice.

```
your app ──POST /jobs──▶ comfy-serve ──/prompt──▶ ComfyUI (your GPU)
         ◀─GET /jobs/id─   job store   ◀─/history─
                          (SQLite)
```

## Why this exists

ComfyUI is excellent at running a graph. It is not a service:
- **Inputs are node ids.** Its API takes the whole graph, so your app has to know that the prompt is `node 3, field text`. Edit the workflow and every client breaks.
- **Restarts lose work.** If whatever submitted the job dies, nothing remembers what was in flight.
- **There are no numbers.** There is no per-job record of render time or GPU memory, so you can't price a job or size a GPU.

comfy-serve adds those three things and nothing else:

| Problem | What comfy-serve does |
|---|---|
| Graph internals leak to clients | Each workflow ships with a small map of **named inputs** (`prompt`, `image`, `seed`…). Unknown names, missing inputs and wrong types are refused with a 400 before anything reaches the GPU. |
| Crashes lose or duplicate work | Every job is a row in SQLite. On start, each job left `running` is checked against ComfyUI: finished → its output is collected; still rendering → it is waited for; unknown to ComfyUI → it goes back in the queue. |
| No numbers | Every job records its **render time** (from ComfyUI's own timestamps) and the **peak GPU memory** seen while it ran. |

One job runs at a time, because there is one GPU; ComfyUI would queue the second anyway, and it keeps the numbers per job honest.

## Run it

1. Start ComfyUI as usual (on Windows, `run_nvidia_gpu.bat`).
2. Start comfy-serve:

   ```
   pip install -r requirements.txt
   python -m comfy_serve --comfy http://127.0.0.1:8188 --port 8199
   ```

   The ComfyUI desktop app listens on port 8000 instead, so there use `--comfy http://127.0.0.1:8000`.

3. Send a job:

   ```
   curl -F workflow=product_scene -F 'inputs={"prompt": "on a marble kitchen counter, soft window light"}' \
        -F image=@product_on_transparent_canvas.png http://127.0.0.1:8199/jobs
   # {"id": "3f2a9c1b7d4e", "status": "queued", "url": "/jobs/3f2a9c1b7d4e"}

   curl http://127.0.0.1:8199/jobs/3f2a9c1b7d4e
   # {"status": "done", "seconds": ..., "peak_vram_mb": ..., "outputs": ["/jobs/3f2a9c1b7d4e/outputs/0"]}
   ```

| Endpoint | |
|---|---|
| `GET /health` | ComfyUI up or down, GPU name and VRAM, jobs per state |
| `GET /workflows` | each workflow and the inputs it takes |
| `POST /jobs` | multipart: `workflow`, `inputs` (JSON), one file per image input; returns 202 and the job id |
| `GET /jobs/{id}` | status, render seconds, peak VRAM, output URLs |
| `GET /jobs/{id}/outputs/{n}` | download an output |

## Add your own workflow

1. In ComfyUI, build the workflow, then use **Save (API Format)**. Put the file in `workflows/<name>.json`.
2. Next to it, write `workflows/<name>.inputs.json`, which names the inputs clients may set:

   ```json
   {
     "inputs": {
       "image":  {"node": "2", "field": "image", "type": "image"},
       "prompt": {"node": "3", "field": "text",  "type": "text"},
       "seed":   {"node": "6", "field": "seed",  "type": "int", "default": "random"}
     },
     "outputs": ["8"]
   }
   ```

   Types are `text`, `int`, `float` and `image`. An input with a `default` is optional. `"random"` picks a new seed for every job. The map is checked against the graph at startup, so a wrong node id fails immediately, not halfway through a batch.

## Included workflow: `product_scene`

Paints a scene **around** a product without drawing the product. Send the product already placed on a transparent canvas: ComfyUI's `LoadImage` turns the transparent area into the mask, and `InpaintModelConditioning` with `noise_mask` keeps the sampler off the product. Core ComfyUI nodes only; the model is any SDXL checkpoint (default `sd_xl_base_1.0.safetensors`).

The saved image still passes the product through the VAE, so its pixels shift slightly: a mean colour difference of ΔE 2.3 inside the product on the benchmark runs. That is why this is only the scene stage of product-lock, a separate pipeline that then pastes the original product pixels back and checks the colour difference.

## Tests: every one passes or fails, nothing judged by eye

The tests run against `tests/fake_comfy.py`, a stand-in that speaks ComfyUI's HTTP API, renders one prompt at a time and needs no GPU:

```
python -m pytest -q tests
```

| Check | Result |
|---|---|
| 10 jobs in → 10 outputs out, each with its own seed's image, a render time and a peak VRAM figure | pass |
| Server killed (`SIGKILL`) with jobs in flight, then restarted: all 10 finish, and the stand-in rendered exactly 10 times (the in-flight job was collected, not re-rendered) | pass |
| Server **and** ComfyUI killed mid-render, ComfyUI restarted empty: the lost job is put back in the queue and all 5 finish | pass |
| ComfyUI down when jobs arrive: they wait in the queue and run once it is back | pass |
| Bad requests (unknown workflow, unknown input, missing image, wrong type, missing prompt, broken JSON) refused with a 400 and a reason; nothing queued | pass |

## Measured on real hardware

To measure your own GPU, keep ComfyUI and comfy-serve running and, in a second window:

```
python bench.py --checkpoint <your SDXL checkpoint file> [--image your_product_cutout.png]
```

It sends 10 jobs, saves the images to `bench_out/`, and prints a row for this table. The first job also loads the checkpoint, so it is reported separately; GPU memory is the whole GPU's use, so the idle figure before the first job (the desktop and other apps) is printed next to the peak. On Windows this also saves you from curl's quoting rules in Command Prompt.

| GPU | Workflow | Resolution | Steps | Render time per image | Peak VRAM |
|---|---|---|---|---|---|
| RTX 5060 (8 GB), ComfyUI 0.28 desktop app, Windows | product_scene, RealVisXL V5.0 (SDXL) | 1024 × 1024 | 30 | **14.9 s** median (first job 52.9 s: it also loads the model) | **7.5 GB** in use at peak, 1.1 GB of it the Windows desktop |

10 of 10 jobs done, 0 failed, measured with `bench.py` on 15 November 2026. Jobs 2–10 took between 14.8 and 15.0 s.

## Layout

```
comfy_serve/
  api.py        HTTP endpoints
  worker.py     runs one job at a time; crash recovery
  comfy.py      the ComfyUI HTTP client (upload, prompt, queue, history, view, system_stats)
  store.py      SQLite job store and its state machine
  workflows.py  named-input maps and validation
workflows/      product_scene.json + product_scene.inputs.json
tests/          fake_comfy.py + end-to-end tests
bench.py        measure render time and GPU memory on your own GPU
```

MIT licence.
