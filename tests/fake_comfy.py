"""A stand-in for ComfyUI, so the job server can be tested on a machine without a GPU.

It speaks the same HTTP API (upload, prompt, queue, history, view, system_stats) and behaves the same way in the
parts that matter for reliability: one prompt at a time, FIFO, results appear in /history only when finished,
graphs that reference a missing input image are rejected with a 400 like ComfyUI does. "Rendering" is a sleep plus
a small PNG whose colour comes from the seed, so outputs are deterministic and checkable.

Run alone:  python tests/fake_comfy.py --port 8188 --delay 1.0
"""
import argparse
import io
import queue
import threading
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

VRAM_TOTAL = 8 * 2**30


def make_fake(workdir, delay=0.5):
    workdir = Path(workdir)
    inputs, outputs = workdir / "input", workdir / "output"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    app = FastAPI()
    state = {"pending": queue.Queue(), "pending_ids": [], "running": None, "history": {}, "executed": [],
             "vram_used": 600 * 2**20, "delay": delay}
    lock = threading.Lock()

    def render_loop():
        while True:
            pid, graph = state["pending"].get()
            with lock:
                state["pending_ids"].remove(pid)
                state["running"] = pid
                state["vram_used"] = 6800 * 2**20                  # what an SDXL render roughly holds
            start = int(time.time() * 1000)
            time.sleep(state["delay"])
            out = {}
            seed = next((n["inputs"].get("seed", 0) for n in graph.values() if n["class_type"] == "KSampler"), 0)
            for node_id, node in graph.items():
                if node["class_type"] == "SaveImage":
                    name = f"fake_{pid[:8]}_{node_id}.png"
                    colour = (seed * 37 % 256, seed * 91 % 256, seed * 53 % 256)
                    Image.new("RGB", (64, 80), colour).save(outputs / name)
                    out[node_id] = {"images": [{"filename": name, "subfolder": "", "type": "output"}]}
            end = int(time.time() * 1000)
            with lock:
                state["history"][pid] = {
                    "prompt": [0, pid, graph, {}, list(out)],
                    "outputs": out,
                    "status": {"status_str": "success", "completed": True,
                               "messages": [["execution_start", {"prompt_id": pid, "timestamp": start}],
                                            ["execution_success", {"prompt_id": pid, "timestamp": end}]]}}
                state["executed"].append(pid)
                state["running"] = None
                state["vram_used"] = 600 * 2**20

    threading.Thread(target=render_loop, daemon=True).start()

    @app.post("/upload/image")
    async def upload(image: UploadFile, request: Request):
        data = await image.read()
        (inputs / image.filename).write_bytes(data)
        return {"name": image.filename, "subfolder": "", "type": "input"}

    @app.post("/prompt")
    async def prompt(request: Request):
        body = await request.json()
        graph = body["prompt"]
        for node_id, node in graph.items():
            if node["class_type"] == "LoadImage" and not (inputs / node["inputs"]["image"]).exists():
                return JSONResponse(status_code=400, content={
                    "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                    "node_errors": {node_id: {"errors": [{"message": "Invalid image file",
                                                          "details": f"image - {node['inputs']['image']}"}]}}})
        pid = str(uuid.uuid4())
        with lock:
            state["pending_ids"].append(pid)
        state["pending"].put((pid, graph))
        return {"prompt_id": pid, "number": len(state["executed"]), "node_errors": {}}

    @app.get("/queue")
    def get_queue():
        with lock:
            running = [[0, state["running"], {}, {}, []]] if state["running"] else []
            pending = [[i, pid, {}, {}, []] for i, pid in enumerate(state["pending_ids"])]
        return {"queue_running": running, "queue_pending": pending}

    @app.get("/history/{pid}")
    def history(pid: str):
        with lock:
            return {pid: state["history"][pid]} if pid in state["history"] else {}

    @app.get("/view")
    def view(filename: str, subfolder: str = "", type: str = "output"):
        path = (outputs if type == "output" else inputs) / filename
        if not path.exists():
            raise HTTPException(404)
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/system_stats")
    def stats():
        return {"system": {"os": "fake"},
                "devices": [{"name": "cuda:0 Fake RTX 5060", "type": "cuda", "index": 0,
                             "vram_total": VRAM_TOTAL, "vram_free": VRAM_TOTAL - state["vram_used"]}]}

    @app.get("/_executed")
    def executed():                                               # test hook: which prompts actually "rendered"
        return {"executed": state["executed"]}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8188)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--dir", default="fake_comfy_data")
    a = ap.parse_args()
    uvicorn.run(make_fake(a.dir, a.delay), host="127.0.0.1", port=a.port, log_level="warning")
