"""The HTTP API your app calls.

    GET  /health                    is ComfyUI reachable, which GPU, how many jobs in each state
    GET  /workflows                 the workflows this server runs and the inputs each one takes
    POST /jobs                      multipart form: workflow=<name>, inputs=<JSON of named inputs>, one file field
                                    per image input. Returns the job id at once; the render happens in the background.
    GET  /jobs/{id}                 status, timing, peak VRAM, and output URLs once done
    GET  /jobs/{id}/outputs/{n}     download output n

Example:
    curl -F workflow=product_scene -F 'inputs={"prompt": "on a marble counter"}' -F image=@bottle.png \
         http://localhost:8199/jobs
"""
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .workflows import WorkflowError


def make_app(store, comfy, workflows, data_dir):
    app = FastAPI(title="comfy-serve", version="0.1")
    uploads = Path(data_dir) / "uploads"

    @app.get("/health")
    def health():
        dev = comfy.device()
        return {"comfyui": "up" if dev else "down",
                "gpu": dev.get("name") if dev else None,
                "vram_total_mb": round(dev["vram_total"] / 2**20) if dev else None,
                "jobs": store.counts()}

    @app.get("/workflows")
    def list_workflows():
        return {name: wf.describe() for name, wf in workflows.items()}

    @app.post("/jobs", status_code=202)
    async def submit(request: Request):
        form = await request.form()
        name = form.get("workflow")
        wf = workflows.get(name)
        if wf is None:
            raise HTTPException(400, f"unknown workflow {name!r}; available: {sorted(workflows)}")
        try:
            values = json.loads(form.get("inputs") or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"inputs is not valid JSON: {e}")
        missing = [k for k in wf.image_inputs() if k not in form]
        if missing:
            raise HTTPException(400, f"missing image file(s): {missing}")
        try:                                                       # validate now, before anything is stored
            wf.build(values, {k: "check.png" for k in wf.image_inputs()})
        except WorkflowError as e:
            raise HTTPException(400, str(e))

        # Files go to disk first, the job row second: the worker must never see a queued job whose files are missing.
        job_id = uuid.uuid4().hex[:12]
        files = {}
        for key in wf.image_inputs():
            upload = form[key]
            dest = uploads / job_id / f"{key}{Path(upload.filename or '.png').suffix or '.png'}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            files[key] = str(dest)
        store.create(name, values, files=files, job_id=job_id)
        return {"id": job_id, "status": "queued", "url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}")
    def status(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return {"id": job_id, "workflow": job["workflow"], "status": job["status"], "error": job["error"],
                "attempts": job["attempts"], "seconds": job["seconds"], "peak_vram_mb": job["peak_vram_mb"],
                "outputs": [f"/jobs/{job_id}/outputs/{i}" for i in range(len(job["outputs"]))]}

    @app.get("/jobs/{job_id}/outputs/{n}")
    def output(job_id: str, n: int):
        job = store.get(job_id)
        if job is None or job["status"] != "done" or not 0 <= n < len(job["outputs"]):
            raise HTTPException(404, "no such output")
        return FileResponse(job["outputs"][n])

    return app
