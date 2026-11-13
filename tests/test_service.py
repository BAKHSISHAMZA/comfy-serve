"""End-to-end checks against the stand-in ComfyUI. Each one is pass/fail:

1. ten jobs in, ten outputs out, each with a render time and a peak-VRAM figure
2. kill the server with jobs in flight, start it again: every job finishes, and no job is rendered twice
3. ComfyUI down when jobs arrive: they wait, and run once ComfyUI is back
4. bad requests are refused with a reason before anything is queued
"""
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.TransportError:
            time.sleep(0.1)
    raise TimeoutError(url)


def start_fake(port, workdir, delay):
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'tests')!r}); import uvicorn, fake_comfy; "
            f"uvicorn.run(fake_comfy.make_fake({str(workdir)!r}, {delay}), host='127.0.0.1', port={port}, "
            f"log_level='warning')")
    p = subprocess.Popen([sys.executable, "-c", code])
    wait_http(f"http://127.0.0.1:{port}/system_stats")
    return p


def start_server(port, comfy_port, data):
    p = subprocess.Popen([sys.executable, "-m", "comfy_serve", "--port", str(port), "--data", str(data),
                          "--comfy", f"http://127.0.0.1:{comfy_port}"], cwd=ROOT,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{port}/health")
    return p


def product_png():
    """A 'product' on a transparent canvas: opaque rectangle in the middle, transparent around it."""
    im = Image.new("RGBA", (64, 80), (0, 0, 0, 0))
    im.paste((180, 140, 90, 255), (20, 20, 44, 64))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def submit(api, seed, **extra):
    r = httpx.post(f"{api}/jobs", data={"workflow": "product_scene",
                                        "inputs": json.dumps({"prompt": "on a marble counter", "seed": seed, **extra})},
                   files={"image": ("product.png", product_png(), "image/png")})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def wait_all(api, ids, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = [httpx.get(f"{api}/jobs/{i}").json() for i in ids]
        if all(j["status"] in ("done", "failed") for j in jobs):
            return jobs
        time.sleep(0.3)
    raise TimeoutError([j["status"] for j in jobs])


@pytest.fixture
def ports():
    return free_port(), free_port()


def test_ten_jobs_ten_outputs(tmp_path, ports):
    comfy_port, api_port = ports
    fake = start_fake(comfy_port, tmp_path / "comfy", delay=0.2)
    server = start_server(api_port, comfy_port, tmp_path / "data")
    api = f"http://127.0.0.1:{api_port}"
    try:
        ids = [submit(api, seed) for seed in range(10)]
        jobs = wait_all(api, ids)
        assert [j["status"] for j in jobs] == ["done"] * 10
        for seed, j in enumerate(jobs):
            assert len(j["outputs"]) == 1 and j["seconds"] > 0 and j["peak_vram_mb"] > 6000
            png = Image.open(io.BytesIO(httpx.get(api + j["outputs"][0]).content))
            assert png.getpixel((0, 0)) == (seed * 37 % 256, seed * 91 % 256, seed * 53 % 256)   # its own seed
        assert len(httpx.get(f"http://127.0.0.1:{comfy_port}/_executed").json()["executed"]) == 10
    finally:
        server.kill(); fake.kill()


def test_restart_loses_nothing_and_renders_nothing_twice(tmp_path, ports):
    comfy_port, api_port = ports
    fake = start_fake(comfy_port, tmp_path / "comfy", delay=0.6)
    data = tmp_path / "data"
    server = start_server(api_port, comfy_port, data)
    api = f"http://127.0.0.1:{api_port}"
    try:
        ids = [submit(api, seed) for seed in range(10)]
        while sum(httpx.get(f"{api}/jobs/{i}").json()["status"] == "done" for i in ids) < 3:
            time.sleep(0.1)
        server.send_signal(signal.SIGKILL)                         # no clean shutdown: a crash
        server.wait()
        server = start_server(api_port, comfy_port, data)
        jobs = wait_all(api, ids)
        assert [j["status"] for j in jobs] == ["done"] * 10
        executed = httpx.get(f"http://127.0.0.1:{comfy_port}/_executed").json()["executed"]
        assert len(executed) == 10, f"{len(executed)} renders for 10 jobs"
    finally:
        server.kill(); fake.kill()


def test_both_crash_and_comfyui_forgets(tmp_path, ports):
    """Server and ComfyUI both die mid-render. The new ComfyUI has an empty history, so the job that was rendering
    is unknown to it: the server must put that job back in the queue instead of waiting forever."""
    comfy_port, api_port = ports
    fake = start_fake(comfy_port, tmp_path / "comfy1", delay=0.6)
    data = tmp_path / "data"
    server = start_server(api_port, comfy_port, data)
    api = f"http://127.0.0.1:{api_port}"
    try:
        ids = [submit(api, seed) for seed in range(5)]
        while sum(httpx.get(f"{api}/jobs/{i}").json()["status"] == "done" for i in ids) < 2:
            time.sleep(0.1)
        server.send_signal(signal.SIGKILL); fake.send_signal(signal.SIGKILL)
        server.wait(); fake.wait()
        fake = start_fake(comfy_port, tmp_path / "comfy2", delay=0.2)    # fresh ComfyUI, empty history
        server = start_server(api_port, comfy_port, data)
        jobs = wait_all(api, ids)
        assert [j["status"] for j in jobs] == ["done"] * 5
        assert any(j["attempts"] == 2 for j in jobs), "the in-flight job should have been requeued once"
    finally:
        server.kill(); fake.kill()


def test_comfyui_down_then_up(tmp_path, ports):
    comfy_port, api_port = ports
    server = start_server(api_port, comfy_port, tmp_path / "data")     # ComfyUI not started yet
    api = f"http://127.0.0.1:{api_port}"
    fake = None
    try:
        assert httpx.get(f"{api}/health").json()["comfyui"] == "down"
        ids = [submit(api, seed) for seed in range(3)]
        time.sleep(2)
        assert all(httpx.get(f"{api}/jobs/{i}").json()["status"] in ("queued", "running") for i in ids)
        fake = start_fake(comfy_port, tmp_path / "comfy", delay=0.2)
        jobs = wait_all(api, ids, timeout=40)
        assert [j["status"] for j in jobs] == ["done"] * 3
    finally:
        server.kill()
        if fake:
            fake.kill()


def test_bad_requests_are_refused(tmp_path, ports):
    comfy_port, api_port = ports
    fake = start_fake(comfy_port, tmp_path / "comfy", delay=0.1)
    server = start_server(api_port, comfy_port, tmp_path / "data")
    api = f"http://127.0.0.1:{api_port}"
    img = {"image": ("p.png", product_png(), "image/png")}
    try:
        cases = [
            ({"workflow": "nope", "inputs": "{}"}, img, "unknown workflow"),
            ({"workflow": "product_scene", "inputs": '{"prompt": "x", "colour": "red"}'}, img, "unknown inputs"),
            ({"workflow": "product_scene", "inputs": '{"prompt": "x"}'}, None, "missing image"),
            ({"workflow": "product_scene", "inputs": '{"prompt": "x", "steps": "many"}'}, img, "must be int"),
            ({"workflow": "product_scene", "inputs": "{}"}, img, "missing input 'prompt'"),
            ({"workflow": "product_scene", "inputs": "not json"}, img, "not valid JSON"),
        ]
        for data, files, why in cases:
            r = httpx.post(f"{api}/jobs", data=data, files=files)
            assert r.status_code == 400 and why in r.text, (why, r.status_code, r.text)
        assert httpx.get(f"{api}/health").json()["jobs"] == {}
    finally:
        server.kill(); fake.kill()
