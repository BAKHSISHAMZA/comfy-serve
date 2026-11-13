"""python -m comfy_serve [--comfy http://127.0.0.1:8188] [--port 8199] [--data ./data] [--workflows ./workflows]

One command starts the API and the worker in the same process. ComfyUI runs separately (run_nvidia_gpu.bat on
Windows, or `python main.py`), exactly as you already use it.
"""
import argparse
import logging
import threading
from pathlib import Path

import uvicorn

from .api import make_app
from .comfy import Comfy
from .store import Store
from .workflows import load_all
from .worker import Worker


def main():
    p = argparse.ArgumentParser(prog="comfy_serve")
    p.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--data", default="data", help="job database, uploads and outputs")
    p.add_argument("--workflows", default=str(Path(__file__).resolve().parent.parent / "workflows"))
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)          # one line per poll is noise, not information
    data = Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    store, comfy, workflows = Store(data / "jobs.sqlite"), Comfy(a.comfy), load_all(a.workflows)
    logging.getLogger("comfy_serve").info("workflows: %s", ", ".join(workflows) or "none")

    worker = Worker(store, comfy, workflows, data)
    threading.Thread(target=worker.run_forever, name="worker", daemon=True).start()
    uvicorn.run(make_app(store, comfy, workflows, data), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
