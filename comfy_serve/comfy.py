"""A small client for ComfyUI's own HTTP API.

ComfyUI already exposes everything a job runner needs:
    POST /upload/image        put an input image in ComfyUI's input folder
    POST /prompt              queue a graph, get back a prompt_id
    GET  /queue               what is running and waiting right now
    GET  /history/{id}        the result once the graph has run: status, timing messages and output files
    GET  /view?filename=...   download an output file
    GET  /system_stats        device name, total and free VRAM

We poll /history instead of listening on the websocket. Polling twice a second costs nothing next to a render
that takes seconds, and it survives restarts: a websocket only tells you about prompts you were connected for,
/history tells you about any prompt, including one queued before this process crashed.
"""
import uuid

import httpx


class ComfyError(RuntimeError):
    """ComfyUI rejected the graph or failed while running it. Not worth retrying with the same inputs."""


class ComfyDown(RuntimeError):
    """ComfyUI could not be reached. Worth retrying later."""


class Comfy:
    def __init__(self, url, timeout=30.0):
        self.url = url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.http = httpx.Client(base_url=self.url, timeout=timeout)

    def _call(self, method, path, **kw):
        try:
            r = self.http.request(method, path, **kw)
        except httpx.TransportError as e:
            raise ComfyDown(f"ComfyUI not reachable at {self.url}: {e}") from e
        return r

    def upload_image(self, data, filename):
        r = self._call("POST", "/upload/image", files={"image": (filename, data, "image/png")},
                       data={"overwrite": "true"})
        if r.status_code != 200:
            raise ComfyError(f"upload of {filename} failed: {r.status_code} {r.text[:300]}")
        j = r.json()
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    def queue(self, graph):
        r = self._call("POST", "/prompt", json={"prompt": graph, "client_id": self.client_id})
        if r.status_code != 200:
            # ComfyUI explains what is wrong per node (missing model file, bad value...): pass that on verbatim.
            raise ComfyError(f"ComfyUI rejected the workflow: {r.text[:1500]}")
        return r.json()["prompt_id"]

    def history(self, prompt_id):
        """The finished entry for this prompt, or None while it is still waiting or running."""
        r = self._call("GET", f"/history/{prompt_id}")
        if r.status_code != 200:
            raise ComfyDown(f"/history returned {r.status_code}")
        return r.json().get(prompt_id)

    def in_queue(self, prompt_id):
        r = self._call("GET", "/queue")
        q = r.json()
        return any(item[1] == prompt_id for item in q.get("queue_running", []) + q.get("queue_pending", []))

    def view(self, filename, subfolder="", folder_type="output"):
        r = self._call("GET", "/view", params={"filename": filename, "subfolder": subfolder, "type": folder_type})
        if r.status_code != 200:
            raise ComfyError(f"could not download {filename}: {r.status_code}")
        return r.content

    def vram(self):
        """(used MB, total MB) on the first GPU, or (None, None) if ComfyUI does not report it."""
        try:
            dev = self._call("GET", "/system_stats").json()["devices"][0]
            total, free = dev["vram_total"], dev["vram_free"]
            return (total - free) / 2**20, total / 2**20
        except (ComfyDown, KeyError, IndexError, ValueError):
            return None, None

    def device(self):
        try:
            return self._call("GET", "/system_stats").json()["devices"][0]
        except (ComfyDown, KeyError, IndexError, ValueError):
            return None


def run_seconds(entry):
    """Render time from ComfyUI's own timestamps (execution_start -> execution_success), in seconds."""
    stamps = {name: data.get("timestamp") for name, data in entry.get("status", {}).get("messages", [])}
    if stamps.get("execution_start") and stamps.get("execution_success"):
        return (stamps["execution_success"] - stamps["execution_start"]) / 1000.0
    return None


def failed(entry):
    """The error text if ComfyUI ran the graph and it failed, else None."""
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for name, data in status.get("messages", []):
            if name == "execution_error":
                return f"{data.get('node_type')} (node {data.get('node_id')}): {data.get('exception_message')}"
        return "ComfyUI reported an error"
    return None


def output_files(entry, nodes=None):
    """Every image/video/gif file the graph saved, in node order. nodes limits it to the workflow's output nodes."""
    files = []
    for node_id, out in sorted(entry.get("outputs", {}).items(), key=lambda kv: kv[0]):
        if nodes and node_id not in nodes:
            continue
        for kind in ("images", "gifs", "videos"):
            for f in out.get(kind, []):
                if f.get("type", "output") == "output":
                    files.append(f)
    return files
