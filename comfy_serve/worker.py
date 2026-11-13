"""The worker: takes queued jobs one at a time and runs them on ComfyUI.

One job at a time, on purpose. There is one GPU. ComfyUI would queue a second graph anyway, and keeping only one
job "running" means the timing and peak-VRAM numbers we record belong to that job alone.

Life of a job:
    claim (queued -> running) -> upload its images -> queue the graph -> poll /history -> download outputs -> done

Restart safety (recover), for every job left `running` by a crash:
    ComfyUI has it in /history     -> it finished while we were down: collect the outputs, no re-render
    ComfyUI still has it queued    -> keep waiting for it, no re-render
    ComfyUI never heard of it      -> ComfyUI restarted too, or we died before queueing: put it back in the queue
A job is retried at most MAX_ATTEMPTS times, so a graph that crashes ComfyUI cannot loop forever.
"""
import logging
import threading
import time
from pathlib import Path

from .comfy import ComfyDown, ComfyError, failed, output_files, run_seconds
from .workflows import WorkflowError

log = logging.getLogger("comfy_serve.worker")

POLL = 0.5           # seconds between /history checks
JOB_TIMEOUT = 900    # a render taking longer than this is treated as stuck
MAX_ATTEMPTS = 3


class Worker:
    def __init__(self, store, comfy, workflows, data_dir):
        self.store, self.comfy, self.workflows = store, comfy, workflows
        self.out_root = Path(data_dir) / "outputs"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        self.recover()
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._stop.wait(POLL)
                continue
            self.run_job(job)

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------ one job
    def run_job(self, job):
        jid, prompt_id = job["id"], None
        try:
            wf = self.workflows.get(job["workflow"])
            if wf is None:
                raise WorkflowError(f"unknown workflow {job['workflow']}")
            uploaded = {name: self.comfy.upload_image(Path(path).read_bytes(), f"{jid}_{name}{Path(path).suffix}")
                        for name, path in job["files"].items()}
            graph, _ = wf.build(job["inputs"], uploaded)
            prompt_id = self.comfy.queue(graph)
            self.store.set_prompt(jid, prompt_id)
            log.info("job %s queued in ComfyUI as %s", jid, prompt_id)
            self.wait_and_collect(jid, prompt_id, wf.output_nodes)
        except ComfyDown as e:
            if prompt_id is None:
                # ComfyUI was down before it ever saw this job. That is not the job's fault: put it back without
                # using up an attempt, and wait. Jobs simply queue up until ComfyUI is back.
                log.warning("job %s: %s; waiting for ComfyUI", jid, e)
                self.store.requeue(jid, e, refund_attempt=True)
                self._stop.wait(5)
            else:
                self._retry_or_fail(jid, job["attempts"], e)
        except (ComfyError, WorkflowError) as e:
            log.warning("job %s failed: %s", jid, e)
            self.store.fail(jid, e)
        except Exception as e:                                     # a bug of ours must not kill the worker
            log.exception("job %s crashed", jid)
            self.store.fail(jid, f"internal error: {e}")

    def wait_and_collect(self, jid, prompt_id, output_nodes):
        t0, peak, down_since = time.time(), 0.0, None
        while True:
            try:
                used, _ = self.comfy.vram()
                if used is not None:
                    peak = max(peak, used)
                entry = self.comfy.history(prompt_id)
                down_since = None
            except ComfyDown:
                # A blip (ComfyUI busy loading a model, a dropped connection) must not requeue a job that is still
                # rendering: that would render it twice. Only give up after a full minute without an answer.
                down_since = down_since or time.time()
                if time.time() - down_since > 60:
                    raise
                entry = None
            if entry is not None:
                break
            if time.time() - t0 > JOB_TIMEOUT:
                raise ComfyError(f"no result after {JOB_TIMEOUT} s")
            time.sleep(POLL)
        err = failed(entry)
        if err:
            raise ComfyError(err)
        files = output_files(entry, output_nodes)
        if not files:
            raise ComfyError("the workflow finished but saved no output")
        out_dir = self.out_root / jid
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, f in enumerate(files):
            dest = out_dir / f"{i:02d}_{f['filename']}"
            dest.write_bytes(self.comfy.view(f["filename"], f.get("subfolder", ""), f.get("type", "output")))
            paths.append(str(dest))
        seconds = run_seconds(entry)
        self.store.finish(jid, paths, seconds if seconds is not None else time.time() - t0, peak or None)
        log.info("job %s done: %d file(s), %.1f s", jid, len(paths), seconds or (time.time() - t0))

    def _retry_or_fail(self, jid, attempts, err):
        if attempts >= MAX_ATTEMPTS:
            self.store.fail(jid, f"gave up after {attempts} attempts: {err}")
        else:
            log.warning("job %s: %s, back in the queue", jid, err)
            self.store.requeue(jid, err)
            self._stop.wait(5)                                     # ComfyUI is down: do not spin

    # ------------------------------------------------------------------ crash recovery
    def recover(self):
        for job in self.store.by_status("running"):
            jid, pid = job["id"], job["prompt_id"]
            wf = self.workflows.get(job["workflow"])
            try:
                if pid and (self.comfy.history(pid) is not None or self.comfy.in_queue(pid)):
                    log.info("recover %s: ComfyUI still has %s, collecting it", jid, pid)
                    self.wait_and_collect(jid, pid, wf.output_nodes if wf else [])
                elif job["attempts"] >= MAX_ATTEMPTS:
                    self.store.fail(jid, f"gave up after {job['attempts']} attempts")
                else:
                    log.info("recover %s: ComfyUI never ran it, back in the queue", jid)
                    self.store.requeue(jid, "server restarted before ComfyUI ran it")
            except ComfyDown:
                self.store.requeue(jid, "ComfyUI down during recovery")
            except ComfyError as e:
                self.store.fail(jid, e)
