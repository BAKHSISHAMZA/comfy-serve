"""Workflow templates with named inputs.

A ComfyUI workflow in API format is a dict of nodes keyed by id ("6": {"class_type": "KSampler", "inputs": {...}}).
Clients should never have to know that the prompt lives in node "3", field "text": the next time the workflow is
edited in ComfyUI, the ids change and every client breaks. So each template comes with a small map next to it:

    workflows/product_scene.json          the graph, exported from ComfyUI with "Save (API Format)"
    workflows/product_scene.inputs.json   {"inputs": {"prompt": {"node": "3", "field": "text", "type": "text"}, ...},
                                           "outputs": ["8"]}

The client sends {"prompt": "...", "seed": 7} plus files; build() writes them into a copy of the graph.
Unknown names, missing required inputs and wrong types are rejected before anything reaches the GPU.
"""
import copy
import json
import random
from pathlib import Path

TYPES = {"text": str, "int": int, "float": (int, float), "image": str}


class WorkflowError(ValueError):
    """The request does not match the workflow's inputs. The API turns this into a 400."""


class Workflow:
    def __init__(self, name, graph, spec):
        self.name, self.graph, self.spec = name, graph, spec
        self.inputs = spec["inputs"]
        self.output_nodes = [str(n) for n in spec.get("outputs", [])]
        for key, inp in self.inputs.items():                      # catch a broken map at startup, not mid-job
            node = str(inp["node"])
            if node not in graph:
                raise WorkflowError(f"{name}: input '{key}' points to node {node}, which is not in the graph")
            if inp["type"] not in TYPES:
                raise WorkflowError(f"{name}: input '{key}' has unknown type {inp['type']}")
        for n in self.output_nodes:
            if n not in graph:
                raise WorkflowError(f"{name}: output node {n} is not in the graph")

    def describe(self):
        return {k: {"type": v["type"], "required": "default" not in v, "default": v.get("default")}
                for k, v in self.inputs.items()}

    def image_inputs(self):
        return [k for k, v in self.inputs.items() if v["type"] == "image"]

    def build(self, values, uploaded):
        """values: named non-file inputs from the client. uploaded: {image input name: name ComfyUI gave the file}."""
        unknown = set(values) - set(self.inputs)
        if unknown:
            raise WorkflowError(f"unknown inputs: {sorted(unknown)}; this workflow takes {sorted(self.inputs)}")
        graph = copy.deepcopy(self.graph)
        resolved = {}
        for key, inp in self.inputs.items():
            if inp["type"] == "image":
                if key not in uploaded:
                    raise WorkflowError(f"missing image input '{key}'")
                value = uploaded[key]
            elif key in values:
                value = values[key]
            elif "default" in inp:
                value = inp["default"]
            else:
                raise WorkflowError(f"missing input '{key}'")
            if inp["type"] == "int" and value == "random":
                value = random.randint(0, 2**31 - 1)               # recorded below, so any result can be re-rendered
            is_bool = isinstance(value, bool)                     # JSON true would otherwise pass as int 1
            if is_bool or not isinstance(value, TYPES[inp["type"]]):
                raise WorkflowError(f"input '{key}' must be {inp['type']}, got {type(value).__name__}")
            graph[str(inp["node"])]["inputs"][inp["field"]] = value
            resolved[key] = value
        return graph, resolved


def load_all(folder):
    """Every <name>.json that has a <name>.inputs.json next to it is a workflow."""
    found = {}
    for spec_path in sorted(Path(folder).glob("*.inputs.json")):
        name = spec_path.name[: -len(".inputs.json")]
        graph_path = spec_path.with_name(f"{name}.json")
        if not graph_path.exists():
            raise WorkflowError(f"{spec_path.name} has no {graph_path.name} next to it")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        if "nodes" in graph and "links" in graph:
            raise WorkflowError(f"{graph_path.name} is a UI workflow; export it with 'Save (API Format)' instead")
        found[name] = Workflow(name, graph, json.loads(spec_path.read_text(encoding="utf-8")))
    return found
