import logging
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import lm_eval
import numpy
import pytest
import torch
import yaml

from tests.utils.server import ServerContext


class Metric(TypedDict):
    name: str
    value: float


class Task(TypedDict):
    name: str
    metrics: List[Metric]


# to support python3.8 typing prior to adding `Required`/`NotRequired`, this
# class stores the optional keys and the `EvalTaskDefinition` subclass inherits
# those alongside the required keys it defines.
class EvalTaskDefinitionOpts(TypedDict, total=False):
    enable_tensor_parallel: bool
    extra_args: Dict[str, Any]


class EvalTaskDefinition(EvalTaskDefinitionOpts):
    model_name: str
    tasks: List[Task]


TEST_DATA_FILE = Path(__file__).parent / "lm-eval-tasks.yaml"
TEST_DATA = yaml.safe_load(TEST_DATA_FILE.read_text(encoding="utf-8"))
TEST_DATA: List[EvalTaskDefinition] = [
    pytest.param(eval_def, id=eval_def["model_name"]) for eval_def in TEST_DATA
]


@pytest.mark.parametrize("eval_data", TEST_DATA)
def test_lm_eval_correctness(
    eval_data: EvalTaskDefinition,
    logger: logging.Logger,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    model_name = eval_data["model_name"]
    logger.info("building server startup args")
    vllm_args = {
        "--model": model_name,
        "--disable-log-requests": None,
        "--max-model-len": 2048,
    }

    if eval_data.get("enable_tensor_parallel") is True:
        tp = torch.cuda.device_count()
        logger.info("Enabling tensor parallelism with %d devices", tp)
        vllm_args["--tensor-parallel-size"] = tp

    if extra_args := eval_data.get("extra_args"):
        vllm_args.update(extra_args)

    openai_args = ",".join([
        f"model={model_name}",
        "tokenizer_backend=huggingface",
        "base_url=http://localhost:8000/v1",
    ])

    logger.info("launching server")
    with ServerContext(vllm_args, logger=logger) as _:
        task_names = [t["name"] for t in eval_data["tasks"]]
        logger.info("getting results for task_names=%s", task_names)
        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=openai_args,
            tasks=task_names,
            batch_size=64,
        )

    logger.info("clearing torch cache")
    lm_eval.models.utils.clear_torch_cache()

    for task in eval_data["tasks"]:
        logger.info("checking metrics for task=%s", task["name"])
        for metric in task["metrics"]:
            ground_truth = metric["value"]
            measured_value = results["results"][task["name"]][metric["name"]]
            logger.info(
                "%s %s:\nground_truth=%s measured_value=%s",
                task["name"],
                metric["name"],
                ground_truth,
                measured_value,
            )

            assert numpy.isclose(ground_truth, measured_value, rtol=0.05)
