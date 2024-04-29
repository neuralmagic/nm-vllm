from typing import Optional

import pytest

from vllm import CompletionOutput, SamplingParams

MODEL = "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse"
MAX_TOKENS = 20


@pytest.fixture(scope="session")
def vllm_model(vllm_runner):
    return vllm_runner(MODEL)


@pytest.mark.skip_global_cleanup
def test_quant_lm_head_layer(vllm_model):
    llm_engine = vllm_model.model.llm_engine

    assert llm_engine.model_config.hf_config.quantization_config["lm_head"] == True

    llm_engine.add_request(
        "id", "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        ), None)

    expected_output = "VLLM is a very popular and successful program"

    output: Optional[CompletionOutput] = None
    output_text = ""
    while llm_engine.has_unfinished_requests():
        (request_output,) = llm_engine.step()
        (output,) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text

    assert output is not None
    assert output_text.startswith(expected_output)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
