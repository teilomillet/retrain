"""TinkerBackend — wraps the Tinker remote GPU API.

Implements TrainingBackend by delegating to Tinker's ServiceClient
for sampling and training. All Tinker-specific protocol details
(ModelInput, SamplingParams, Datum, AdamParams) are encapsulated here.
"""

from python import Python, PythonObject

from src.backend import TrainingBackend, SampleSequence
from src.config import TrainConfig
from src.pybridge import (
    to_python_list,
    to_python_float_list,
    from_python_int_list,
    from_python_float_list,
    py_float,
    py_len,
)


# ---------------------------------------------------------------------------
# Tinker-specific helpers
# ---------------------------------------------------------------------------


fn _extract_sample_results(result: PythonObject) raises -> List[SampleSequence]:
    """Extract token IDs and logprobs from a Tinker sample result."""
    var sequences = result.sequences
    var n = py_len(sequences)
    var out = List[SampleSequence]()

    for i in range(n):
        var seq = sequences[i]
        var tokens = from_python_int_list(seq.tokens)
        var logprobs = from_python_float_list(seq.logprobs)
        out.append(SampleSequence(tokens^, logprobs^))

    return out^


fn _batch_build_datums(
    all_full_tokens: List[List[Int]],
    all_padded_logprobs: List[List[Float64]],
    all_padded_advantages: List[List[Float64]],
) raises -> PythonObject:
    """Build all Tinker Datum objects in one batch.

    Imports torch/tinker once, constructs all datums in a single
    Python context. Reduces per-datum overhead.
    """
    var torch = Python.import_module("torch")
    var types = Python.import_module("tinker.types")
    var TensorData = Python.import_module("tinker.types.tensor_data").TensorData
    var builtins = Python.import_module("builtins")

    var datums = builtins.list()
    var n = len(all_full_tokens)

    for i in range(n):
        var py_tokens = to_python_list(all_full_tokens[i])
        var py_logprobs = to_python_float_list(all_padded_logprobs[i])
        var py_advantages = to_python_float_list(all_padded_advantages[i])

        var model_input = types.ModelInput.from_ints(py_tokens)

        var loss_fn_inputs = Python.dict()
        loss_fn_inputs["target_tokens"] = TensorData.from_torch(
            torch.tensor(py_tokens, dtype=torch.long)
        )
        loss_fn_inputs["logprobs"] = TensorData.from_torch(
            torch.tensor(py_logprobs, dtype=torch.float32)
        )
        loss_fn_inputs["advantages"] = TensorData.from_torch(
            torch.tensor(py_advantages, dtype=torch.float32)
        )

        datums.append(types.Datum(
            model_input=model_input,
            loss_fn_inputs=loss_fn_inputs,
        ))

    return datums


# ---------------------------------------------------------------------------
# TinkerBackend
# ---------------------------------------------------------------------------


struct TinkerBackend(TrainingBackend):
    """Training backend using the Tinker remote GPU service."""

    var training_client: PythonObject
    var sampling_client: PythonObject

    fn __init__(out self, config: TrainConfig) raises:
        """Create Tinker service client and LoRA training client."""
        print("Connecting to Tinker...")
        var tinker = Python.import_module("tinker")
        var service_client: PythonObject
        if len(config.base_url) > 0:
            service_client = tinker.ServiceClient(base_url=config.base_url)
        else:
            service_client = tinker.ServiceClient()

        print(
            "Creating LoRA training client (model="
            + config.model + ", rank=" + String(config.lora_rank) + ")..."
        )
        self.training_client = service_client.create_lora_training_client(
            base_model=config.model,
            rank=config.lora_rank,
        )
        self.sampling_client = Python.none()
        print("Training client ready.")

    fn checkpoint_for_sampling(mut self, name: String) raises:
        """Save weights and get sampling client."""
        self.sampling_client = (
            self.training_client.save_weights_and_get_sampling_client(name=name)
        )

    fn sample_batch(
        mut self,
        prompts: List[List[Int]],
        num_samples: Int,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> List[List[SampleSequence]]:
        """Submit sampling requests and collect results."""
        var types = Python.import_module("tinker.types")

        # Fire all futures
        var futures = List[PythonObject]()
        for i in range(len(prompts)):
            var py_ids = to_python_list(prompts[i])
            var model_input = types.ModelInput.from_ints(py_ids)
            var sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            futures.append(self.sampling_client.sample(
                prompt=model_input,
                num_samples=num_samples,
                sampling_params=sampling_params,
            ))

        # Collect results
        var results = List[List[SampleSequence]]()
        for i in range(len(futures)):
            var sample_result = futures[i].result()
            results.append(_extract_sample_results(sample_result))
        return results^

    fn train_step(
        mut self,
        all_tokens: List[List[Int]],
        all_logprobs: List[List[Float64]],
        all_advantages: List[List[Float64]],
        lr: Float64,
        weight_decay: Float64,
    ) raises -> Float64:
        """Build datums, run forward-backward + optimizer step, return mean loss."""
        var datums = _batch_build_datums(all_tokens, all_logprobs, all_advantages)

        var fwd_bwd_future = self.training_client.forward_backward(
            datums, loss_fn="importance_sampling"
        )

        var types = Python.import_module("tinker.types")
        var adam_params = types.AdamParams(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
        )
        var optim_future = self.training_client.optim_step(adam_params)

        var fwd_bwd_result = fwd_bwd_future.result()
        _ = optim_future.result()

        # Extract mean loss
        var builtins = Python.import_module("builtins")
        var has_metrics = builtins.hasattr(fwd_bwd_result, "metrics")
        if Python.is_true(has_metrics) and Python.is_true(fwd_bwd_result.metrics):
            var n = py_len(datums)
            var loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
            return py_float(loss_sum) / Float64(max(n, 1))
        return 0.0

    fn save(mut self, name: String) raises:
        """Save training state checkpoint."""
        self.training_client.save_state(name=name)
