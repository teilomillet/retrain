"""LocalBackend — local GPU training via PyTorch/PEFT + pluggable inference.

This backend uses PyTorch/PEFT for LoRA training locally and delegates
inference to a pluggable InferenceEngine (PyTorch fallback, vLLM, SGLang,
or any OpenAI-compatible server).

The heavy lifting is delegated to retrain.local_train_helper.LocalTrainHelper.
"""

from python import Python, PythonObject

from src.backend import TrainingBackend, SampleSequence
from src.config import TrainConfig
from src.pybridge import (
    to_python_list,
    from_python_int_list,
    from_python_float_list,
    py_float,
    py_len,
)


struct LocalBackend(TrainingBackend):
    """Training backend using local GPU via PyTorch/PEFT."""

    var helper: PythonObject
    var adapter_path: String

    fn __init__(out self, config: TrainConfig) raises:
        """Initialize PyTorch model with PEFT LoRA adapter."""
        print("Initializing local backend (PyTorch/PEFT)...")

        # Ensure retrain package is importable from project root
        var sys = Python.import_module("sys")
        sys.path.insert(0, ".")

        var helper_mod = Python.import_module("retrain.local_train_helper")
        self.helper = helper_mod.LocalTrainHelper(
            config.model,
            config.adapter_path,
            config.devices,
            config.lora_rank,
            config.inference_engine,
            config.inference_url,
        )
        self.adapter_path = config.adapter_path
        print("Local backend ready.")

    fn checkpoint_for_sampling(mut self, name: String) raises:
        """Prepare for sampling by syncing weights to inference engine."""
        self.helper.checkpoint(name)

    fn sample_batch(
        mut self,
        prompts: List[List[Int]],
        num_samples: Int,
        max_tokens: Int,
        temperature: Float64,
        top_p: Float64,
    ) raises -> List[List[SampleSequence]]:
        """Generate completions via the configured inference engine."""
        var builtins = Python.import_module("builtins")

        # Convert prompts to Python list of lists
        var py_prompts = builtins.list()
        for i in range(len(prompts)):
            py_prompts.append(to_python_list(prompts[i]))

        # Call Python helper
        var py_results = self.helper.sample(
            py_prompts, num_samples, max_tokens, temperature, top_p
        )

        # Convert back to Mojo
        var results = List[List[SampleSequence]]()
        var n_prompts = py_len(py_results)
        for i in range(n_prompts):
            var group = List[SampleSequence]()
            var py_group = py_results[i]
            var n_samples = py_len(py_group)
            for j in range(n_samples):
                var py_sample = py_group[j]
                var tokens = from_python_int_list(py_sample[0])
                var logprobs = from_python_float_list(py_sample[1])
                group.append(SampleSequence(tokens^, logprobs^))
            results.append(group^)
        return results^

    fn train_step(
        mut self,
        all_tokens: List[List[Int]],
        all_logprobs: List[List[Float64]],
        all_advantages: List[List[Float64]],
        lr: Float64,
        weight_decay: Float64,
    ) raises -> Float64:
        """Run forward-backward + optimizer step via PyTorch/PEFT."""
        var builtins = Python.import_module("builtins")

        # Convert Mojo lists to Python lists
        var py_tokens = builtins.list()
        var py_logprobs = builtins.list()
        var py_advantages = builtins.list()

        for i in range(len(all_tokens)):
            py_tokens.append(to_python_list(all_tokens[i]))
        for i in range(len(all_logprobs)):
            var py_lp = builtins.list()
            for j in range(len(all_logprobs[i])):
                py_lp.append(all_logprobs[i][j])
            py_logprobs.append(py_lp)
        for i in range(len(all_advantages)):
            var py_adv = builtins.list()
            for j in range(len(all_advantages[i])):
                py_adv.append(all_advantages[i][j])
            py_advantages.append(py_adv)

        var loss = self.helper.train_step(
            py_tokens, py_logprobs, py_advantages, lr, weight_decay
        )
        return py_float(loss)

    fn save(mut self, name: String) raises:
        """Save LoRA adapter checkpoint."""
        self.helper.save_adapter(self.adapter_path, name)
