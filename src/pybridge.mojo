"""Python interop wrappers for external services.

All Python calls (Tinker API, HuggingFace tokenizer, datasets, wandb)
are isolated in this module. The training loop calls these thin wrappers
so the rest of the codebase stays pure Mojo.
"""

from python import Python, PythonObject
from collections import Optional

from src.advantages import identify_planning_tokens_native


# ---------------------------------------------------------------------------
# Python -> Mojo conversion helpers
# ---------------------------------------------------------------------------


fn py_int(obj: PythonObject) raises -> Int:
    """Convert a Python integer to a Mojo Int."""
    return Int(String(obj))


fn py_float(obj: PythonObject) raises -> Float64:
    """Convert a Python float to a Mojo Float64."""
    return Float64(String(obj))


fn py_len(obj: PythonObject) raises -> Int:
    """Get the length of a Python object."""
    var builtins = Python.import_module("builtins")
    return py_int(builtins.len(obj))


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------


fn to_python_list(data: List[Int]) raises -> PythonObject:
    """Convert a Mojo List[Int] to a Python list."""
    var builtins = Python.import_module("builtins")
    var py_list = builtins.list()
    for i in range(len(data)):
        py_list.append(data[i])
    return py_list


fn to_python_float_list(data: List[Float64]) raises -> PythonObject:
    """Convert a Mojo List[Float64] to a Python list."""
    var builtins = Python.import_module("builtins")
    var py_list = builtins.list()
    for i in range(len(data)):
        py_list.append(data[i])
    return py_list


fn from_python_int_list(obj: PythonObject) raises -> List[Int]:
    """Convert a Python list/sequence to Mojo List[Int]."""
    var n = py_len(obj)
    var result = List[Int](capacity=n)
    for i in range(n):
        result.append(py_int(obj[i]))
    return result^


fn from_python_float_list(obj: PythonObject) raises -> List[Float64]:
    """Convert a Python list/sequence to Mojo List[Float64]."""
    var n = py_len(obj)
    var result = List[Float64](capacity=n)
    for i in range(n):
        result.append(py_float(obj[i]))
    return result^


# ---------------------------------------------------------------------------
# Sample result extraction
# ---------------------------------------------------------------------------


@fieldwise_init
struct SampleSequence(Copyable, Movable):
    """Extracted data from one sample sequence."""

    var tokens: List[Int]
    var logprobs: List[Float64]


fn extract_sample_results(result: PythonObject) raises -> List[SampleSequence]:
    """Extract token IDs and logprobs from a Tinker sample result.

    Args:
        result: The SampleResult object from Tinker's sampling client.

    Returns:
        List of SampleSequence structs with tokens and logprobs.
    """
    var sequences = result.sequences
    var n = py_len(sequences)
    var out = List[SampleSequence]()

    for i in range(n):
        var seq = sequences[i]
        var tokens = from_python_int_list(seq.tokens)
        var logprobs = from_python_float_list(seq.logprobs)
        out.append(SampleSequence(tokens^, logprobs^))

    return out^


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


fn load_tokenizer(model: String) raises -> PythonObject:
    """Load a HuggingFace tokenizer.

    Args:
        model: HuggingFace model ID.

    Returns:
        A transformers.AutoTokenizer instance.
    """
    var transformers = Python.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(model)


fn encode_prompt(
    tokenizer: PythonObject,
    text: String,
) raises -> List[Int]:
    """Encode a prompt using chat template if available, otherwise raw encode.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Prompt text.

    Returns:
        Token IDs as a Mojo list.
    """
    var builtins = Python.import_module("builtins")
    var has_chat = builtins.hasattr(tokenizer, "apply_chat_template")
    var prompt_ids: PythonObject

    if Python.is_true(has_chat):
        var messages = Python.evaluate("lambda t: [{'role': 'user', 'content': t}]")(text)
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    else:
        prompt_ids = tokenizer.encode(text)

    # Handle BatchEncoding dicts
    var has_input_ids = builtins.hasattr(prompt_ids, "input_ids")
    if Python.is_true(has_input_ids):
        prompt_ids = prompt_ids["input_ids"]

    # Ensure it's a plain list
    var is_list = builtins.isinstance(prompt_ids, builtins.list)
    if not Python.is_true(is_list):
        prompt_ids = builtins.list(prompt_ids)

    return from_python_int_list(prompt_ids)


fn decode(tokenizer: PythonObject, token_ids: List[Int]) raises -> String:
    """Decode token IDs to text.

    Args:
        tokenizer: HuggingFace tokenizer.
        token_ids: Token IDs to decode.

    Returns:
        Decoded text string.
    """
    var py_ids = to_python_list(token_ids)
    var result = tokenizer.decode(py_ids, skip_special_tokens=True)
    return String(result)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@fieldwise_init
struct MathExample(Copyable, Movable):
    """A single MATH dataset example."""

    var problem: String
    var answer: String
    var level: String
    var type_: String


fn load_math_dataset(
    max_examples: Int = 0,
    subjects: Optional[List[String]] = None,
) raises -> List[MathExample]:
    """Load hendrycks/MATH dataset via EleutherAI mirror.

    Args:
        max_examples: Max examples to load (0 = unlimited).
        subjects: List of MATH subject names to include.

    Returns:
        List of MathExample structs.
    """
    var datasets = Python.import_module("datasets")

    # Default subjects
    var subject_list: List[String]
    if subjects is None:
        subject_list = List[String]()
        subject_list.append("intermediate_algebra")
        subject_list.append("precalculus")
        subject_list.append("number_theory")
        subject_list.append("counting_and_probability")
        subject_list.append("geometry")
    else:
        subject_list = subjects.value().copy()

    var examples = List[MathExample]()

    for s in range(len(subject_list)):
        var subject = subject_list[s]
        var ds = datasets.load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        var ds_len = py_len(ds)

        for i in range(ds_len):
            var item = ds[i]
            var solution = String(item["solution"])
            var answer = _extract_boxed_py(solution)
            var problem = String(item["problem"])

            var level_obj = item.get("level", "")
            var type_obj = item.get("type", "")
            var level = String(level_obj)
            var type_val = String(type_obj)

            examples.append(MathExample(problem, answer, level, type_val))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        if max_examples > 0 and len(examples) >= max_examples:
            break

    return examples^


fn _extract_boxed_py(text: String) -> String:
    """Extract \\boxed{...} answer from MATH solution text (native Mojo)."""
    var marker = "\\boxed{"
    var marker_len = len(marker)

    var idx = text.rfind(marker)
    if idx == -1:
        return String("")

    var start = idx + marker_len
    var depth = 1
    var pos = start
    var text_len = len(text)
    var bytes = text.as_bytes()
    while pos < text_len and depth > 0:
        if bytes[pos] == UInt8(ord("{")):
            depth += 1
        elif bytes[pos] == UInt8(ord("}")):
            depth -= 1
        pos += 1

    var extracted = String(text[start : pos - 1])
    return String(extracted.strip())


# ---------------------------------------------------------------------------
# Tinker API
# ---------------------------------------------------------------------------


fn create_tinker_client(base_url: String) raises -> PythonObject:
    """Create a Tinker ServiceClient."""
    var tinker = Python.import_module("tinker")
    if len(base_url) > 0:
        return tinker.ServiceClient(base_url=base_url)
    return tinker.ServiceClient()


fn create_lora_training_client(
    client: PythonObject,
    model: String,
    rank: Int,
) raises -> PythonObject:
    """Create a LoRA training client on Tinker's GPUs."""
    return client.create_lora_training_client(
        base_model=model,
        rank=rank,
    )


fn save_weights_and_get_sampling_client(
    training_client: PythonObject,
    name: String,
) raises -> PythonObject:
    """Save current weights and get a sampling client."""
    return training_client.save_weights_and_get_sampling_client(name=name)


fn sample(
    sampling_client: PythonObject,
    prompt_ids: List[Int],
    num_samples: Int,
    max_tokens: Int,
    temperature: Float64,
    top_p: Float64 = 0.95,
) raises -> PythonObject:
    """Submit a sampling request to Tinker."""
    var types = Python.import_module("tinker.types")
    var py_ids = to_python_list(prompt_ids)
    var model_input = types.ModelInput.from_ints(py_ids)
    var sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return sampling_client.sample(
        prompt=model_input,
        num_samples=num_samples,
        sampling_params=sampling_params,
    )


fn forward_backward(
    training_client: PythonObject,
    datums: PythonObject,
) raises -> PythonObject:
    """Submit forward-backward pass to Tinker."""
    return training_client.forward_backward(datums, loss_fn="importance_sampling")


fn optim_step(
    training_client: PythonObject,
    lr: Float64,
    weight_decay: Float64 = 0.0,
) raises -> PythonObject:
    """Submit an optimizer step to Tinker."""
    var types = Python.import_module("tinker.types")
    var adam_params = types.AdamParams(
        learning_rate=lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=weight_decay,
    )
    return training_client.optim_step(adam_params)


fn save_state(training_client: PythonObject, name: String) raises:
    """Save training state checkpoint."""
    training_client.save_state(name=name)


# ---------------------------------------------------------------------------
# Datum packing
# ---------------------------------------------------------------------------


fn build_datum(
    full_tokens: List[Int],
    padded_logprobs: List[Float64],
    padded_advantages: List[Float64],
) raises -> PythonObject:
    """Build a Tinker Datum object for training."""
    var torch = Python.import_module("torch")
    var types = Python.import_module("tinker.types")
    var TensorData = Python.import_module("tinker.types.tensor_data").TensorData

    var py_tokens = to_python_list(full_tokens)
    var py_logprobs = to_python_float_list(padded_logprobs)
    var py_advantages = to_python_float_list(padded_advantages)

    var model_input = types.ModelInput.from_ints(py_tokens)

    var target_tensor = TensorData.from_torch(
        torch.tensor(py_tokens, dtype=torch.long)
    )
    var logprob_tensor = TensorData.from_torch(
        torch.tensor(py_logprobs, dtype=torch.float32)
    )
    var advantage_tensor = TensorData.from_torch(
        torch.tensor(py_advantages, dtype=torch.float32)
    )

    var loss_fn_inputs = Python.dict()
    loss_fn_inputs["target_tokens"] = target_tensor
    loss_fn_inputs["logprobs"] = logprob_tensor
    loss_fn_inputs["advantages"] = advantage_tensor

    return types.Datum(
        model_input=model_input,
        loss_fn_inputs=loss_fn_inputs,
    )


# ---------------------------------------------------------------------------
# Vocabulary table (one-time load at startup)
# ---------------------------------------------------------------------------


fn load_vocab_table(tokenizer: PythonObject) raises -> List[String]:
    """Load the full tokenizer vocabulary into a Mojo List[String].

    Indexed by token ID. One Python call at startup; after this,
    token ID -> string fragment lookups are pure Mojo array access.
    """
    var builtins = Python.import_module("builtins")
    var vocab_size = py_int(tokenizer.vocab_size)

    # Some tokenizers have added tokens beyond vocab_size
    var has_added = builtins.hasattr(tokenizer, "added_tokens_encoder")
    var total_size = vocab_size
    if Python.is_true(has_added):
        var added_len = py_len(tokenizer.added_tokens_encoder)
        if vocab_size + added_len > total_size:
            total_size = vocab_size + added_len

    # Build a list of all IDs [0, total_size)
    var py_ids = builtins.list(builtins.range(total_size))
    var py_tokens = tokenizer.convert_ids_to_tokens(py_ids)
    var none_type = builtins.type(Python.none())

    var table = List[String](capacity=total_size)
    for i in range(total_size):
        var tok = py_tokens[i]
        if Python.is_true(builtins.isinstance(tok, none_type)):
            table.append(String(""))
        else:
            table.append(String(tok))
    return table^


fn vocab_lookup(
    vocab_table: List[String],
    token_ids: List[Int],
) -> List[String]:
    """Look up token strings from a pre-loaded vocabulary table (pure Mojo).

    Falls back to empty string for out-of-range IDs.
    """
    var n = len(token_ids)
    var table_size = len(vocab_table)
    var result = List[String](capacity=n)
    for i in range(n):
        var tid = token_ids[i]
        if tid >= 0 and tid < table_size:
            result.append(vocab_table[tid])
        else:
            result.append(String(""))
    return result^


# ---------------------------------------------------------------------------
# Batch decode
# ---------------------------------------------------------------------------


fn batch_decode(
    tokenizer: PythonObject,
    sequences: List[List[Int]],
) raises -> List[String]:
    """Decode multiple token sequences in one Python call.

    Uses tokenizer.batch_decode() — one call instead of N.
    """
    var builtins = Python.import_module("builtins")
    var py_batch = builtins.list()
    for i in range(len(sequences)):
        py_batch.append(to_python_list(sequences[i]))

    var py_results = tokenizer.batch_decode(py_batch, skip_special_tokens=True)
    var n = py_len(py_results)
    var result = List[String](capacity=n)
    for i in range(n):
        result.append(String(py_results[i]))
    return result^


# ---------------------------------------------------------------------------
# Token string decoding + planning tokens
# ---------------------------------------------------------------------------


fn convert_ids_to_token_strs(
    tokenizer: PythonObject,
    token_ids: List[Int],
) raises -> List[String]:
    """Decode token IDs to string fragments via tokenizer (one Python call).

    Returns Mojo strings ready for identify_planning_tokens_native.
    Prefer vocab_lookup() with a pre-loaded table for hot paths.
    """
    var py_ids = to_python_list(token_ids)
    var py_tokens = tokenizer.convert_ids_to_tokens(py_ids)
    var n = py_len(py_tokens)
    var builtins = Python.import_module("builtins")
    var none_type = builtins.type(Python.none())

    var result = List[String](capacity=n)
    for i in range(n):
        var tok = py_tokens[i]
        if Python.is_true(builtins.isinstance(tok, none_type)):
            result.append(String(""))
        else:
            result.append(String(tok))
    return result^


fn identify_planning_tokens(
    token_ids: List[Int],
    tokenizer: PythonObject,
    strategic_grams: List[String],
    max_window: Int = 5,
) raises -> List[Int]:
    """Identify planning tokens via strategic gram matching.

    Convenience wrapper: decodes token IDs then runs native matching.
    For hot paths, prefer vocab_lookup + identify_planning_tokens_native.
    """
    var token_strs = convert_ids_to_token_strs(tokenizer, token_ids)
    return identify_planning_tokens_native(token_strs, strategic_grams, max_window)


# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------


fn init_wandb(
    project: String,
    name: String,
    config_dict: PythonObject,
) raises -> PythonObject:
    """Initialize a wandb run."""
    var wandb = Python.import_module("wandb")
    return wandb.init(
        project=project,
        name=name,
        config=config_dict,
    )


fn log_wandb(
    run: PythonObject,
    metrics: PythonObject,
    step: Int,
) raises:
    """Log metrics to wandb."""
    run.log(metrics, step=step)


fn finish_wandb(run: PythonObject) raises:
    """Finish a wandb run."""
    run.finish()
