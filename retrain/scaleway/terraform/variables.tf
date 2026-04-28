# =============================================================================
# retrain Scaleway backend — Variables
# =============================================================================

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
variable "project_name" {
  description = "Name used for resource naming"
  type        = string
  default     = "retrain"
}

variable "project_id" {
  description = "Scaleway project ID (falls back to SCW_DEFAULT_PROJECT_ID env var)"
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Zone & Instance
# -----------------------------------------------------------------------------
variable "zone" {
  description = "Scaleway availability zone"
  type        = string
  default     = "fr-par-2"

  validation {
    condition     = can(regex("^(fr-par|nl-ams|pl-waw)-[1-3]$", var.zone))
    error_message = "Zone must be a valid Scaleway zone (e.g. fr-par-2)"
  }
}

variable "instance_type" {
  description = "Scaleway GPU instance type (e.g. L40S-1-48G, H100-1-80G)"
  type        = string
  default     = "L40S-1-48G"
}

variable "gpu_image" {
  description = "Scaleway image for GPU instances (Ubuntu with NVIDIA drivers pre-installed)"
  type        = string
  default     = "ubuntu_noble_gpu_os_13_nvidia"
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 150

  validation {
    condition     = var.root_volume_size >= 50 && var.root_volume_size <= 10000
    error_message = "Root volume size must be between 50 and 10000 GB"
  }
}

# -----------------------------------------------------------------------------
# Security
# -----------------------------------------------------------------------------
variable "caller_ip" {
  description = "CIDR of the machine running retrain — restricts inference/training ports"
  type        = string
  default     = "0.0.0.0/0"
}

variable "ssh_cidr" {
  description = "CIDR allowed to reach SSH (port 22). Defaults to caller_ip."
  type        = string
  default     = ""
}

# -----------------------------------------------------------------------------
# Model & Training
# -----------------------------------------------------------------------------
variable "model" {
  description = "HuggingFace model ID to serve"
  type        = string
}

variable "lora_rank" {
  description = "LoRA rank for training"
  type        = number
  default     = 32
}

variable "max_model_len" {
  description = "Maximum sequence length for the inference engine (prompt + completion tokens)"
  type        = number
  default     = 32768

  validation {
    condition     = var.max_model_len > 0
    error_message = "max_model_len must be > 0"
  }
}

variable "inference_engine" {
  description = "Inference engine: vllm or sglang"
  type        = string
  default     = "vllm"

  validation {
    condition     = contains(["vllm", "sglang"], var.inference_engine)
    error_message = "inference_engine must be 'vllm' or 'sglang'."
  }
}

# -----------------------------------------------------------------------------
# PRIME-RL deployment
# -----------------------------------------------------------------------------
variable "num_train_gpus" {
  description = "Number of GPUs allocated to the PRIME-RL trainer process"
  type        = number
  default     = 1
}

variable "num_infer_gpus" {
  description = "Number of GPUs allocated to the PRIME-RL inference server"
  type        = number
  default     = 1
}

