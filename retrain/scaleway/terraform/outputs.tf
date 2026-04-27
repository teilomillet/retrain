# =============================================================================
# retrain Scaleway backend — Outputs
# =============================================================================

output "inference_url" {
  description = "vLLM / SGLang inference endpoint"
  value       = "http://${scaleway_instance_ip.gpu.address}:8000"
}

output "training_url" {
  description = "retrain training server endpoint"
  value       = "http://${scaleway_instance_ip.gpu.address}:8001"
}

output "instance_ip" {
  description = "Public IP of the GPU instance"
  value       = scaleway_instance_ip.gpu.address
}

output "vpc_private_ip" {
  description = "VPC private IP of the GPU instance"
  value       = data.scaleway_ipam_ip.gpu.address
}
