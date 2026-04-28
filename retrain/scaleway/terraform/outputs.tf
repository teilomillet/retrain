# =============================================================================
# retrain Scaleway backend — Outputs
# =============================================================================

output "inference_url" {
  description = "vLLM inference endpoint (PRIME-RL patched)"
  value       = "http://${scaleway_instance_ip.gpu.address}:8000"
}

output "instance_ip" {
  description = "Public IP of the GPU instance (used as ZMQ host for PRIME-RL transport)"
  value       = scaleway_instance_ip.gpu.address
}

output "vpc_private_ip" {
  description = "VPC private IP of the GPU instance"
  value       = data.scaleway_ipam_ip.gpu.address
}
