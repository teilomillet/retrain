# =============================================================================
# retrain Scaleway backend — Outputs
# =============================================================================

output "instance_ip" {
  description = "Public IP of the GPU instance (used for SSH)"
  value       = scaleway_instance_ip.gpu.address
}
