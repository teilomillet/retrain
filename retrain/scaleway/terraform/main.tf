# =============================================================================
# retrain Scaleway backend — Main Infrastructure
# =============================================================================
# Architecture: 1 GPU instance running retrain with the local backend.
# retrain (local machine) rsyncs the project and runs via SSH.
# Only SSH (port 22) needs to be reachable from the caller.
# =============================================================================

locals {
  # ssh_cidr defaults to caller_ip when not explicitly set
  ssh_cidr = var.ssh_cidr != "" ? var.ssh_cidr : var.caller_ip
}

# -----------------------------------------------------------------------------
# VPC Private Network
# -----------------------------------------------------------------------------
resource "scaleway_vpc_private_network" "retrain" {
  name = "${var.project_name}-vpc"
  tags = ["retrain", "rlvr"]
}

# -----------------------------------------------------------------------------
# Security Group: GPU instance
# Inference (8000) and training (8001) ports are restricted to caller_ip.
# -----------------------------------------------------------------------------
resource "scaleway_instance_security_group" "gpu" {
  name                    = "${var.project_name}-sg-gpu"
  description             = "GPU: SSH only — retrain runs on the instance via SSH"
  zone                    = var.zone
  inbound_default_policy  = "drop"
  outbound_default_policy = "accept"

  # SSH — key-based auth enforced via Scaleway IAM SSH keys
  inbound_rule {
    action   = "accept"
    protocol = "TCP"
    port     = 22
    ip_range = local.ssh_cidr
  }
}

# -----------------------------------------------------------------------------
# Flexible IP
# -----------------------------------------------------------------------------
resource "scaleway_instance_ip" "gpu" {
  zone = var.zone
  tags = ["retrain", "rlvr", "gpu"]
}

# -----------------------------------------------------------------------------
# GPU Instance
# -----------------------------------------------------------------------------
resource "scaleway_instance_server" "gpu" {
  name              = "${var.project_name}-gpu"
  type              = var.instance_type
  image             = var.gpu_image
  zone              = var.zone
  security_group_id = scaleway_instance_security_group.gpu.id
  ip_id             = scaleway_instance_ip.gpu.id

  root_volume {
    size_in_gb = var.root_volume_size
  }

  cloud_init = templatefile("${path.module}/cloud-init.yaml", {
    model = var.model
  })

  private_network {
    pn_id = scaleway_vpc_private_network.retrain.id
  }

  tags = ["retrain", "rlvr", "gpu"]

}


data "scaleway_ipam_ip" "gpu" {
  mac_address = scaleway_instance_server.gpu.private_network[0].mac_address
  type        = "ipv4"
}
