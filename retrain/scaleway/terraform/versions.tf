terraform {
  required_version = ">= 1.6.0"

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.50"
    }
  }
}

provider "scaleway" {
  zone       = var.zone
  region     = substr(var.zone, 0, length(var.zone) - 2)
  project_id = var.project_id != "" ? var.project_id : null
}
