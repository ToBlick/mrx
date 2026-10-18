terraform {
  required_version = ">= 1.5.0"

  # Shared state in gs://<project_id>-tfstate (created by gke/provision-cluster.sh, passed with -backend-config)
  backend "gcs" {}

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.7"
    }
  }
}

provider "google" {
  project = var.project_id
}
