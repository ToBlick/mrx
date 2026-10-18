output "gcs_bucket" {
  description = "GCS bucket for the XLA cache and job outputs"
  value       = google_storage_bucket.mrx_bucket.name
}

output "resolved_tpu_zones" {
  description = "Zones that offer the TPU v5p and v6e machine types to this project"
  value = {
    v5p = local.v5p_zones
    v6e = local.v6e_zones
  }
}
