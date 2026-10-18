output "gcs_bucket" {
  description = "GCS bucket for the XLA cache and job outputs"
  value       = google_storage_bucket.mrx_bucket.name
}

output "resolved_tpu_zones" {
  description = "Zones of each TPU pool, keyed <cluster>-<family>"
  value       = { for key, pool in local.tpu_pools : key => local.tpu_pool_zones[key] }
}
