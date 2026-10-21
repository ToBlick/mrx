# MRX on GKE Cloud TPUs: bucket, registry, optional service accounts and VPC, one cluster per enabled TPU family (each
# with TPU pools of every enabled family its region offers), and the jobs' access to the bucket. gke/provision-cluster.sh
# (make setup) applies it with the settings of gcp.env.

locals {
  # Fixed names: gke/settings.sh and the Makefile use the same ones
  v5p_cluster_name = "mrx-tpu-cluster"
  v6e_cluster_name = "mrx-tpu-cluster-v6e"
  bucket_name      = "${var.project_id}-mrx-tpu"
  builds_bucket    = "${var.project_id}-mrx-builds"
  ar_repo          = "mrx-repo"
  ksa_name         = "mrx-runner"
  namespace        = "default"
  # Nodes per zone in each TPU pool below; the pools the ComputeClasses auto-create are not bounded by it
  max_tpu_nodes = 16
  # CPU nodes for the system Pods in each cluster
  system_pool_nodes = 1

  # The nodes run as mrx-gke-nodes with dedicated service accounts (section 3), else as the project's default one
  node_service_account = var.dedicated_service_accounts ? google_service_account.nodes[0].email : null

  # The TPU node pools, keyed <cluster>-<family>: each cluster gets one for every enabled TPU family that its region
  # offers, so a job can start in either region (gke/run.sh submits it to every pool and keeps the first to get a node).
  # The pool of a cluster's own family must have zones; the other family's pool is left out where it has none
  tpu_pools_wanted = {
    for key, pool in {
      "v5p-v5p" = { cluster = "v5p", family = "v5p", own = true, enabled = var.enable_v5p }
      "v5p-v6e" = { cluster = "v5p", family = "v6e", own = false, enabled = var.enable_v5p && var.enable_v6e }
      "v6e-v6e" = { cluster = "v6e", family = "v6e", own = true, enabled = var.enable_v6e }
      "v6e-v5p" = { cluster = "v6e", family = "v5p", own = false, enabled = var.enable_v6e && var.enable_v5p }
    } : key => merge(pool, { region = pool.cluster == "v5p" ? var.v5p_region : var.v6e_region }) if pool.enabled
  }
  tpu_machine_type = { v5p = "ct5p-hightpu-4t", v6e = "ct6e-standard-4t" }
  # Every zone of each pool's region, whose machine types are looked up below
  tpu_zone_checks = merge([
    for key, pool in local.tpu_pools_wanted : {
      for zone in(pool.cluster == "v5p" ? data.google_compute_zones.v5p.names : data.google_compute_zones.v6e.names) :
      "${key}/${zone}" => { pool = key, zone = zone, type = local.tpu_machine_type[pool.family] }
    }
  ]...)
  # The zones of each pool: they list its machine type and, when gke/provision-cluster.sh looked them up, the Cloud TPU
  # API offers its TPU there too (us-east1-b lists ct6e-standard-4t, yet GKE refuses v6e nodes there)
  tpu_pool_zones = {
    for key, pool in local.tpu_pools_wanted : key => sort([
      for check_key, check in local.tpu_zone_checks : check.zone
      if check.pool == key && length(data.google_compute_machine_types.tpu[check_key].machine_types) > 0 &&
      contains(lookup(var.tpu_api_zones, "${pool.family}@${pool.region}", [check.zone]), check.zone)
    ])
  }
  tpu_pools = { for key, pool in local.tpu_pools_wanted : key => pool if pool.own || length(local.tpu_pool_zones[key]) > 0 }

  # The clusters' VPC network: the project's "default" one, an existing one named in gcp.env, or one Terraform creates
  # (section 4). An existing network needs a subnet named local.subnetwork in each cluster region.
  network    = var.network != "" ? var.network : (var.create_network ? "mrx-net" : "default")
  subnetwork = var.subnetwork != "" ? var.subnetwork : local.network
  # A created network gets one subnet per enabled TPU family. All subnets of a network share one address space, so
  # their ranges must not overlap.
  subnets = var.create_network ? {
    for family, subnet in {
      v5p = { enabled = var.enable_v5p, region = var.v5p_region, nodes = "10.10.0.0/20", pods = "10.11.0.0/16", services = "10.12.0.0/20" }
      v6e = { enabled = var.enable_v6e, region = var.v6e_region, nodes = "10.20.0.0/20", pods = "10.21.0.0/16", services = "10.22.0.0/20" }
    } : family => subnet if subnet.enabled
  } : {}
  # Private nodes in a created network reach the internet through Cloud NAT, one gateway per region
  nat_regions = var.private_nodes ? toset([for subnet in values(local.subnets) : subnet.region]) : toset([])
}

data "google_project" "current" {
  project_id = var.project_id
}

# The project's VPC networks, to check that the one the clusters use exists
data "google_compute_networks" "all" {
  project = var.project_id
}

# The zones of the two clusters' regions
data "google_compute_zones" "v5p" {
  project = var.project_id
  region  = var.v5p_region
  status  = "UP"
}

data "google_compute_zones" "v6e" {
  project = var.project_id
  region  = var.v6e_region
  status  = "UP"
}

# TPU machine types are offered in only some zones of a region, depending on the project: those zones get TPU nodes
data "google_compute_machine_types" "tpu" {
  for_each = local.tpu_zone_checks
  project  = var.project_id
  zone     = each.value.zone
  filter   = "name = \"${each.value.type}\""
}

# 1. GCS bucket for persistent XLA cache (/gcs/xla_cache) and job outputs (/gcs/outputs)
resource "google_storage_bucket" "mrx_bucket" {
  name                        = local.bucket_name
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = var.destroy_storage
  labels = {
    app = "mrx"
  }

  # Deletes xla_cache/ objects 30 days after creation, even if jobs still read them: each cache entry is recompiled
  # at most once a month, which also drops entries of old JAX versions
  lifecycle_rule {
    condition {
      age            = 30
      matches_prefix = ["xla_cache/"]
    }
    action {
      type = "Delete"
    }
  }
}

# 2. Artifact Registry repository for MRX TPU container images
resource "google_artifact_registry_repository" "mrx_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = local.ar_repo
  description   = "MRX TPU container images"
  format        = "DOCKER"
  labels = {
    app = "mrx"
  }
}

# 3. Optional service accounts for the nodes and builds, with only the roles they need (var.dedicated_service_accounts)
resource "google_service_account" "nodes" {
  count        = var.dedicated_service_accounts ? 1 : 0
  project      = var.project_id
  account_id   = "mrx-gke-nodes"
  display_name = "MRX GKE nodes"
}

# Node logs and metrics
resource "google_project_iam_member" "nodes" {
  count   = var.dedicated_service_accounts ? 1 : 0
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
  member  = google_service_account.nodes[0].member
}

# Image pulls
resource "google_artifact_registry_repository_iam_member" "nodes" {
  count      = var.dedicated_service_accounts ? 1 : 0
  project    = var.project_id
  location   = google_artifact_registry_repository.mrx_repo.location
  repository = google_artifact_registry_repository.mrx_repo.name
  role       = "roles/artifactregistry.reader"
  member     = google_service_account.nodes[0].member
}

resource "google_service_account" "builds" {
  count        = var.dedicated_service_accounts ? 1 : 0
  project      = var.project_id
  account_id   = "mrx-build"
  display_name = "MRX image builds (make build)"
}

# Image pushes
resource "google_artifact_registry_repository_iam_member" "builds" {
  count      = var.dedicated_service_accounts ? 1 : 0
  project    = var.project_id
  location   = google_artifact_registry_repository.mrx_repo.location
  repository = google_artifact_registry_repository.mrx_repo.name
  role       = "roles/artifactregistry.writer"
  member     = google_service_account.builds[0].member
}

# Source uploads and logs of the builds run as mrx-build (`make build` streams the logs from here). It holds nothing
# else, so it can always be deleted.
resource "google_storage_bucket" "builds" {
  count                       = var.dedicated_service_accounts ? 1 : 0
  name                        = local.builds_bucket
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true
  labels = {
    app = "mrx"
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Storage Admin on this bucket only: what Cloud Build documents for builds that keep their logs in your own bucket
resource "google_storage_bucket_iam_member" "builds" {
  count  = var.dedicated_service_accounts ? 1 : 0
  bucket = google_storage_bucket.builds[0].name
  role   = "roles/storage.admin"
  member = google_service_account.builds[0].member
}

# 4. Optional VPC network (var.create_network): subnets with Private Google Access, and Cloud NAT for private nodes
resource "google_compute_network" "mrx" {
  count                   = var.create_network ? 1 : 0
  name                    = local.network
  project                 = var.project_id
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "mrx" {
  for_each                 = local.subnets
  name                     = "${local.network}-${each.key}"
  project                  = var.project_id
  region                   = each.value.region
  network                  = google_compute_network.mrx[0].id
  ip_cidr_range            = each.value.nodes
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = each.value.pods
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = each.value.services
  }
}

resource "google_compute_router" "mrx" {
  for_each = local.nat_regions
  name     = "${local.network}-router"
  project  = var.project_id
  region   = each.value
  network  = google_compute_network.mrx[0].id
}

resource "google_compute_router_nat" "mrx" {
  for_each                           = local.nat_regions
  name                               = "${local.network}-nat"
  project                            = var.project_id
  region                             = each.value
  router                             = google_compute_router.mrx[each.value].name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# 5. TPU v5p cluster in var.v5p_region, when var.enable_v5p
resource "google_container_cluster" "mrx_cluster" {
  count      = var.enable_v5p ? 1 : 0
  name       = local.v5p_cluster_name
  project    = var.project_id
  location   = var.v5p_region
  network    = var.create_network ? google_compute_network.mrx[0].id : "projects/${var.project_id}/global/networks/${local.network}"
  subnetwork = var.create_network ? google_compute_subnetwork.mrx["v5p"].id : "projects/${var.project_id}/regions/${var.v5p_region}/subnetworks/${local.subnetwork}"

  # Off so that make destroy can delete the cluster; gke/provision-cluster.sh refuses plans that delete it instead
  deletion_protection      = false
  remove_default_node_pool = true
  initial_node_count       = 1

  # The default node pool only lives until the cluster is created, but with dedicated service accounts it too must
  # run as mrx-gke-nodes
  dynamic "node_config" {
    for_each = var.dedicated_service_accounts ? [1] : []
    content {
      service_account = local.node_service_account
      oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    }
  }

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  dynamic "private_cluster_config" {
    for_each = var.private_nodes ? [1] : []
    content {
      enable_private_nodes = true
    }
  }

  # Node auto-provisioning stays off: the CPU pools it would add for pending system Pods never scale down. The
  # ComputeClasses still auto-create TPU pools (GKE 1.33.3+); the profile makes the autoscaler drop idle nodes sooner
  cluster_autoscaling {
    enabled             = false
    autoscaling_profile = "OPTIMIZE_UTILIZATION"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = var.create_network ? "pods" : null
    services_secondary_range_name = var.create_network ? "services" : null
  }

  # With dedicated service accounts, the nodes need their roles before they start
  depends_on = [google_project_iam_member.nodes, google_artifact_registry_repository_iam_member.nodes]

  lifecycle {
    # After creation the API reports the first remaining node pool's settings here, not the default pool's
    ignore_changes = [node_config]

    precondition {
      condition     = var.create_network || contains(data.google_compute_networks.all.networks, local.network)
      error_message = "Project ${var.project_id} has no VPC network named ${local.network}. Set CREATE_NETWORK=\"true\" in gcp.env to create one, or NETWORK to an existing network."
    }
  }
}

# CPU nodes for the system Pods, in one zone: a regional pool would run system_pool_nodes nodes in every zone (the TPU
# pools are in section 7)
resource "google_container_node_pool" "system_pool" {
  count              = var.enable_v5p ? 1 : 0
  name               = "system-pool"
  project            = var.project_id
  location           = var.v5p_region
  cluster            = google_container_cluster.mrx_cluster[0].name
  node_locations     = [sort(data.google_compute_zones.v5p.names)[0]]
  initial_node_count = local.system_pool_nodes

  autoscaling {
    min_node_count = 1
    max_node_count = local.system_pool_nodes
  }

  node_config {
    machine_type    = "e2-standard-4"
    service_account = local.node_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# 6. TPU v6e cluster in var.v6e_region, when var.enable_v6e: same as section 5
resource "google_container_cluster" "mrx_v6e_cluster" {
  count      = var.enable_v6e ? 1 : 0
  name       = local.v6e_cluster_name
  project    = var.project_id
  location   = var.v6e_region
  network    = var.create_network ? google_compute_network.mrx[0].id : "projects/${var.project_id}/global/networks/${local.network}"
  subnetwork = var.create_network ? google_compute_subnetwork.mrx["v6e"].id : "projects/${var.project_id}/regions/${var.v6e_region}/subnetworks/${local.subnetwork}"

  deletion_protection      = false
  remove_default_node_pool = true
  initial_node_count       = 1

  dynamic "node_config" {
    for_each = var.dedicated_service_accounts ? [1] : []
    content {
      service_account = local.node_service_account
      oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    }
  }

  release_channel {
    channel = "REGULAR"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  dynamic "private_cluster_config" {
    for_each = var.private_nodes ? [1] : []
    content {
      enable_private_nodes = true
    }
  }

  cluster_autoscaling {
    enabled             = false
    autoscaling_profile = "OPTIMIZE_UTILIZATION"
  }

  ip_allocation_policy {
    cluster_secondary_range_name  = var.create_network ? "pods" : null
    services_secondary_range_name = var.create_network ? "services" : null
  }

  depends_on = [google_project_iam_member.nodes, google_artifact_registry_repository_iam_member.nodes]

  lifecycle {
    ignore_changes = [node_config]

    precondition {
      condition     = var.create_network || contains(data.google_compute_networks.all.networks, local.network)
      error_message = "Project ${var.project_id} has no VPC network named ${local.network}. Set CREATE_NETWORK=\"true\" in gcp.env to create one, or NETWORK to an existing network."
    }
  }
}

# CPU nodes for the system Pods, in one zone, as in section 5 (the TPU pools are in section 7)
resource "google_container_node_pool" "system_pool_v6e" {
  count              = var.enable_v6e ? 1 : 0
  name               = "system-pool"
  project            = var.project_id
  location           = var.v6e_region
  cluster            = google_container_cluster.mrx_v6e_cluster[0].name
  node_locations     = [sort(data.google_compute_zones.v6e.names)[0]]
  initial_node_count = local.system_pool_nodes

  autoscaling {
    min_node_count = 1
    max_node_count = local.system_pool_nodes
  }

  node_config {
    machine_type    = "e2-standard-4"
    service_account = local.node_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}

# 7. Scale-to-zero Spot TPU pools, local.tpu_pools. Their cloud.google.com/compute-class label lets the ComputeClass of
#    their family use them (gke/compute-classes.yaml); a class auto-creates another pool only when its own cannot scale
resource "google_container_node_pool" "tpu" {
  for_each           = local.tpu_pools
  name               = "tpu-${each.value.family}-4t-spot-pool"
  project            = var.project_id
  location           = each.value.region
  cluster            = each.value.cluster == "v5p" ? google_container_cluster.mrx_cluster[0].name : google_container_cluster.mrx_v6e_cluster[0].name
  node_locations     = local.tpu_pool_zones[each.key]
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = local.max_tpu_nodes
  }

  node_config {
    machine_type    = local.tpu_machine_type[each.value.family]
    spot            = true
    service_account = local.node_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      "cloud.google.com/compute-class" = "mrx-tpu-${each.value.family}"
    }

    # Only Pods that tolerate this taint, the MRX jobs (gke/job-tpu.yaml), run on these nodes
    taint {
      key    = "cloud.google.com/gke-quick-remove"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    precondition {
      condition     = length(local.tpu_pool_zones[each.key]) > 0
      error_message = "No zone of ${each.value.region} offers ${local.tpu_machine_type[each.value.family]} to project ${var.project_id}. Set ${upper(each.value.cluster)}_REGION in gcp.env to a region that does."
    }
  }
}

# The pools of the clusters' own families had their own resources before
moved {
  from = google_container_node_pool.tpu_v5p_spot[0]
  to   = google_container_node_pool.tpu["v5p-v5p"]
}

moved {
  from = google_container_node_pool.tpu_v6e_spot[0]
  to   = google_container_node_pool.tpu["v6e-v6e"]
}

# 8. Bucket read/write access for the jobs' Kubernetes service account in every cluster (Workload Identity, GCS Fuse)
resource "google_storage_bucket_iam_member" "ksa_storage_access" {
  bucket = google_storage_bucket.mrx_bucket.name
  role   = "roles/storage.objectUser"
  member = "principal://iam.googleapis.com/projects/${data.google_project.current.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/subject/ns/${local.namespace}/sa/${local.ksa_name}"
}
