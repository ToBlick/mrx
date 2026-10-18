# Set from gcp.env by gke/provision-cluster.sh and the Makefile (TF_VARS); make destroy-all also sets destroy_storage

variable "project_id" {
  description = "Google Cloud project ID, the only required input"
  type        = string
}

variable "region" {
  description = "Region of the bucket and the Artifact Registry, which both clusters share"
  type        = string
  default     = "us-east5"
}

variable "enable_v5p" {
  description = "Deploy a GKE cluster and scale-to-zero TPU v5p (ct5p-hightpu-4t) node pool in v5p_region"
  type        = bool
  default     = false
}

variable "v5p_region" {
  description = "Region of the TPU v5p cluster"
  type        = string
  default     = "us-east5"
}

variable "enable_v6e" {
  description = "Deploy a GKE cluster and scale-to-zero TPU v6e (ct6e-standard-4t) node pool in v6e_region"
  type        = bool
  default     = true
}

variable "v6e_region" {
  description = "Region of the TPU v6e cluster; gke/provision-cluster.sh may pass V6E_BACKUP_REGION instead"
  type        = string
  default     = "us-east1"
}

variable "network" {
  description = "VPC network of the clusters (defaults to \"default\", or \"mrx-net\" when create_network is true)"
  type        = string
  default     = ""
}

variable "subnetwork" {
  description = "Subnet of the network in each cluster region (defaults to the network's name, as in the auto-mode \"default\" network); ignored when create_network is true"
  type        = string
  default     = ""
}

variable "create_network" {
  description = "Create the VPC network, with one subnet per TPU family and Private Google Access (plus Cloud NAT when private_nodes is true)"
  type        = bool
  default     = false
}

variable "private_nodes" {
  description = "Give nodes internal IP addresses only, for org policies that forbid external IPs (constraints/compute.vmExternalIpAccess)"
  type        = bool
  default     = false
}

variable "dedicated_service_accounts" {
  description = "Run the nodes and image builds as their own service accounts (mrx-gke-nodes, mrx-build) with only the roles they need, for orgs whose default service accounts have no roles; granting the roles needs Owner"
  type        = bool
  default     = false
}

variable "destroy_storage" {
  description = "Delete the bucket even if it holds objects (force_destroy); only make destroy-all sets it"
  type        = bool
  default     = false
}
