#!/usr/bin/env bash
# Resolves the settings that the Makefile and the other gke/ scripts share, from gcp.env
#   bash gke/settings.sh                     summary (make regions)
#   eval "$(bash gke/settings.sh --export)"  the settings as variables
set -euo pipefail

MRX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# A project, TPU family or region given on the command line (or exported) wins; otherwise gcp.env decides, a
# commented-out ENABLE_TPU_* line meaning off
_CLI_PROJECT_ID="${PROJECT_ID:-}"
_CLI_REGION="${REGION:-}"
_CLI_ENABLE_V5P="${ENABLE_TPU_V5P:-}"
_CLI_ENABLE_V6E="${ENABLE_TPU_V6E:-}"
_CLI_V5P_REGION="${V5P_REGION:-}"
_CLI_V6E_REGION="${V6E_REGION:-}"
_CLI_V6E_BACKUP="${V6E_BACKUP_REGION:-}"

if [[ -f "${MRX_ROOT}/gcp.env" ]]; then
    source "${MRX_ROOT}/gcp.env"
elif [[ -f "${MRX_ROOT}/gcp.env.example" ]]; then
    source "${MRX_ROOT}/gcp.env.example"
fi

[[ -n "${_CLI_PROJECT_ID}" ]] && PROJECT_ID="${_CLI_PROJECT_ID}"
[[ -n "${_CLI_REGION}" ]] && REGION="${_CLI_REGION}"
[[ -n "${_CLI_ENABLE_V5P}" ]] && ENABLE_TPU_V5P="${_CLI_ENABLE_V5P}"
[[ -n "${_CLI_ENABLE_V6E}" ]] && ENABLE_TPU_V6E="${_CLI_ENABLE_V6E}"
[[ -n "${_CLI_V5P_REGION}" ]] && V5P_REGION="${_CLI_V5P_REGION}"
[[ -n "${_CLI_V6E_REGION}" ]] && V6E_REGION="${_CLI_V6E_REGION}"
[[ -n "${_CLI_V6E_BACKUP}" ]] && V6E_BACKUP_REGION="${_CLI_V6E_BACKUP}"

# The placeholder of gcp.env.example means the project is not set
[[ "${PROJECT_ID:-}" == "your-gcp-project-id" ]] && PROJECT_ID=""
PROJECT_ID="${PROJECT_ID:-}"

# Neither family enabled: take the one COMPUTE_CLASS or TPU_GEN names, else v6e, the documented default
ENABLE_TPU_V5P="${ENABLE_TPU_V5P:-false}"
ENABLE_TPU_V6E="${ENABLE_TPU_V6E:-false}"
if [[ "${ENABLE_TPU_V5P}" != "true" && "${ENABLE_TPU_V6E}" != "true" ]]; then
    if [[ "${COMPUTE_CLASS:-}" == *"v5p"* || "${TPU_GEN:-}" == "v5p" ]]; then
        ENABLE_TPU_V5P="true"
    else
        ENABLE_TPU_V6E="true"
    fi
fi

V5P_REGION="${V5P_REGION:-us-east5}"
V6E_REGION="${V6E_REGION:-us-east1}"
V6E_BACKUP_REGION="${V6E_BACKUP_REGION:-us-central1}"
# Region of the bucket and the image registry, which both clusters share
REGION="${REGION:-${V5P_REGION}}"

# The names of the clusters, bucket, registry, and the jobs' Kubernetes service account and namespace (the locals in
# terraform/main.tf), and of the Terraform state bucket gke/provision-cluster.sh creates. They are fixed, so that a
# variable of the same name in your shell cannot redirect a job
V5P_CLUSTER_NAME="mrx-tpu-cluster"
V6E_CLUSTER_NAME="mrx-tpu-cluster-v6e"
GCS_BUCKET="${PROJECT_ID}-mrx-tpu"
AR_REPO="mrx-repo"
KSA_NAME="mrx-runner"
NAMESPACE="default"
TF_STATE_BUCKET="${PROJECT_ID}-tfstate"

# Org-policy settings (gke/README.md, "Network and org policies")
NETWORK="${NETWORK:-}"
SUBNETWORK="${SUBNETWORK:-}"
CREATE_NETWORK="${CREATE_NETWORK:-false}"
PRIVATE_NODES="${PRIVATE_NODES:-false}"
DEDICATED_SERVICE_ACCOUNTS="${DEDICATED_SERVICE_ACCOUNTS:-false}"

# Lowercase [a-z0-9-], at most 20 characters: valid in job names, labels and image tags
MRX_USER="$(printf '%s' "${MRX_USER:-${USER:-}}" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' \
    | cut -c1-20 | sed 's/^-*//; s/-*$//')"
MRX_USER="${MRX_USER:-user}"
# Your image, which make build pushes and make run runs
IMAGE_NAME="${IMAGE_NAME:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/mrx-tpu:${MRX_USER}}"

if [[ "${1:-}" == "--export" ]]; then
    for name in PROJECT_ID REGION ENABLE_TPU_V5P ENABLE_TPU_V6E V5P_REGION V5P_CLUSTER_NAME V6E_REGION \
        V6E_BACKUP_REGION V6E_CLUSTER_NAME GCS_BUCKET AR_REPO KSA_NAME NAMESPACE NETWORK SUBNETWORK CREATE_NETWORK \
        PRIVATE_NODES DEDICATED_SERVICE_ACCOUNTS TF_STATE_BUCKET MRX_USER IMAGE_NAME; do
        echo "export ${name}=\"${!name}\""
    done
else
    if [[ -z "${PROJECT_ID}" ]]; then
        echo "Project: not set (copy gcp.env.example to gcp.env and set PROJECT_ID; user: ${MRX_USER})"
    else
        echo "Project: ${PROJECT_ID} (Terraform state: gs://${TF_STATE_BUCKET}, user: ${MRX_USER})"
    fi
    echo "  TPU v5p: enabled=${ENABLE_TPU_V5P} | cluster=${V5P_CLUSTER_NAME} | region=${V5P_REGION}"
    echo "  TPU v6e: enabled=${ENABLE_TPU_V6E} | cluster=${V6E_CLUSTER_NAME} | region=${V6E_REGION} (backup=${V6E_BACKUP_REGION})"
    echo "  Network: NETWORK=${NETWORK:-(default)} | CREATE_NETWORK=${CREATE_NETWORK} | PRIVATE_NODES=${PRIVATE_NODES}"
    echo "  Service accounts: DEDICATED_SERVICE_ACCOUNTS=${DEDICATED_SERVICE_ACCOUNTS}"
    echo "  Zones (admins, after make setup): terraform -chdir=terraform output resolved_tpu_zones"
fi
