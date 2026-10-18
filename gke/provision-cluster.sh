#!/usr/bin/env bash
# Creates or updates the MRX GKE cluster(s) of gcp.env: checks the logins, enables the APIs, creates the Terraform
# state bucket, applies terraform/ and prepares each cluster for jobs. Run by make setup and make infra
set -euo pipefail

MRX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(bash "${MRX_ROOT}/gke/settings.sh" --export)"
: "${PROJECT_ID:?Copy gcp.env.example to gcp.env and set PROJECT_ID}"
PLAN="$(mktemp -t mrx_tfplan.XXXXXX)"
trap 'rm -f "${PLAN}"' EXIT

tf() { terraform -chdir="${MRX_ROOT}/terraform" "$@"; }

# The zones whose Cloud TPU API offers each TPU of a pool region, as a Terraform map keyed <family>@<region>: a zone
# can list the machine type and still refuse TPU nodes (us-east1-b with v6e). A region whose lookup fails gets no key,
# so Terraform keeps every zone that lists the machine type there
tpu_api_zones() {
    local pools=() entries=() pool family region accel zone types zones failed
    [[ "${ENABLE_TPU_V5P}" == "true" ]] && pools+=("v5p@${V5P_REGION}")
    [[ "${ENABLE_TPU_V6E}" == "true" ]] && pools+=("v6e@${V6E_REGION}")
    if [[ "${ENABLE_TPU_V5P}" == "true" && "${ENABLE_TPU_V6E}" == "true" ]]; then
        pools+=("v6e@${V5P_REGION}" "v5p@${V6E_REGION}")
    fi
    for pool in "${pools[@]}"; do
        family="${pool%@*}" region="${pool#*@}" accel="v6e-4" zones=() failed=0
        [[ "${family}" == "v5p" ]] && accel="v5p-8"
        for zone in $(gcloud compute zones list --project="${PROJECT_ID}" --filter="name~^${region}- AND status=UP" \
            --format="value(name)" 2>/dev/null); do
            if types="$(gcloud compute tpus accelerator-types list --zone="${zone}" --project="${PROJECT_ID}" \
                --format="value(type)" 2>/dev/null)"; then
                grep -qx "${accel}" <<< "${types}" && zones+=("\"${zone}\"")
            else
                failed=1
            fi
        done
        (( failed )) || entries+=("\"${pool}\" = [$(IFS=,; echo "${zones[*]:-}")]")
    done
    echo "{$(IFS=,; echo "${entries[*]:-}")}"
}

# Plans with the settings of gcp.env (the same -var list as TF_VARS in the Makefile) and applies the plan, unless it
# deletes or replaces shared resources
apply() {
    tf plan -input=false -out="${PLAN}" \
        -var="project_id=${PROJECT_ID}" \
        -var="region=${REGION}" \
        -var="enable_v5p=${ENABLE_TPU_V5P}" \
        -var="enable_v6e=${ENABLE_TPU_V6E}" \
        -var="v5p_region=${V5P_REGION}" \
        -var="v6e_region=${V6E_REGION}" \
        -var="network=${NETWORK}" \
        -var="subnetwork=${SUBNETWORK}" \
        -var="create_network=${CREATE_NETWORK}" \
        -var="private_nodes=${PRIVATE_NODES}" \
        -var="dedicated_service_accounts=${DEDICATED_SERVICE_ACCOUNTS}" \
        -var="tpu_api_zones=$(tpu_api_zones)" || return 1
    # Others share the clusters, the bucket and the images: a gcp.env that disables a TPU family or changes a
    # region must not delete them
    local show deletes
    show="$(tf show -no-color "${PLAN}")" || return 1
    deletes="$(grep -Eo '# google_(container_cluster|container_node_pool|storage_bucket|artifact_registry_repository)\.[^ ]+ (will be destroyed|must be replaced)' \
        <<< "${show}" || true)"
    if [[ -n "${deletes}" && "${ALLOW_CLUSTER_DELETE:-0}" != "1" ]]; then
        echo "[infra] ERROR: not applying, this plan deletes shared resources:" >&2
        sed 's/^# /[infra]   /' <<< "${deletes}" >&2
        echo "[infra] Check the regions in gcp.env. Remove clusters with make destroy, or re-run with" >&2
        echo "[infra] ALLOW_CLUSTER_DELETE=1 if that is intended" >&2
        exit 1
    fi
    tf apply -input=false "${PLAN}"
}

# Points kubectl at a cluster, and adds the jobs' Kubernetes service account and the ComputeClasses to it
prepare() {
    local cluster="$1" region="$2"
    gcloud container clusters get-credentials "${cluster}" --region="${region}" --project="${PROJECT_ID}"
    kubectl create serviceaccount "${KSA_NAME}" --namespace="${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    if [[ "${DEDICATED_SERVICE_ACCOUNTS}" == "true" ]]; then
        # The node pools that the ComputeClasses auto-create run as mrx-gke-nodes too
        awk -v sa="mrx-gke-nodes@${PROJECT_ID}.iam.gserviceaccount.com" \
            '{ print } $0 == "spec:" { print "  nodePoolConfig:"; print "    serviceAccount: " sa }' \
            "${MRX_ROOT}/gke/compute-classes.yaml" | kubectl apply -f -
    else
        kubectl apply -f "${MRX_ROOT}/gke/compute-classes.yaml"
    fi
}

# gcloud and kubectl run as the gcloud account, Terraform with Application Default Credentials (or an exported
# GOOGLE_OAUTH_ACCESS_TOKEN). Check that both reach the project before anything is created
gcloud projects describe "${PROJECT_ID}" --format="value(projectId)" >/dev/null \
    || { echo "[infra] ERROR: the gcloud account cannot access ${PROJECT_ID}. Run: gcloud auth login" >&2; exit 1; }
TF_TOKEN="${GOOGLE_OAUTH_ACCESS_TOKEN:-$(gcloud auth application-default print-access-token 2>/dev/null || true)}"
[[ -n "${TF_TOKEN}" ]] \
    || { echo "[infra] ERROR: Terraform needs Application Default Credentials. Run: gcloud auth application-default login" >&2; exit 1; }
curl -sf -o /dev/null -H "Authorization: Bearer ${TF_TOKEN}" \
    "https://cloudresourcemanager.googleapis.com/v1/projects/${PROJECT_ID}" \
    || { echo "[infra] ERROR: Application Default Credentials cannot access ${PROJECT_ID}. Run: gcloud auth application-default login, with an account that can" >&2; exit 1; }
unset TF_TOKEN

# What Terraform needs before it can plan: the APIs (its zone lookups call Compute Engine, which is off in a new
# project) and the bucket that holds its state
gcloud services enable compute.googleapis.com container.googleapis.com tpu.googleapis.com \
    artifactregistry.googleapis.com cloudbuild.googleapis.com storage.googleapis.com \
    cloudresourcemanager.googleapis.com iam.googleapis.com --project="${PROJECT_ID}"
if ! gcloud storage buckets describe "gs://${TF_STATE_BUCKET}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "[infra] Creating the shared Terraform state bucket gs://${TF_STATE_BUCKET}"
    gcloud storage buckets create "gs://${TF_STATE_BUCKET}" --project="${PROJECT_ID}" --location="${REGION}" \
        --uniform-bucket-level-access --public-access-prevention
    gcloud storage buckets update "gs://${TF_STATE_BUCKET}" --versioning
fi
tf init -input=false -reconfigure -backend-config="bucket=${TF_STATE_BUCKET}"

# Keep an existing v6e cluster where it is: it may be in V6E_BACKUP_REGION (see the retry below), whatever this
# checkout's gcp.env says
V6E_IN_STATE="$(tf state show -no-color 'google_container_cluster.mrx_v6e_cluster[0]' 2>/dev/null \
    | awk -F'"' '$1 ~ /^ *location *= *$/ {print $2; exit}' || true)"
if [[ "${ENABLE_TPU_V6E}" == "true" && -n "${V6E_IN_STATE}" && "${V6E_IN_STATE}" != "${V6E_REGION}" ]]; then
    echo "[infra] The v6e cluster is in ${V6E_IN_STATE}; keeping it there (V6E_REGION=${V6E_REGION} applies after make destroy)"
    V6E_REGION="${V6E_IN_STATE}"
fi

if ! apply; then
    # Move the v6e cluster only if Terraform did not create it, so an unrelated error never relocates a live cluster
    if [[ "${ENABLE_TPU_V6E}" == "true" && "${V6E_REGION}" != "${V6E_BACKUP_REGION}" ]] \
        && ! grep -q '^google_container_cluster.mrx_v6e_cluster' <<< "$(tf state list 2>/dev/null)"; then
        echo "[infra] Apply failed before the v6e cluster existed; retrying it in ${V6E_BACKUP_REGION}"
        V6E_REGION="${V6E_BACKUP_REGION}"
        apply
    else
        exit 1
    fi
fi

if [[ "${ENABLE_TPU_V5P}" == "true" ]]; then
    prepare "${V5P_CLUSTER_NAME}" "${V5P_REGION}"
fi
if [[ "${ENABLE_TPU_V6E}" == "true" ]]; then
    prepare "${V6E_CLUSTER_NAME}" "${V6E_REGION}"
fi
echo "[infra] Done (v5p=${ENABLE_TPU_V5P}@${V5P_REGION}, v6e=${ENABLE_TPU_V6E}@${V6E_REGION})"
