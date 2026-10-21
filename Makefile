# MRX on GKE Cloud TPUs: `make help` lists the targets; gcp.env holds the settings (gke/README.md)
SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help
# The targets chain (infra, build, run): never run them in parallel
.NOTPARALLEL:

export CLOUDSDK_CORE_DISABLE_PROMPTS := 1
# gcloud and kubectl use the gcloud login; only Terraform uses Application Default Credentials
export CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS := false

.PHONY: help regions setup infra build deploy deploy-sharded run sharded logs status pull stop destroy destroy-all

# LOAD_ENV: gcp.env (else the template), then the shared settings (gke/settings.sh). ENABLE_TPU_* and COMPUTE_CLASS
# are unset first, so that only gcp.env sets them and a commented-out line turns them off
LOAD_ENV := unset ENABLE_TPU_V5P ENABLE_TPU_V6E COMPUTE_CLASS; [[ -f gcp.env ]] && source gcp.env || source gcp.env.example; \
	eval "$$(bash gke/settings.sh --export)"
CHECK_PROJECT := if [[ -z "$${PROJECT_ID}" ]]; then echo "ERROR: Copy gcp.env.example to gcp.env and set PROJECT_ID" >&2; exit 1; fi
# ACTIVE_CLUSTERS: set CLUSTERS to the cluster:region pairs that exist (the v6e backup region included), with a
# kubeconfig entry for each
ACTIVE_CLUSTERS := CLUSTERS=(); \
	for pair in "$${V5P_CLUSTER_NAME}:$${V5P_REGION}" "$${V6E_CLUSTER_NAME}:$${V6E_REGION}" "$${V6E_CLUSTER_NAME}:$${V6E_BACKUP_REGION}"; do \
		IFS=: read -r c r <<< "$${pair}"; \
		[[ " $${CLUSTERS[*]:-} " == *" $${c}:$${r} "* ]] && continue; \
		if gcloud container clusters get-credentials "$${c}" --region="$${r}" --project="$${PROJECT_ID}" >/dev/null 2>&1; then \
			CLUSTERS+=("$${c}:$${r}"); \
		fi; \
	done; \
	if [[ -z "$${CLUSTERS[*]:-}" ]]; then \
		echo "No MRX cluster is running in $${PROJECT_ID} (an admin creates them with: make setup)" >&2; \
	fi

# The settings of gcp.env for Terraform (the same list as apply() in gke/provision-cluster.sh)
TF_VARS := -var="project_id=$${PROJECT_ID}" -var="region=$${REGION}" -var="enable_v5p=$${ENABLE_TPU_V5P}" \
	-var="enable_v6e=$${ENABLE_TPU_V6E}" -var="v5p_region=$${V5P_REGION}" -var="v6e_region=$${V6E_REGION}" \
	-var="network=$${NETWORK}" -var="subnetwork=$${SUBNETWORK}" -var="create_network=$${CREATE_NETWORK}" \
	-var="private_nodes=$${PRIVATE_NODES}" -var="dedicated_service_accounts=$${DEDICATED_SERVICE_ACCOUNTS}"
TF_INIT := terraform -chdir=terraform init -input=false -reconfigure -backend-config="bucket=$${TF_STATE_BUCKET}" >/dev/null

help:
	@$(LOAD_ENV); \
	echo "MRX on GKE Cloud TPUs (project=$${PROJECT_ID}, region=$${REGION}, class=$${COMPUTE_CLASS:-auto})"; \
	echo ""; \
	echo "  cp gcp.env.example gcp.env       Configure PROJECT_ID (first time)"; \
	echo "  make deploy                      Build your image from this checkout and run it"; \
	echo "  RUNS=\"<args>; <args>\" make deploy  Up to 4 relax.py runs on one node, one per chip"; \
	echo "  make deploy-sharded              Build your image and run the shard_map benchmark (4 equilibria, 4 chips)"; \
	echo "  make build                       Build and push your image (mrx-tpu:<user>) only"; \
	echo "  make run                         Run your last build (no rebuild)"; \
	echo "  make sharded                     Run the benchmark with your last build (SHARDED_ARGS)"; \
	echo "  make status                      Show everyone's jobs and pods, and the cluster nodes"; \
	echo "  make logs                        Show recent logs from your latest job (FOLLOW=1 to stream)"; \
	echo "  make pull                        Sync gs://<bucket>/outputs/ to ./outputs/"; \
	echo "  make stop                        Delete your jobs (idle TPU nodes scale to zero)"; \
	echo "  make regions                     Show the enabled TPU families, their regions and cluster names"; \
	echo "  make setup                       Admin: create or update the clusters (Terraform) and build your image"; \
	echo "  make infra                       Admin: make setup without the image build"; \
	echo "  make destroy                     Admin: delete the clusters (\$$0 compute), keep the bucket and outputs"; \
	echo "  make destroy-all                 Admin: delete everything (clusters, bucket, Artifact Registry)"

regions:
	@bash gke/settings.sh

infra:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && bash gke/provision-cluster.sh

setup: infra build

# A build config: `gcloud builds submit --tag` only builds a file named Dockerfile. With DEDICATED_SERVICE_ACCOUNTS
# the build runs as mrx-build and keeps its source and logs in gs://<project>-mrx-builds, from where gcloud streams them
build:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && \
	IMAGE="$${IMAGE_NAME}"; \
	BUILD_CONFIG="$$(mktemp -t mrx_cloudbuild.XXXXXX.yaml)" && \
	printf "steps:\n- name: 'gcr.io/cloud-builders/docker'\n  args: ['build', '-f', 'Dockerfile.tpu', '-t', '%s', '.']\nimages:\n- '%s'\n" "$${IMAGE}" "$${IMAGE}" > "$${BUILD_CONFIG}" && \
	BUILD_FLAGS=(); \
	if [[ "$${DEDICATED_SERVICE_ACCOUNTS}" == "true" ]]; then \
		BUILDS="gs://$${PROJECT_ID}-mrx-builds"; \
		printf "logsBucket: '%s/logs'\noptions:\n  logging: GCS_ONLY\n" "$${BUILDS}" >> "$${BUILD_CONFIG}"; \
		BUILD_FLAGS=(--service-account="projects/$${PROJECT_ID}/serviceAccounts/mrx-build@$${PROJECT_ID}.iam.gserviceaccount.com" \
			--gcs-source-staging-dir="$${BUILDS}/source"); \
	fi; \
	gcloud builds submit . --config="$${BUILD_CONFIG}" --project="$${PROJECT_ID}" "$${BUILD_FLAGS[@]}" && \
	rm -f "$${BUILD_CONFIG}"

run:
	@FOLLOW="$${FOLLOW:-1}" bash gke/run.sh

# bench_multichip.py runs its own four equilibria, one per chip; RUNS is for relax.py, so it is cleared here
sharded:
	@SCRIPT="scripts/bench/bench_multichip.py" RUNS="" \
	ARGS="$${SHARDED_ARGS:---modes shard_map --resolution 12 20 12}" \
	FOLLOW="$${FOLLOW:-1}" \
	bash gke/run.sh

deploy: build run

deploy-sharded: build sharded

status:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && $(ACTIVE_CLUSTERS); \
	for pair in "$${CLUSTERS[@]}"; do \
		c="$${pair%%:*}"; r="$${pair##*:}"; ctx="gke_$${PROJECT_ID}_$${r}_$${c}"; \
		echo "=== $${c} ($${r}) ==="; \
		kubectl --context="$${ctx}" get jobs,pods -n "$${NAMESPACE}" -o wide; \
		kubectl --context="$${ctx}" get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-accelerator; \
	done

logs:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && $(ACTIVE_CLUSTERS); \
	FOLLOW_FLAG=""; [[ "$${FOLLOW:-0}" == "1" ]] && FOLLOW_FLAG="-f"; \
	for pair in "$${CLUSTERS[@]}"; do \
		c="$${pair%%:*}"; r="$${pair##*:}"; ctx="gke_$${PROJECT_ID}_$${r}_$${c}"; \
		LATEST_JOB="$$(kubectl --context="$${ctx}" get jobs -l app=mrx,mrx-user=$${MRX_USER} -n "$${NAMESPACE}" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null || true)"; \
		if [[ -n "$${LATEST_JOB}" ]]; then \
			echo "=== $${c} ($${r}): job/$${LATEST_JOB} ==="; \
			kubectl --context="$${ctx}" logs -n "$${NAMESPACE}" $${FOLLOW_FLAG} --tail=100 "job/$${LATEST_JOB}"; \
		fi; \
	done

pull:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && \
	mkdir -p outputs && \
	gcloud storage rsync -r "gs://$${GCS_BUCKET}/outputs" ./outputs --project="$${PROJECT_ID}"

stop:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && $(ACTIVE_CLUSTERS); \
	for pair in "$${CLUSTERS[@]}"; do \
		c="$${pair%%:*}"; r="$${pair##*:}"; ctx="gke_$${PROJECT_ID}_$${r}_$${c}"; \
		echo "Stopping $${MRX_USER}'s jobs on $${c} ($${r})..."; \
		kubectl --context="$${ctx}" delete jobs -l app=mrx,mrx-user=$${MRX_USER} -n "$${NAMESPACE}" --ignore-not-found; \
	done

# pull first: a local copy of the team's outputs (destroy-all, which runs this target, then deletes the bucket)
destroy: pull
	@$(LOAD_ENV) && $(CHECK_PROJECT) && $(ACTIVE_CLUSTERS); \
	if [[ "$${FORCE:-0}" != "1" ]]; then \
		for pair in "$${CLUSTERS[@]}"; do \
			c="$${pair%%:*}"; r="$${pair##*:}"; ctx="gke_$${PROJECT_ID}_$${r}_$${c}"; \
			ACTIVE_JOBS="$$(kubectl --context="$${ctx}" get jobs -l app=mrx -n "$${NAMESPACE}" -o jsonpath='{.items[?(@.status.active>0)].metadata.name}' 2>/dev/null || true)"; \
			if [[ -n "$${ACTIVE_JOBS}" ]]; then \
				echo "WARNING: Active MRX job(s) running on $${c} ($${r}): $${ACTIVE_JOBS}"; \
				read -r -p "Destroy cluster(s) anyway? [y/N]: " ans; \
				[[ "$${ans}" == "y" || "$${ans}" == "Y" ]] || { echo "Aborted (FORCE=1 skips this check)"; exit 1; }; \
			fi; \
		done; \
	fi && \
	echo "Destroying GKE cluster(s) (GCS bucket & outputs are preserved)..." && \
	$(TF_INIT) && \
	if terraform -chdir=terraform state list 2>/dev/null | grep -q 'google_container_cluster'; then \
		terraform -chdir=terraform destroy -input=false -auto-approve \
			-target=google_container_cluster.mrx_cluster \
			-target=google_container_cluster.mrx_v6e_cluster \
			$(TF_VARS); \
	fi && \
	for pair in "$${CLUSTERS[@]}"; do \
		c="$${pair%%:*}"; r="$${pair##*:}"; \
		if gcloud container clusters describe "$${c}" --region="$${r}" --project="$${PROJECT_ID}" >/dev/null 2>&1; then \
			echo "WARNING: $${c} ($${r}) still exists outside Terraform state. Delete it with: gcloud container clusters delete $${c} --region=$${r} --project=$${PROJECT_ID}"; \
		fi; \
	done

# Plain `make`, not $(MAKE): make -n runs every line that contains $(MAKE), so `make -n destroy-all` would tear down.
# The targeted apply puts force_destroy in the state first: only then can destroy delete a bucket that holds objects
destroy-all:
	@$(LOAD_ENV) && $(CHECK_PROJECT) && \
	echo "WARNING: This will permanently delete GKE clusters, gs://$${GCS_BUCKET}, Artifact Registry ($${AR_REPO}) and everything else Terraform created!"; \
	read -r -p "Type '$${GCS_BUCKET}' to confirm permanent deletion: " confirm; \
	if [[ "$${confirm}" != "$${GCS_BUCKET}" ]]; then echo "Aborted"; exit 1; fi && \
	FORCE=1 make destroy && \
	echo "Destroying remaining storage, registry, service account & network resources..." && \
	$(TF_INIT) && \
	if terraform -chdir=terraform state list 2>/dev/null | grep -q 'google_storage_bucket.mrx_bucket'; then \
		terraform -chdir=terraform apply -input=false -auto-approve -target=google_storage_bucket.mrx_bucket \
			$(TF_VARS) -var="destroy_storage=true" && \
		terraform -chdir=terraform destroy -input=false -auto-approve $(TF_VARS) -var="destroy_storage=true"; \
	fi && \
	if gcloud storage buckets describe "gs://$${GCS_BUCKET}" --project="$${PROJECT_ID}" >/dev/null 2>&1; then \
		echo "WARNING: gs://$${GCS_BUCKET} still exists outside Terraform state. Delete it with: gcloud storage rm -r gs://$${GCS_BUCKET}"; \
	fi; \
	if gcloud artifacts repositories describe "$${AR_REPO}" --location="$${REGION}" --project="$${PROJECT_ID}" >/dev/null 2>&1; then \
		echo "WARNING: Artifact Registry $${AR_REPO} still exists outside Terraform state. Delete it with: gcloud artifacts repositories delete $${AR_REPO} --location=$${REGION} --project=$${PROJECT_ID}"; \
	fi; \
	echo "Terraform state stays in gs://$${TF_STATE_BUCKET} (delete with: gcloud storage rm -r gs://$${TF_STATE_BUCKET})"
