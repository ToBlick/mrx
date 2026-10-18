# MRX on GKE Cloud TPUs

Infrastructure and job manifests for running MRX on Google Cloud TPU v6e and v5p nodes (`MRX_DTYPE=float32` with double-single refinement). The measurements are in [`../docs/gke_tpu_architecture_and_benchmarks.md`](../docs/gke_tpu_architecture_and_benchmarks.md).

## Files

* [`../gcp.env.example`](../gcp.env.example): the configuration template (`PROJECT_ID`, the TPU families, regions, network settings, `MRX_USER`, `SCRIPT`, `ARGS`). Copy it to `../gcp.env` (gitignored), the only configuration file.
* [`../terraform/`](../terraform): the GKE cluster(s), the scale-to-zero TPU node pools (`ct5p-hightpu-4t`, `ct6e-standard-4t`), the GCS bucket, Artifact Registry, Workload Identity, and optionally a VPC and the service accounts the nodes and image builds run as. Each cluster gets a TPU pool for every enabled family its region offers, in the zones where the Cloud TPU API offers it (`make setup` looks them up). The state is shared in `gs://<PROJECT_ID>-tfstate`.
* [`../Dockerfile.tpu`](../Dockerfile.tpu): the job image (Cloud TPU JAX and your checkout of `mrx`), built by `make build`.
* [`provision-cluster.sh`](provision-cluster.sh): run by `make setup`. It checks the logins, enables the APIs, creates the state bucket, applies the Terraform module (never deleting shared resources, see "Team use"; v6e falls back to `V6E_BACKUP_REGION`) and prepares each cluster (kubeconfig, Kubernetes service account, ComputeClasses).
* [`settings.sh`](settings.sh): resolves the enabled TPU families, their regions and cluster names, the network settings, the state bucket and `MRX_USER` from `gcp.env` (`make regions`).
* [`compute-classes.yaml`](compute-classes.yaml): the GKE `ComputeClass`es `mrx-tpu-v5p` and `mrx-tpu-v6e`, both Spot only.
* [`run.sh`](run.sh) and [`job-tpu.yaml`](job-tpu.yaml): submit a job running your image (`mrx-tpu:<MRX_USER>`) to every TPU pool that can run it, and keep the first to get a node: one `relax.py` run, up to four (`RUNS`, one per chip), or the benchmark, with `/gcs/xla_cache` and `/gcs/outputs` mounted from the bucket.
* [`../scripts/bench/bench_multichip.py`](../scripts/bench/bench_multichip.py): the `jax.shard_map` benchmark behind the node numbers in the docs (`make sharded`, `make deploy-sharded`).

## Team use

Everyone works in one shared project and one set of clusters.

* **Everyone** needs `gcloud`, `kubectl` with `gke-gcloud-auth-plugin`, `envsubst` (GNU gettext), `make` and `bash` (on Windows, use WSL), runs `gcloud auth login`, and copies `gcp.env.example` to `gcp.env` with the team's `PROJECT_ID`. Editor on the project is enough to run jobs.
* **Images and jobs are per user** (`MRX_USER`, default: your login name): `make deploy` builds and runs `mrx-tpu:<user>`, jobs are named `mrx-<user>-<timestamp>`, and `make stop` and `make logs` act only on your jobs. `make status` shows everyone's.
* **Admins** also run `make setup`, `make destroy` and `make destroy-all`. These need `terraform` >= 1.5 and `gcloud auth application-default login`, because Terraform uses Application Default Credentials. The first `make setup` in a project grants the jobs access to the bucket, which takes Owner (as does `DEDICATED_SERVICE_ACCOUNTS`); otherwise Editor is enough. The Terraform state is shared and locked during runs, so any admin can run them from any checkout. `make setup` never deletes or replaces a cluster, a node pool, the bucket or the registry: use `make destroy`, or `ALLOW_CLUSTER_DELETE=1 make setup` to drop a TPU family on purpose.

## Network and org policies

The defaults (the project's `default` VPC, nodes with external IP addresses, the project's default service accounts) work in most projects. If your org's policies rule them out, set in `gcp.env`:

| Org constraint | Setting in `gcp.env` |
| :--- | :--- |
| No `default` VPC (`compute.skipDefaultNetworkCreation`), or the clusters should have their own | `CREATE_NETWORK="true"`: Terraform creates `mrx-net`, with one subnet per TPU family and Private Google Access |
| A VPC from your network team, in the same project | `NETWORK`, plus `SUBNETWORK` unless its subnets carry the network's name (one subnet of that name in each cluster region) |
| No external IPs on VMs (`compute.vmExternalIpAccess`) | `PRIVATE_NODES="true"`: nodes reach Google APIs (images, GCS) through Private Google Access. With `CREATE_NETWORK`, Cloud NAT also gives them internet access; on another network, ask its owners for NAT if jobs need the internet |
| Allowed regions (`gcp.resourceLocations`) | `V5P_REGION`, `V6E_REGION`, `V6E_BACKUP_REGION` and `REGION` (v6e falls back to `V6E_BACKUP_REGION` by itself) |
| Default service accounts without roles (orgs created since May 2024), or banned for GKE | `DEDICATED_SERVICE_ACCOUNTS="true"`: nodes and builds run as `mrx-gke-nodes` and `mrx-build`, with only the roles they need (`make setup` then needs Owner, to grant them) |

Changing the network settings later replaces the clusters, which `make setup` refuses: run `make destroy` first. Not covered yet: Shared VPC, customer-managed encryption keys (CMEK), Binary Authorization, private Cloud Build pools and DNS-only control-plane access.

## Where to start

* **TPU v6e** (`ENABLE_TPU_V6E`, the default in `gcp.env.example`): the cheaper TPU on every mesh measured, 2.4–3.2× cheaper per equilibrium than v5p up to 12×20×12 and 1.6–1.8× at 32×44×20.
* **Both families when capacity is short.** With `ENABLE_TPU_V5P` and `ENABLE_TPU_V6E`, each cluster also gets the other family's pool where its region offers it: v6e in `us-east1-d` and `us-east5` a, b and c, v5p in `us-east5` a, b and c and `us-east1-d`. A job waits in all of these pools at once and runs in the first to get a node, v6e first when two get one together; `TPU_GEN=v6e` or `TPU_GEN=v5p` keeps it to one family.
* **Use all four chips of a node.** A node has four chips and is billed whole, and one `relax.py` run uses one chip. Give `RUNS` up to four runs (below) and they run side by side, one per chip, with the same results and speed as alone: on v6e, four runs cost 2.6× less per equilibrium than one.
* **Small meshes.** At 8×12×12 a v6e node was 5.4× faster than one NVIDIA H200 on Newton and about 10× cheaper; at 48×64×24 a v5p node was 5–6× slower and 6–8× more expensive. The crossover in between is not measured, so keep 48×64×24 and larger on the GPU for now.

## Quick reference

```bash
# 1. Copy the template, then set PROJECT_ID in gcp.env (the only required setting; TPU v6e is enabled by default):
cp gcp.env.example gcp.env

# 2. Log in (the second login, for Terraform, is needed only by admins who run make setup or destroy):
gcloud auth login
gcloud auth application-default login

# 3. One-time setup by an admin (from the repository root):
make setup

# 4. Build your checkout and run it on a TPU node:
make deploy
```

| Command | Description |
| :--- | :--- |
| `make setup` | Admin: provision the clusters and scale-to-zero TPU pools with Terraform, and build your image |
| `make deploy` | Rebuild your image with your local edits and run it on a TPU node |
| `RUNS="..." make deploy` | The same, with up to 4 `relax.py` runs on one node, one per chip |
| `make deploy-sharded` | Rebuild your image and run the `shard_map` benchmark (4 copies of one setup, one per chip) |
| `make build` | Build and push your image only |
| `ARGS="..." make run` | Run with your last build, without rebuilding |
| `make sharded` | Run the benchmark with your last build (`SHARDED_ARGS`) |
| `make status` | Show everyone's jobs and pods, and the nodes, across all live clusters |
| `make logs` | Show recent logs from your latest job (`FOLLOW=1 make logs` to stream) |
| `make pull` | Sync `gs://<bucket>/outputs/` to `./outputs/` |
| `make stop` | Delete your jobs across all live clusters (idle TPU nodes scale to zero about a minute later) |
| `make destroy` | Admin: tear down the GKE clusters ($0 compute) while **keeping** the GCS bucket and outputs |
| `make destroy-all` | Admin: delete **everything** (GKE clusters, GCS bucket, Artifact Registry) |

`make help` lists every target.

## Cost: the cheapest Spot pools, and clusters only while jobs run

TPU Spot prices differ by region far more than between families: on 2026-10-18 a v6e chip cost $0.243/h in
`us-east4`, $0.584 in `us-central1`, `us-east1` and `us-west1`, and $1.62 in `us-east5`. A region is only usable
where a zone offers the family **and** the project has Spot quota for it there: without quota GKE never starts the
node ("GCE quota exceeded") and the job waits until its deadline.

* `make prices` ([`spot_prices.py`](spot_prices.py)) ranks the regions by the current Spot price per chip (Cloud
  Billing Catalog API), marks those whose zones offer the family and where the project has Spot quota (Cloud Quotas
  API), and says how the regions of `gcp.env` compare. Prices change at most monthly; run it then, and set
  `V6E_REGION` / `V5P_REGION` to the cheapest usable region. It needs the Cloud Billing API enabled in the project.
* [`ephemeral.sh`](ephemeral.sh) runs one job on clusters that exist only for it: `make infra` (about 10-15 min),
  `make <target>` with `FOLLOW=1`, then `make destroy` once no MRX job of anyone is active, also after a failure or
  Ctrl-C. It warns when `gcp.env` points at an unusable or overpriced region, stops the job and deletes the clusters
  when no TPU node has started it after `NODE_WAIT_MIN` minutes (default 45) or at once on a quota failure, prints
  each job's Spot cost, and saves each job's full log to `outputs/<job>.log`.
* [`scan.sh`](scan.sh) runs a list of `relax.py` runs (one per line of a file), four to a node, one job after another
  on one set of clusters, and deletes them at the end.

```bash
make prices
bash gke/ephemeral.sh                                         # build and run the ARGS of gcp.env, then delete
RUNS="--newton.maxiter 100; --newton.maxiter 200" bash gke/ephemeral.sh run
bash gke/scan.sh runs.txt                                     # many runs, 4 per node
bash gke/ephemeral.sh down                                    # delete idle clusters left by a crash or a sleep
```

`scripts/bench/compare_runs.py` turns the `relax.json` of runs on any hardware into one table: setup, compile and
steady-state time per step, and the solver time and cost to given force residuals.

## Running jobs

`make deploy` runs the whole loop, so you need not call `run.sh` or `kubectl` yourself:

1. Builds your image from your checkout with Cloud Build (`mrx-tpu:<MRX_USER>`), so your local edits to `mrx/` are what runs.
2. Submits a Kubernetes Job (`run.sh` fills in `job-tpu.yaml`) to every TPU pool that can run it (see "Where to start"); the first to get a node runs it, and the others are deleted.
3. Waits for a TPU node: one takes about 3 minutes to start, longer when a zone has no capacity. The job mounts the bucket for the shared XLA compile cache and the outputs.
4. Streams the logs, waits for the end, copies the outputs to `./outputs/<job>/`, and exits 0 on success or 1 on failure. The node scales back to zero about a minute later.

Other cases:

```bash
ARGS="--geometry.path data/wout_li383_1.4m.nc --geometry.resolution 32 44 20 --descent.method newton --budget.steps 20" make deploy
SCRIPT="scripts/your_script.py" OUTSUB=my-run ARGS="--out /gcs/outputs/my-run" make deploy   # another script: it must write under /gcs/outputs/
make run                                                    # run again with your last build
FOLLOW=0 make run                                           # submit and return; follow with make status / make logs
TPU_GEN=v5p make run                                        # this job on the v5p cluster (when both are set up)
SHARDED_ARGS="--modes shard_map --resolution 32 44 20 --methods newton" make deploy-sharded
```

### Several `relax.py` runs on one node (`RUNS`)

`RUNS` lists up to four runs, separated by `;`. Each run is `relax.py` with `ARGS` followed by its own arguments, which win over `ARGS`, so a run only names what differs:

```bash
RUNS="--geometry.resolution 12 20 12; --geometry.resolution 12 24 12; --geometry.resolution 16 20 12; --geometry.resolution 16 24 12" make deploy
RUNS="--descent.method newton; --descent.method gradient --budget.steps 400" make run
```

* Each run gets its own chip (`TPU_VISIBLE_CHIPS`) and its own output directory, `outputs/<job>/run-1/` to `run-4/`, with `relax.py`'s usual `relax.json` and checkpoints. Log lines start with `[run <n>]`. Tested on v6e and v5p.
* The node is billed until the slowest run ends: group runs of similar size, and give each its own job when sizes differ a lot. Each run must fit in one chip's memory (v6e: 32 GB, v5p: 95 GB).
* A run that ends well is marked done at once. After a Spot preemption the restarted pod skips the runs marked done and resumes the others (see Notes). The job fails if any run fails, after the others have ended.
* On the first run of a new mesh, the runs compile the same kernels at the same time, and JAX may warn `Error reading persistent compilation cache entry`. It then compiles that kernel itself: this costs a little time, never correctness.

### Notes

* **Preemptions:** TPU nodes are Spot. A preempted `relax.py` run resumes from its newest checkpoint that opens, for the steps left of its `--budget.steps`; its record so far is kept as `relax.until-<step>.json`, since `relax.py` rewrites `relax.json` from the restart (unless the preemption cut that file short). A resumed run follows the uninterrupted one closely, not bit for bit. A run with `--seed` starts over instead, because `relax.py` would seed the restart state a second time. A `make deploy-sharded` run starts over.
* **Waiting for a node:** `make deploy` streams the logs if the node starts within 10 minutes; otherwise it waits for the job without logs (`make logs` shows them). A job that stays `Pending` in `make status` waits for Spot capacity at no cost. With `FOLLOW=0` and several pools, a watcher on your machine deletes the other jobs once one gets a node (log: `outputs/<job>-race.log`); if the machine sleeps first, `make stop` deletes the jobs left over.
* **Deadline:** a job stops after 8 hours, restarts included; set `TIMEOUT_MIN` (minutes) for longer runs.
* **Output directories:** a job writes to `outputs/<JOB_NAME>/`, or `outputs/<JOB_NAME>-<family>-<region>/` when it waited in several pools. Reusing a `JOB_NAME` (or `OUTSUB`) resumes the runs already in that directory and skips those that finished, as after a preemption; a job still running under that name is deleted first.
* **Timing:** the first run in a project is the slowest (image build, node start, compilation). Later runs reuse the compile cache in the bucket.
* **Cost:** TPU nodes cost nothing while no job runs. With several pools, a node that a deleted job had already asked for can still start; GKE removes it about 2 minutes after it is ready. Each cluster costs about $0.23/h while it exists (cluster fee and one small system node): `make destroy` when the team is done.
