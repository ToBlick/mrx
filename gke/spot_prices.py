#!/usr/bin/env python3
"""Current Spot price per TPU chip-hour by region, for the families the MRX GKE setup can run (v6e, v5p), with the
regions whose zones offer them and where the project has Spot quota, ranked cheapest first; compared with the regions
gcp.env uses (make prices).

    python3 gke/spot_prices.py            # US regions
    python3 gke/spot_prices.py --all      # every region

Prices come from the Cloud Billing Catalog API (Compute Engine SKUs "Tpu... attached to Spot Preemptible VMs"), zone
offers from the Cloud TPU API, both read with your gcloud login. Spot prices change at most once every 30 days, so
running this when a month starts (or before a large campaign) is enough; gke/ephemeral.sh warns when the saved copy
(~/.cache/mrx/spot_prices.json) is older than that. Google bills Spot TPU nodes per second, per chip: a 4-chip node
costs 4x the per-chip price.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import time
import urllib.request

MRX_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The project and the user of gcp.env (gke/settings.sh)
PROJECT, MRX_USER = (subprocess.run(["bash", "-c", 'eval "$(bash gke/settings.sh --export)"; echo "${PROJECT_ID} ${MRX_USER}"'],
                                    capture_output=True, text=True, cwd=MRX_ROOT).stdout.split() + ["", ""])[:2]
COMPUTE_SERVICE = "6F81-5844-456A"
# SKU description prefix -> (family, TPU API accelerator type of one 4-chip node)
FAMILIES = {"TpuV6e": ("v6e", "v6e-4"), "TpuV5p": ("v5p", "v5p-8")}
GCP_ENV = os.path.join(MRX_ROOT, "gcp.env")
CACHE = os.path.expanduser("~/.cache/mrx/spot_prices.json")


def gcloud(*args):
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=True).stdout


def spot_prices(token):
    """{family: {region: USD per chip-hour}} for Spot TPUs."""
    prices, page = {}, ""
    while True:
        url = (f"https://cloudbilling.googleapis.com/v1/services/{COMPUTE_SERVICE}/skus?pageSize=5000&currencyCode=USD"
               + (f"&pageToken={page}" if page else ""))
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "x-goog-user-project": PROJECT})
        data = json.load(urllib.request.urlopen(req))
        for sku in data.get("skus", []):
            m = re.match(r"(TpuV6e|TpuV5p) attached to Spot Preemptible VMs", sku["description"])
            if not m:
                continue
            unit = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"][-1]["unitPrice"]
            usd = int(unit.get("units", 0)) + unit.get("nanos", 0) / 1e9
            for region in sku["serviceRegions"]:
                prices.setdefault(FAMILIES[m.group(1)][0], {})[region] = usd
        page = data.get("nextPageToken", "")
        if not page:
            return prices


def offering_zones(regions):
    """{region: {family: [zones]}} from the Cloud TPU API."""
    zones = [z for z in gcloud("compute", "zones", "list", f"--project={PROJECT}", "--filter=status=UP",
                               "--format=value(name)").split() if z.rsplit("-", 1)[0] in regions]

    def types(zone):
        out = subprocess.run(["gcloud", "compute", "tpus", "accelerator-types", "list", f"--zone={zone}",
                              f"--project={PROJECT}", "--format=value(type)"], capture_output=True, text=True).stdout
        return zone, set(out.split())

    offers = {}
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for zone, have in pool.map(types, zones):
            for family, accel in FAMILIES.values():
                if accel in have:
                    offers.setdefault(zone.rsplit("-", 1)[0], {}).setdefault(family, []).append(zone.rsplit("-", 1)[1])
    return offers


def spot_quota(family):
    """{region: Spot chips the project may use} from Cloud Quotas: a region missing from the answer has none."""
    out = subprocess.run(["gcloud", "beta", "quotas", "info", "describe",
                          f"PREEMPTIBLE-TPU-{family.upper()}-per-project-region", "--service=compute.googleapis.com",
                          f"--project={PROJECT}", "--format=json"], capture_output=True, text=True).stdout
    quota = {}
    for dim in json.loads(out or "{}").get("dimensionsInfos", []):
        value = int(dim["details"].get("value") or 0)
        for region in dim.get("applicableLocations", []):
            quota[region] = value
    return quota


def configured():
    """The regions gcp.env enables, as {family: [regions]} (main region first)."""
    env = {}
    if os.path.exists(GCP_ENV):
        for line in open(GCP_ENV):
            m = re.match(r'\s*export (\w+)="(?:\$\{\w+:-)?([^"}]*)\}?"', line)
            if m:
                env[m.group(1)] = m.group(2)
    out = {}
    if env.get("ENABLE_TPU_V6E") == "true":
        out["v6e"] = [env.get("V6E_REGION", "us-east1"), env.get("V6E_BACKUP_REGION", "us-central1")]
    if env.get("ENABLE_TPU_V5P") == "true":
        out["v5p"] = [env.get("V5P_REGION", "us-east5")]
    return out


def check():
    """Warnings for gke_ephemeral.sh, from the saved prices: a stale copy, or gcp.env not in the cheapest region."""
    if not os.path.exists(CACHE):
        print(f"no saved Spot prices: run python3 {__file__}")
        return
    cache = json.load(open(CACHE))
    age = (time.time() - time.mktime(time.strptime(cache["date"], "%Y-%m-%d"))) / 86400
    if age > 30:
        print(f"Spot prices saved {age:.0f} days ago (they change monthly): run python3 {__file__}")
    for family, regions in configured().items():
        prices = cache["spot_usd_per_chip_hour"].get(family, {})
        offered = [(p, r) for r, p in prices.items() if cache["offers"].get(r, {}).get(family)]
        if not cache["offers"].get(regions[0], {}).get(family):
            print(f"{family} in {regions[0]} is NOT usable (no zone offers it, or no Spot quota): jobs would wait "
                  f"forever. Pick a region marked * by python3 {__file__}")
            continue
        if not offered or regions[0] not in prices:
            continue
        usd, best = min(offered)
        if prices[regions[0]] > 1.05 * usd:
            print(f"{family} in {regions[0]} costs {prices[regions[0]]:.3f}/chip-h, {prices[regions[0]] / usd:.1f}x "
                  f"{best} ({usd:.3f}): consider {family.upper()}_REGION={best} in gcp.env")


def cost_since(since):
    """Chip-hours and Spot cost of the MRX pods of $MRX_USER that started after `since` (ISO 8601, UTC)."""
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {"spot_usd_per_chip_hour": {}}
    user = os.environ.get("MRX_USER") or MRX_USER
    clusters = gcloud("container", "clusters", "list", f"--project={PROJECT}", "--format=value(name,location)").split("\n")
    total = 0.0
    for line in filter(None, clusters):
        name, region = line.split()
        if not name.startswith("mrx-tpu-cluster"):
            continue
        out = subprocess.run(["kubectl", f"--context=gke_{PROJECT}_{region}_{name}", "get", "pods", "-n", "default",
                              "-l", f"app=mrx,mrx-user={user}", "-o", "json"], capture_output=True, text=True).stdout
        for pod in json.loads(out or '{"items": []}')["items"]:
            start = pod["status"].get("startTime")
            if not start or start < since:
                continue
            ends = [c["state"]["terminated"]["finishedAt"] for c in pod["status"].get("containerStatuses", [])
                    if "terminated" in c.get("state", {})]
            end = max(ends) if ends else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            secs = (time.mktime(time.strptime(end, "%Y-%m-%dT%H:%M:%SZ"))
                    - time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ")))
            family = pod["spec"].get("nodeSelector", {}).get("cloud.google.com/compute-class", "").replace("mrx-tpu-", "")
            usd = cache["spot_usd_per_chip_hour"].get(family, {}).get(region)
            cost = 4 * secs / 3600 * usd if usd else float("nan")
            total += cost if usd else 0
            print(f"  {pod['metadata']['name']}: {family} {region}, {secs / 60:.1f} min on the node x 4 chips"
                  f" x {usd if usd else '?'}/chip-h = ${cost:.2f}")
    print(f"  total ${total:.2f} (Spot, pod run time; the node also bills ~3 min to boot and ~1 min idle)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="every region, not only us-*")
    ap.add_argument("--check", action="store_true", help="only the warnings, from the saved prices (no API calls)")
    ap.add_argument("--cost-since", metavar="ISO", help="cost of $MRX_USER's MRX pods started after this UTC time")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.cost_since:
        return cost_since(a.cost_since)
    prices = spot_prices(gcloud("auth", "print-access-token").strip())
    regions = {r for fam in prices.values() for r in fam if a.all or r.startswith("us-")}
    offers = offering_zones(regions)
    quotas = {family: spot_quota(family) for family, _ in FAMILIES.values()}
    # A region without Spot quota for a family cannot start its nodes (GKE: "GCE quota exceeded"): not usable
    for region, fams in offers.items():
        for family in list(fams):
            if quotas[family].get(region, 0) < 4:
                fams[family] = []
    conf = configured()
    # Read by gke/ephemeral.sh for its staleness warning and the cost of each job
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump({"date": time.strftime("%Y-%m-%d"), "spot_usd_per_chip_hour": prices, "offers": offers}, fh, indent=1)
    for family in ("v6e", "v5p"):
        print(f"\nTPU {family} Spot, USD per chip-hour (x4 per node-hour); * = usable (zones offer it and the project "
              f"has Spot quota), < = in gcp.env")
        rows = sorted((p, r) for r, p in prices.get(family, {}).items() if r in regions)
        best = None
        for usd, region in rows:
            zones = offers.get(region, {}).get(family, [])
            if zones and best is None:
                best = (usd, region)
            mark = "<" if region in conf.get(family, []) else " "
            offered = family in offers.get(region, {})
            why = ",".join(sorted(zones)) if zones else ("no Spot quota" if offered else "not offered")
            print(f"  {'*' if zones else ' '}{mark} {region:<24} {usd:7.4f}  ({4 * usd:6.3f}/node-h)  {why}")
        if best:
            print(f"  cheapest usable: {best[1]} at {best[0]:.4f}/chip-h")
            for region in conf.get(family, [])[:1]:
                if not offers.get(region, {}).get(family):
                    print(f"  gcp.env uses {region}: NOT usable (no zone or no Spot quota)")
                elif region in prices.get(family, {}) and region != best[1]:
                    ratio = prices[family][region] / best[0]
                    print(f"  gcp.env uses {region}: {ratio:.1f}x the cheapest usable region")


if __name__ == "__main__":
    main()
