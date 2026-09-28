"""
dataforseo_wikipedia_check.py

Vérifie, pour chaque hôtel Accor, si un lien wikipedia.org apparaît
dans le top 10 des résultats organiques Google (requête = nom de l'hôtel seul).

Réutilise les patterns validés sur le pipeline de ranking local pack/Maps :
- batch task_post + poll_and_fetch fusionné (draine aussi les orphelines)
- checkpoint incrémental (reprise sans repayer)
- retry réseau sur coupures transitoires
- gestion NaN/pandas (keep_default_na=False, astype(object))
- sanitize JSON (numpy types, NaN)

Entrée attendue : hotels.csv avec au minimum les colonnes :
    - hotel_id (identifiant unique stable, ex: place_id ou Yext ID)
    - hotel_name
    - country_code (ISO2, pour location_code DataForSEO)
    - language_code (ex: "fr", "en" — à défaut "en" par défaut)

Sortie : wikipedia_check_results.csv avec :
    - hotel_id, hotel_name
    - has_wikipedia_top10 (bool)
    - wikipedia_position (1-10, vide si absent)
    - wikipedia_url (vide si absent)
    - status (ok / no_results / error)

Mode test (avant le run complet à 5000) :
    python3 dataforseo_wikipedia_check.py --sample-per-market 10
        -> échantillon stratifié 10/marché sur FR/GB/AU/BR/DE (50 hôtels),
           même logique que la validation à 50 du pipeline ranking.
    python3 dataforseo_wikipedia_check.py --limit 50
        -> simplement les 50 premières lignes de hotels.csv.

Le mode test écrit dans des fichiers séparés (checkpoint/wikipedia_results_test.json,
wikipedia_check_results_test.csv) : aucun risque de mélanger test et run complet,
et aucune tâche de test n'est jamais resoumise lors du run complet.
"""

import os
import csv
import json
import argparse
import random
import time
import base64
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # charge les variables du fichier .env dans os.environ

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATAFORSEO_LOGIN = os.environ["DATAFORSEO_LOGIN"]
DATAFORSEO_PASSWORD = os.environ["DATAFORSEO_PASSWORD"]

API_BASE = "https://api.dataforseo.com/v3"
AUTH_HEADER = {
    "Authorization": "Basic "
    + base64.b64encode(
        f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()
    ).decode(),
    "Content-Type": "application/json",
}

HOTELS_CSV = "hotels.csv"
CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "wikipedia_results.json")
OUTPUT_CSV = "wikipedia_check_results.csv"

# Marchés utilisés pour l'échantillon stratifié de test (mêmes que la
# validation à 50 hôtels du pipeline ranking : FR/GB/AU/BR/DE)
TEST_MARKETS = ["FR", "GB", "AU", "BR", "DE"]

BATCH_SIZE = 100          # tasks par appel task_post (limite pratique DataForSEO)
DEPTH = 10                # top 10 seulement
MAX_WAIT_LOOPS = 60       # budget d'itérations "réellement improductives"
POLL_SLEEP_SECONDS = 5
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10
DEFAULT_LANGUAGE_CODE = "en"

WIKIPEDIA_DOMAIN = "wikipedia.org"


# ---------------------------------------------------------------------------
# Utilitaires réseau (retry sur coupures transitoires)
# ---------------------------------------------------------------------------

def api_post(path, payload):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE}{path}",
                headers=AUTH_HEADER,
                data=json.dumps(sanitize_for_json(payload)),
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(f"  [retry {attempt}/{MAX_RETRIES}] POST {path} : {exc}")
            time.sleep(RETRY_WAIT_SECONDS)
    raise last_exc


def api_get(path):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(f"{API_BASE}{path}", headers=AUTH_HEADER, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(f"  [retry {attempt}/{MAX_RETRIES}] GET {path} : {exc}")
            time.sleep(RETRY_WAIT_SECONDS)
    raise last_exc


# ---------------------------------------------------------------------------
# Sanitize JSON (numpy / NaN)
# ---------------------------------------------------------------------------

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN check sans numpy
        return None
    if hasattr(obj, "item"):  # numpy scalar (int64, float64, bool_...)
        return obj.item()
    return obj


# ---------------------------------------------------------------------------
# Chargement des hôtels
# ---------------------------------------------------------------------------

def load_hotels():
    df = pd.read_csv(HOTELS_CSV, keep_default_na=False, na_values=[""])
    for col in df.columns:
        if df[col].dtype != object:
            df[col] = df[col].astype(object)
    df = df.where(pd.notnull(df), None)

    required = {"hotel_id", "hotel_name", "country_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {HOTELS_CSV} : {missing}")

    if "language_code" not in df.columns:
        df["language_code"] = DEFAULT_LANGUAGE_CODE
    df["language_code"] = df["language_code"].apply(
        lambda v: v if isinstance(v, str) and v else DEFAULT_LANGUAGE_CODE
    )

    hotels = []
    for _, row in df.iterrows():
        hid = row["hotel_id"]
        if not isinstance(hid, str):
            hid = str(hid)
        hotels.append(
            {
                "hotel_id": hid,
                "hotel_name": row["hotel_name"],
                "country_code": row["country_code"],
                "language_code": row["language_code"],
            }
        )
    return hotels


def apply_test_selection(hotels, args):
    """Réduit la liste d'hôtels selon --limit ou --sample-per-market. Ne modifie
    jamais hotels.csv sur disque, uniquement la liste en mémoire pour ce run."""
    if args.sample_per_market:
        random.seed(42)  # reproductible d'un run test à l'autre
        selected = []
        for market in TEST_MARKETS:
            market_hotels = [h for h in hotels if h["country_code"] == market]
            if len(market_hotels) < args.sample_per_market:
                print(
                    f"  ATTENTION : seulement {len(market_hotels)} hôtels dispo "
                    f"pour {market} (demandé : {args.sample_per_market})"
                )
            selected.extend(random.sample(market_hotels, min(args.sample_per_market, len(market_hotels))))
        print(f"Échantillon stratifié : {len(selected)} hôtels sur {TEST_MARKETS}")
        return selected

    if args.limit:
        print(f"Limite appliquée : {args.limit} premiers hôtels de {HOTELS_CSV}")
        return hotels[: args.limit]

    return hotels


def parse_args():
    parser = argparse.ArgumentParser(description="Vérification présence Wikipedia top 10 organique")
    parser.add_argument(
        "--sample-per-market",
        type=int,
        default=None,
        help=f"Échantillon stratifié N hôtels par marché ({'/'.join(TEST_MARKETS)}). Ex: 10 -> 50 hôtels.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ne traite que les N premiers hôtels de hotels.csv (mode test simple).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results):
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(results), f, ensure_ascii=False)
    os.replace(tmp_path, CHECKPOINT_FILE)


# ---------------------------------------------------------------------------
# Soumission des tâches (batch task_post)
# ---------------------------------------------------------------------------

def submit_tasks(hotels, already_done_ids):
    to_submit = [h for h in hotels if h["hotel_id"] not in already_done_ids]
    print(f"À soumettre : {len(to_submit)} / {len(hotels)} (déjà traités : {len(already_done_ids)})")

    pending_tags = set()
    for i in range(0, len(to_submit), BATCH_SIZE):
        batch = to_submit[i : i + BATCH_SIZE]
        payload = []
        for h in batch:
            payload.append(
                {
                    "keyword": h["hotel_name"],
                    "location_name": None,
                    "language_code": h["language_code"],
                    "device": "desktop",
                    "depth": DEPTH,
                    "tag": h["hotel_id"],
                }
            )
        # location_name absent -> DataForSEO utilise le paramètre le plus proche
        # dispo ; si vous avez un mapping country_code -> location_code fiable,
        # remplacez location_name=None par location_code=<code> ici.
        resp = api_post("/serp/google/organic/task_post", payload)
        tasks = resp.get("tasks", [])
        for t in tasks:
            tag = t.get("data", {}).get("tag")
            if tag:
                pending_tags.add(tag)
        print(f"  Lot {i // BATCH_SIZE + 1} soumis ({len(batch)} tâches)")

    return pending_tags


# ---------------------------------------------------------------------------
# Extraction : présence Wikipedia dans le top 10
# ---------------------------------------------------------------------------

def extract_wikipedia_info(task_result):
    """Retourne (position, url) ou (None, None) si absent du top 10."""
    result_list = task_result.get("result") or []
    if not result_list:
        return None, None
    items = result_list[0].get("items") or []
    for item in items:
        if item.get("type") != "organic":
            continue
        rank = item.get("rank_group") or item.get("rank_absolute")
        if rank is not None and rank > DEPTH:
            continue
        domain = (item.get("domain") or "").lower()
        url = item.get("url") or ""
        if WIKIPEDIA_DOMAIN in domain or WIKIPEDIA_DOMAIN in url.lower():
            return rank, url
    return None, None


# ---------------------------------------------------------------------------
# Poll + fetch fusionné (drainage des orphelines inclus)
# ---------------------------------------------------------------------------

def poll_and_fetch(pending_tags, results):
    empty_loops = 0
    while pending_tags and empty_loops < MAX_WAIT_LOOPS:
        ready_resp = api_get("/serp/google/organic/tasks_ready")
        ready_tasks = []
        for t in ready_resp.get("tasks", []):
            ready_tasks.extend(t.get("result") or [])

        if not ready_tasks:
            empty_loops += 1
            time.sleep(POLL_SLEEP_SECONDS)
            continue

        productive = False
        for rt in ready_tasks:
            task_id = rt.get("id")
            tag = rt.get("tag")
            if not task_id:
                continue

            fetched = api_get(f"/serp/google/organic/task_get/advanced/{task_id}")
            task_results = fetched.get("tasks", [])
            if not task_results:
                continue
            task_result = task_results[0]
            resolved_tag = task_result.get("data", {}).get("tag") or tag

            if not resolved_tag:
                continue  # tâche non identifiable, ignorée (pas de crédit perdu, déjà facturée mais non exploitable)

            position, url = extract_wikipedia_info(task_result)
            results[resolved_tag] = {
                "has_wikipedia_top10": position is not None,
                "wikipedia_position": position,
                "wikipedia_url": url,
                "status": "ok",
            }

            if resolved_tag in pending_tags:
                pending_tags.discard(resolved_tag)
                productive = True
            else:
                # orpheline (run précédent interrompu) : gardée quand même,
                # ne consomme pas le budget MAX_WAIT_LOOPS
                productive = True

        save_checkpoint(results)
        print(f"  Checkpoint sauvegardé : {len(results)} résultats, {len(pending_tags)} restants")

        if not productive:
            empty_loops += 1
        else:
            empty_loops = 0

    if pending_tags:
        print(f"ATTENTION : {len(pending_tags)} tâches non résolues après {MAX_WAIT_LOOPS} itérations improductives.")
        print("Relancez le script : il reprendra uniquement sur les tâches manquantes.")

    return results


# ---------------------------------------------------------------------------
# Export final
# ---------------------------------------------------------------------------

def export_csv(hotels, results):
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["hotel_id", "hotel_name", "has_wikipedia_top10", "wikipedia_position", "wikipedia_url", "status"]
        )
        for h in hotels:
            r = results.get(h["hotel_id"]) or {"status": "no_result"}
            writer.writerow(
                [
                    h["hotel_id"],
                    h["hotel_name"],
                    r.get("has_wikipedia_top10", ""),
                    r.get("wikipedia_position", ""),
                    r.get("wikipedia_url", ""),
                    r.get("status", "no_result"),
                ]
            )
    print(f"Export terminé : {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    is_test = bool(args.sample_per_market or args.limit)

    global CHECKPOINT_FILE, OUTPUT_CSV
    if is_test:
        CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "wikipedia_results_test.json")
        OUTPUT_CSV = "wikipedia_check_results_test.csv"
        print("=== MODE TEST (fichiers séparés du run complet) ===\n")

    hotels = load_hotels()
    hotels = apply_test_selection(hotels, args)

    results = load_checkpoint()
    already_done_ids = set(results.keys())

    pending_tags = submit_tasks(hotels, already_done_ids)
    results = poll_and_fetch(pending_tags, results)
    export_csv(hotels, results)

    with_wiki = sum(1 for r in results.values() if r.get("has_wikipedia_top10"))
    print(f"\nBilan : {with_wiki}/{len(results)} hôtels ont un lien Wikipedia dans le top 10.")


if __name__ == "__main__":
    main()