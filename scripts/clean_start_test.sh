#!/usr/bin/env bash
# Clean-room startup: tear everything down, bring it up from nothing, and prove
# it works.
#
# The claim this verifies is the one every README makes and few can support:
# that a stranger with Docker and this repository ends up with a working system.
# It runs against *destroyed* volumes, so nothing carried over from a previous
# run can make it pass.
#
#   ./scripts/clean_start_test.sh          full run, ~10 minutes
#   ./scripts/clean_start_test.sh --keep   leave the stack up afterwards
#
# DESTRUCTIVE: removes this project's Docker volumes, which is the whole point.

set -uo pipefail
cd "$(dirname "$0")/.."

KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

PASS=0; FAIL=0
declare -a RESULTS=()

step() {
  local name="$1"; shift
  printf '\n\033[36m==> %s\033[0m\n' "$name"
  if "$@"; then
    PASS=$((PASS + 1)); RESULTS+=("PASS  $name")
  else
    FAIL=$((FAIL + 1)); RESULTS+=("FAIL  $name")
  fi
}

wait_for_health() {
  echo "  waiting for the API to become healthy (up to 5 minutes)…"
  for _ in $(seq 1 100); do
    if curl -sf http://localhost:8000/health/live >/dev/null 2>&1; then
      echo "  API is live"
      return 0
    fi
    sleep 3
  done
  echo "  API never became healthy"
  docker compose logs api --tail 40
  return 1
}

# --- 1. destroy ---------------------------------------------------------------
# --volumes is what makes this a clean-room test rather than a restart. Without
# it Postgres and Neo4j keep their data and the run proves nothing about a first
# install. The model cache volume is kept deliberately: re-downloading ~120 MB of
# ONNX weights tests Hugging Face's availability, not this system.
step "Tear down (volumes destroyed)" \
  docker compose down --volumes --remove-orphans

# --- 2. build and start -------------------------------------------------------
step "Build images" docker compose build
step "Start stack" docker compose up -d
step "API becomes healthy" wait_for_health

# --- 3. schema ----------------------------------------------------------------
# Migrations run on startup; this asserts they actually reached a good state
# rather than silently degrading.
step "Schema applied cleanly" docker compose exec -T api python scripts/_clean_start_checks.py health
step "All three stores reachable" docker compose exec -T api python scripts/_clean_start_checks.py dependencies
# --- 4. generate and ingest ---------------------------------------------------
# reportlab is a dev-only dependency: it exists to *produce* the test PDFs and is
# deliberately absent from the runtime image, which only needs to read them. A
# genuinely clean environment therefore has to install it before this step, and
# the first run of this test found exactly that gap.
step "Generate synthetic corpus" bash -c '
  docker compose exec -T api pip install --quiet reportlab==4.2.5 >/dev/null 2>&1
  docker compose exec -T api python data/synthetic/generate.py >/dev/null &&
  docker compose exec -T api python data/synthetic/generate_pdfs.py >/dev/null &&
  echo "  corpus generated (Markdown, CSV and real PDFs)"'

step "Ingest Markdown and CSV" bash -c '
  docker compose exec -T api python scripts/ingest_dir.py data/synthetic/generated \
    --data-class synthetic_test_data --source-system synthetic_cmms --wait 2>&1 | tail -3'

step "Ingest PDFs (incl. a scanned page and a P&ID)" bash -c '
  docker compose exec -T api python scripts/ingest_dir.py data/synthetic/generated_pdf \
    --data-class synthetic_test_data --source-system pdf_corpus --wait 2>&1 | tail -3'

step "Load the requirement corpus" bash -c '
  docker compose exec -T api python scripts/load_requirements.py 2>&1 | tail -2'

# --- 5. the pipeline actually produced something ------------------------------
step "Documents ingested" docker compose exec -T api python scripts/_clean_start_checks.py documents
step "Entities resolved into the graph" docker compose exec -T api python scripts/_clean_start_checks.py entities
step "P&ID digitised" docker compose exec -T api python scripts/_clean_start_checks.py drawings
step "Structured records extracted" docker compose exec -T api python scripts/_clean_start_checks.py incidents
# --- 6. the read path works ---------------------------------------------------
step "Copilot answers with citations" docker compose exec -T api python scripts/_clean_start_checks.py answer
step "Copilot abstains on an unknown asset" docker compose exec -T api python scripts/_clean_start_checks.py abstains
step "RCA ranks causes from evidence" docker compose exec -T api python scripts/_clean_start_checks.py rca
step "Compliance evaluates against records" docker compose exec -T api python scripts/_clean_start_checks.py compliance
# --- 7. every surface agrees --------------------------------------------------
step "All surfaces share one backend" bash -c '
  docker compose exec -T api python scripts/verify_surfaces.py 2>&1 | tail -3'

step "Full test suite" bash -c '
  docker compose exec -T api pip install --quiet pytest==8.3.4 pytest-asyncio==0.25.2 >/dev/null 2>&1
  docker compose exec -T api python -m pytest tests -q -p no:cacheprovider 2>&1 | tail -2'

# --- summary ------------------------------------------------------------------
printf '\n\033[36m%s\033[0m\n' "────────────────────────────────────────────────────────────"
for line in "${RESULTS[@]}"; do
  if [[ "$line" == PASS* ]]; then printf '  \033[32m%s\033[0m\n' "$line"
  else printf '  \033[31m%s\033[0m\n' "$line"; fi
done
printf '\n  %d passed, %d failed\n\n' "$PASS" "$FAIL"

if [[ "$KEEP" -eq 0 ]]; then
  echo "  Leaving the stack running. Use 'docker compose down --volumes' to clean up,"
  echo "  or open http://localhost:8000 to look around."
fi

exit $(( FAIL > 0 ? 1 : 0 ))
