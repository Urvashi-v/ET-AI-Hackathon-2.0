#!/usr/bin/env bash
# One-shot verification: bring the stack up, load reference data, ingest the
# corpus, and run every check. Prints a summary of what actually passed.
#
#   ./scripts/verify.sh            full run
#   ./scripts/verify.sh --quick    skip Docker; static checks and unit tests only
#
# Written for Git Bash on Windows and for Linux CI alike.

set -uo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

# Prefer the project virtualenv when present, so a bare `python` on PATH that
# lacks the dependencies does not produce a confusing failure.
if [[ -x ".venv/Scripts/python.exe" ]]; then PY=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]];      then PY=".venv/bin/python"
else PY="python"; fi

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

step "Ruff lint"            "$PY" -m ruff check services eval tests data scripts
step "Ruff format"          "$PY" -m ruff format --check services eval tests data scripts
step "mypy"                 "$PY" -m mypy services
step "Unit tests"           "$PY" -m pytest -q -m "not integration"
step "Requirement provenance" "$PY" scripts/load_requirements.py --dry-run

if [[ $QUICK -eq 0 ]]; then
  step "Docker stack"       docker compose up -d --build --wait --wait-timeout 300
  step "Health report"      bash -c "curl -fsS localhost:8000/health | $PY -m json.tool >/dev/null"
  step "Load requirements"  docker compose exec -T api python scripts/load_requirements.py
  step "Generate corpus"    "$PY" data/synthetic/generate.py
  step "Ingest corpus"      "$PY" scripts/ingest_dir.py data/synthetic/generated \
                                --data-class synthetic_test_data --wait
  step "Idempotent re-ingest" "$PY" scripts/ingest_dir.py data/synthetic/generated \
                                --data-class synthetic_test_data --wait
  step "API contract tests" "$PY" -m pytest -q -m integration
  step "Evaluation harness" "$PY" eval/run_eval.py --api http://localhost:8000 --tag verify
fi

printf '\n\033[36m==> Summary\033[0m\n'
for line in "${RESULTS[@]}"; do
  if [[ "$line" == PASS* ]]; then printf '  \033[32m%s\033[0m\n' "$line"
  else printf '  \033[31m%s\033[0m\n' "$line"; fi
done
printf '\n  %d passed, %d failed\n\n' "$PASS" "$FAIL"
exit $(( FAIL > 0 ? 1 : 0 ))
