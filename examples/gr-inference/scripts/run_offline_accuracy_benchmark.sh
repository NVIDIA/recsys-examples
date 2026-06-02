#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common_paths.sh
source "${SCRIPT_DIR}/common_paths.sh"
cd "${GR_INFERENCE_REPO_ROOT}"

MODEL_DIR="${MODEL_DIR:-${GR_MODEL_DIR:-}}"
MODEL="${MODEL:-${GR_MODEL:-}}"
MODEL_REVISION="${MODEL_REVISION:-${GR_MODEL_REVISION:-}}"
MODEL_DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-${GR_MODEL_DOWNLOAD_DIR:-}}"
if [[ -z "${MODEL_VARIANT:-}" ]]; then
  if [[ -n "${MODEL}" ]]; then
    model_ref_for_variant="${MODEL%/}"
    MODEL_VARIANT="${model_ref_for_variant##*/}"
  else
    MODEL_VARIANT="Qwen3-1.7B"
  fi
fi
DEFAULT_MODEL_DIR="$(gr_default_model_dir "${MODEL_VARIANT}")"
if [[ -z "${MODEL_DIR}" && -z "${MODEL}" ]]; then
  if [[ -d "${DEFAULT_MODEL_DIR}" ]]; then
    MODEL_DIR="${DEFAULT_MODEL_DIR}"
  else
    MODEL="$(gr_default_hf_model_id "${MODEL_VARIANT}")"
  fi
fi
if [[ -z "${MODEL_DIR}" && -z "${MODEL_DOWNLOAD_DIR}" ]]; then
  MODEL_DOWNLOAD_DIR="$(gr_model_download_dir "${MODEL}" "${MODEL_ROOT:-${GR_INFERENCE_WORKSPACE_ROOT}/models}" "${MODEL_VARIANT}")"
fi
SGLANG_REPO="${SGLANG_REPO:-$(gr_default_sglang_repo)}"
SGLANG_PYTHON="${SGLANG_PYTHON:-$(gr_default_sglang_python)}"
GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT:-$(gr_default_decode_atten_root)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-benchmark_artifacts/sglang_compare/offline_accuracy_${RUN_ID}}"

CONTEXT_LENS="${CONTEXT_LENS:-1000 5000}"
BEAM_WIDTHS="${BEAM_WIDTHS:-256}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
GR_DECODE_STEPS="${GR_DECODE_STEPS:-2}"
SGLANG_DECODE_STEPS="${SGLANG_DECODE_STEPS:-3}"
CORRECTNESS_REPEAT="${CORRECTNESS_REPEAT:-1}"

mkdir -p "${OUT_DIR}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
gr_setup_local_cache_env
gr_require_sglang_repo "${SGLANG_REPO}"
if [[ -z "${MODEL_DIR}" && -n "${MODEL_DOWNLOAD_DIR}" ]]; then
  echo "model_download_dir: ${MODEL_DOWNLOAD_DIR}"
fi
MODEL_DIR="$(gr_resolve_model_dir "${MODEL_DIR}" "${MODEL}" "${MODEL_REVISION}" "$(gr_default_hf_model_id "${MODEL_VARIANT}")" "${MODEL_DOWNLOAD_DIR}")"

echo "== Offline accuracy benchmark =="
echo "model: ${MODEL_DIR}"
if [[ -n "${MODEL}" ]]; then
  echo "model_ref: ${MODEL}"
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  echo "model_revision: ${MODEL_REVISION}"
fi
echo "sglang_repo: ${SGLANG_REPO}"
echo "out_dir: ${OUT_DIR}"
echo "repo: ${GR_INFERENCE_REPO_ROOT}"
echo "run_id: ${RUN_ID}"
echo "contexts: ${CONTEXT_LENS}; beams: ${BEAM_WIDTHS}; batches: ${BATCH_SIZES}"
echo

for context_len in ${CONTEXT_LENS}; do
  for beam_width in ${BEAM_WIDTHS}; do
    for requests in ${BATCH_SIZES}; do
      suffix="ctx${context_len}_beam${beam_width}_req${requests}"
      workload_jsonl="${OUT_DIR}/qwen3_ctx${context_len}_req${requests}.jsonl"
      gr_json="${OUT_DIR}/gr_${suffix}.json"
      sglang_json="${OUT_DIR}/sglang_${suffix}.json"
      report_json="${OUT_DIR}/compare_${suffix}.json"
      report_md="${OUT_DIR}/compare_${suffix}.md"
      gr_log="${LOG_DIR}/gr_${suffix}.log"
      sglang_log="${LOG_DIR}/sglang_${suffix}.log"
      compare_log="${LOG_DIR}/compare_${suffix}.log"

      echo "== ${suffix} =="
      PYTHONPATH=src \
      python tools/make_qwen3_beam_workload.py \
        --model-dir "${MODEL_DIR}" \
        --context-len "${context_len}" \
        --requests "${requests}" \
        --no-tokenizer \
        --output-jsonl "${workload_jsonl}"

      echo "[GR] ${gr_json} (log: ${gr_log})"
      if ! PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
        GR_DECODE_ATTEN_ROOT="${GR_DECODE_ATTEN_ROOT}" \
        python tools/run_qwen3_real_weight_serving.py \
          --model-dir "${MODEL_DIR}" \
          --workload-jsonl "${workload_jsonl}" \
          --context-len "${context_len}" \
          --decode-steps "${GR_DECODE_STEPS}" \
          --beam-width "${beam_width}" \
          --requests "${requests}" \
          --max-batch-size "${requests}" \
          --batched-decode \
          --continuous \
          --beam-kv-pool-capacity "${requests}" \
          --context-kv-pool-capacity "${requests}" \
          --decode-backend real \
          --device cuda \
          --return-beam-details \
          --record-outputs \
          --warmup-runs 1 \
          --repeat "${CORRECTNESS_REPEAT}" \
          --output-json "${gr_json}" >"${gr_log}" 2>&1; then
        echo "GR accuracy run failed for ${suffix}; tail of ${gr_log}:" >&2
        tail -n 80 "${gr_log}" >&2 || true
        exit 1
      fi

      echo "[SGLang] ${sglang_json} (log: ${sglang_log})"
      if ! PYTHONPATH="${SGLANG_REPO}/python:src:${PYTHONPATH:-}" \
        "${SGLANG_PYTHON}" tools/run_sglang_beam_benchmark.py \
          --model-dir "${MODEL_DIR}" \
          --sglang-repo "${SGLANG_REPO}" \
          --workload-jsonl "${workload_jsonl}" \
          --context-len "${context_len}" \
          --decode-steps "${SGLANG_DECODE_STEPS}" \
          --beam-width "${beam_width}" \
          --requests "${requests}" \
          --arrival-mode batch \
          --disable-radix-cache \
          --warmup-runs 1 \
          --repeat "${CORRECTNESS_REPEAT}" \
          --use-input-ids \
          --no-tokenizer \
          --output-json "${sglang_json}" >"${sglang_log}" 2>&1; then
        echo "SGLang accuracy run failed for ${suffix}; tail of ${sglang_log}:" >&2
        tail -n 80 "${sglang_log}" >&2 || true
        exit 1
      fi

      echo "[compare] ${report_md} (log: ${compare_log})"
      if ! PYTHONPATH=src \
        python tools/compare_gr_sglang_beam.py \
          --gr-json "${gr_json}" \
          --sglang-json "${sglang_json}" \
          --output-json "${report_json}" \
          --output-markdown "${report_md}" >"${compare_log}" 2>&1; then
        echo "Comparison failed for ${suffix}; tail of ${compare_log}:" >&2
        tail -n 80 "${compare_log}" >&2 || true
        exit 1
      fi
    done
  done
done

python - "${OUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("compare_ctx*_beam*_req*.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    workload = report["workload"]
    correctness = report["correctness"]
    rows.append(
        {
            "context_len": workload.get("context_len"),
            "beam_width": workload.get("beam_width"),
            "batch_requests": workload.get("matched_requests"),
            "top1_exact_match_rate": correctness.get("top1_exact_match_rate"),
            "topk_set_overlap_mean": correctness.get("topk_set_overlap_mean"),
            "ordered_prefix_match_mean": correctness.get("ordered_prefix_match_mean"),
            "token_length_match_rate": correctness.get("token_length_match_rate"),
            "output_token_budget_match": workload.get("output_token_budget_match"),
            "report": path.name.replace(".json", ".md"),
        }
    )

rows.sort(key=lambda row: (int(row["context_len"]), int(row["beam_width"]), int(row["batch_requests"])))

fieldnames = [
    "context_len",
    "beam_width",
    "batch_requests",
    "top1_exact_match_rate",
    "topk_set_overlap_mean",
    "ordered_prefix_match_mean",
    "token_length_match_rate",
    "output_token_budget_match",
    "report",
]
with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

lines = [
    "# GR vs SGLang Offline Accuracy",
    "",
    "| ctx | beam | batch | top1 exact | topK overlap | ordered prefix | token len match | output budget match | report |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
]
for row in rows:
    lines.append(
        "| {ctx} | {beam} | {batch} | {top1} | {topk} | {ordered} | {token_len} | {budget} | `{report}` |".format(
            ctx=row["context_len"],
            beam=row["beam_width"],
            batch=row["batch_requests"],
            top1=fmt(row["top1_exact_match_rate"]),
            topk=fmt(row["topk_set_overlap_mean"]),
            ordered=fmt(row["ordered_prefix_match_mean"]),
            token_len=fmt(row["token_length_match_rate"]),
            budget=fmt(row["output_token_budget_match"]),
            report=row["report"],
        )
    )
lines.extend(
    [
        "",
        "Metric notes:",
        "- top1 exact: whether GR and SGLang rank-1 token IDs match.",
        "- topK overlap: set overlap between GR and SGLang beam candidates.",
        "- output budget match should be true for the fixed-length offline benchmark.",
        "",
    ]
)
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print((root / "summary.md").read_text(encoding="utf-8"))
PY

echo
echo "Offline accuracy summary:"
echo "  ${OUT_DIR}/summary.md"
echo "  ${OUT_DIR}/summary.csv"
