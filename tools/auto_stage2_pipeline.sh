#!/usr/bin/env bash
# Wait for the active Point2RBox-v2 stage-1 run, then autonomously generate
# pseudo labels, train/evaluate/test stage-2, and collect DOTA submission zips.
set -Eeuo pipefail

REPO=/root/work/A/Point2RBox-v2-jittor
CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh
ENV_NAME=p2r-jittor
GPU_ID=0
STAGE1_PID=${STAGE1_PID:-1615615}
STAGE1_WORK="$REPO/work_dirs/point2rbox_v2_1x_dota"
STAGE1_CKPT="$STAGE1_WORK/checkpoints/ckpt_12.pkl"
STAGE1_ZIP="$REPO/submit_zips/point2rbox_v2_1x_dota.zip"
PSEUDO_TMP_PREFIX=/root/data/split_ss_dota/point2rbox_v2_pseudo_labels_jittor_final
PSEUDO_TMP="${PSEUDO_TMP_PREFIX}.bbox.json"
PSEUDO_CANON=/root/data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json
STAGE2_CONFIG=configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py
STAGE2_WORK="$REPO/work_dirs/rotated_fcos_1x_dota_using_pseudo"
STAGE2_CKPT="$STAGE2_WORK/checkpoints/ckpt_12.pkl"
STAGE2_ZIP="$REPO/submit_zips/rotated_fcos_1x_dota_using_pseudo.zip"
OUT_DIR=/root/work/A/DOTA_SUBMISSIONS
STATE_DIR="$REPO/work_dirs/auto_stage2_pipeline"
LOG="$STATE_DIR/pipeline.log"
LOCK="$STATE_DIR/pipeline.lock"

mkdir -p "$STATE_DIR" "$OUT_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "another auto_stage2_pipeline instance is already running" >&2
    exit 2
fi
exec > >(tee -a "$LOG") 2>&1

log() { printf '[%s] %s\n' "$(date -u '+%F %T UTC')" "$*"; }
fail() { log "FATAL: $*"; exit 1; }
trap 'rc=$?; if (( rc != 0 )); then log "pipeline exited with rc=$rc"; fi' EXIT

stage1_is_alive() {
    kill -0 "$STAGE1_PID" 2>/dev/null || return 1
    local cmd
    cmd=$(tr '\0' ' ' <"/proc/$STAGE1_PID/cmdline" 2>/dev/null || true)
    [[ "$cmd" == *"run_net.py"* &&
       "$cmd" == *"point2rbox_v2_1x_dota.py"* &&
       "$cmd" == *"--task train"* ]]
}

validate_stage1_map() {
    python - "$STAGE1_WORK" <<'PY'
import pathlib, re, sys
work = pathlib.Path(sys.argv[1])
matches = []
for path in work.glob('textlog/*.txt'):
    for line in path.read_text(errors='ignore').splitlines():
        if 'eval/0_meanAP:' in line and 'iter:76800' in line:
            m = re.search(r'eval/0_meanAP:([0-9.eE+-]+)', line)
            if m:
                matches.append((float(m.group(1)), path))
if not matches:
    raise SystemExit('final stage-1 mAP at iter 76800 not found')
value, path = matches[-1]
print(f'final stage-1 mAP50={value:.6f} from {path}')
if value < 0.535:
    raise SystemExit(f'stage-1 quality gate failed: {value:.6f} < 0.535')
PY
}

validate_pseudo() {
    python - "$PSEUDO_TMP" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
images = {row['image_id'] for row in data}
required = {'image_id', 'bbox', 'score', 'category_id'}
if len(data) != 245953:
    raise SystemExit(f'pseudo count mismatch: {len(data)} != 245953')
if len(images) != 12800:
    raise SystemExit(f'pseudo image count mismatch: {len(images)} != 12800')
if any(set(row) != required or len(row['bbox']) != 5 for row in data):
    raise SystemExit('pseudo schema mismatch')
print(f'pseudo labels valid: {len(data)} boxes / {len(images)} images')
PY
}

collect_zip() {
    local src=$1 dst=$2
    [[ -s "$src" ]] || fail "submission zip missing or empty: $src"
    cp -f "$src" "$dst"
    unzip -tq "$dst" >/dev/null || fail "corrupt submission zip: $dst"
    sha256sum "$dst" | tee "$dst.sha256"
}

cd "$REPO"
source "$CONDA_SH"
conda activate "$ENV_NAME"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$REPO/python"
export PYTHONUNBUFFERED=1

if [[ ! -f "$STATE_DIR/stage1.done" ]]; then
    log "waiting for stage-1 pid=$STAGE1_PID to finish naturally"
    while stage1_is_alive; do
        latest=$(grep -h 'name:point2rbox_v2_1x_dota' \
            "$STAGE1_WORK"/textlog/*.txt 2>/dev/null | tail -1 || true)
        log "stage-1 alive; ${latest:-no log line yet}"
        sleep 60
    done
    [[ -s "$STAGE1_CKPT" ]] || fail "stage-1 exited without ckpt_12"
    validate_stage1_map
    collect_zip "$STAGE1_ZIP" "$OUT_DIR/point2rbox_v2_stage1_e2e.zip"
    touch "$STATE_DIR/stage1.done"
    log "stage-1 accepted"
fi

if [[ ! -f "$STATE_DIR/pseudo.done" ]]; then
    log "generating final Jittor pseudo labels"
    python tools/generate_pseudo_labels.py \
        --config configs/point2rbox_v2/point2rbox_v2_pseudo_generator_dota.py \
        --ckpt "$STAGE1_CKPT" \
        --out "$PSEUDO_TMP_PREFIX"
    validate_pseudo
    if [[ -e "$PSEUDO_CANON" &&
          ! -e "${PSEUDO_CANON}.pre_jittor_final_backup" ]]; then
        cp -a "$PSEUDO_CANON" "${PSEUDO_CANON}.pre_jittor_final_backup"
    fi
    cp -f "$PSEUDO_TMP" "${PSEUDO_CANON}.tmp"
    mv -f "${PSEUDO_CANON}.tmp" "$PSEUDO_CANON"
    touch "$STATE_DIR/pseudo.done"
    log "canonical pseudo labels installed at $PSEUDO_CANON"
fi

if [[ ! -f "$STATE_DIR/stage2.done" ]]; then
    log "starting/resuming stage-2 on physical GPU $GPU_ID"
    python tools/run_net.py --config-file "$STAGE2_CONFIG" --task train
    [[ -s "$STAGE2_CKPT" ]] || fail "stage-2 exited without ckpt_12"
    [[ -s "$STAGE2_WORK/test/test_12.pkl" ]] || \
        fail "stage-2 exited without test_12.pkl"
    collect_zip "$STAGE2_ZIP" \
        "$OUT_DIR/point2rbox_v2_stage2_pseudo_fcos.zip"
    touch "$STATE_DIR/stage2.done"
    log "stage-2 train/val/test/merge complete"
fi

printf '%s\n' \
    'Point2RBox-v2 Jittor DOTA-v1.0 submissions' \
    '' \
    'point2rbox_v2_stage1_e2e.zip' \
    '  End-to-end Point2RBox-v2 stage-1 test submission.' \
    '' \
    'point2rbox_v2_stage2_pseudo_fcos.zip' \
    '  Rotated FCOS stage-2 test submission trained from final Jittor pseudo labels.' \
    '' \
    "Generated by commit: $(git rev-parse HEAD)" \
    "Completed at: $(date -u '+%F %T UTC')" \
    >"$OUT_DIR/README.txt"
log "ALL DONE; submissions are in $OUT_DIR"
