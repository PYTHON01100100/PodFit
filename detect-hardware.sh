#!/usr/bin/env bash
# Detects this host's GPUs and writes a seed rig that podfit loads at startup.
# Safe to re-run: the app only seeds a rig id it has never seen before.
set -euo pipefail
OUT="$(dirname "$0")/data/detected_hardware.json"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found - skipping hardware detection." >&2
  exit 0
fi

mapfile -t LINES < <(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version \
                                --format=csv,noheader,nounits)
COUNT=${#LINES[@]}
[ "$COUNT" -eq 0 ] && { echo "no GPUs reported" >&2; exit 0; }

IFS=',' read -r NAME MEM CC DRV <<<"${LINES[0]}"
NAME="$(echo "$NAME" | xargs)"; MEM="$(echo "$MEM" | xargs)"
CC="$(echo "$CC" | xargs)";     DRV="$(echo "$DRV" | xargs)"
VRAM_GIB=$(awk -v m="$MEM" 'BEGIN{printf "%.1f", m/1024}')
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)

# Map the reported name onto a catalog entry where we can.
lower=$(echo "$NAME" | tr 'A-Z' 'a-z')
case "$lower" in
  *"l20"*)      GPU_ID=l20 ;;      *"l40s"*)  GPU_ID=l40s ;;
  *"l40"*)      GPU_ID=l40 ;;      *"l4"*)    GPU_ID=l4 ;;
  *"h200"*)     GPU_ID=h200-sxm ;; *"h20"*)   GPU_ID=h20 ;;
  *"h800"*)     GPU_ID=h800-sxm ;; *"h100"*)  GPU_ID=h100-sxm ;;
  *"b200"*)     GPU_ID=b200 ;;
  *"a100"*)     if [ "${MEM%%.*}" -gt 60000 ]; then GPU_ID=a100-80; else GPU_ID=a100-40; fi ;;
  *"a6000"*)    GPU_ID=a6000 ;;    *"a10"*)   GPU_ID=a10 ;;
  *"4090"*)     GPU_ID=rtx-4090 ;; *"5090"*)  GPU_ID=rtx-5090 ;;
  *"3090"*)     GPU_ID=rtx-3090 ;; *"v100"*)  GPU_ID=v100-32 ;;
  *"t4"*)       GPU_ID=t4 ;;       *"mi300x"*) GPU_ID=mi300x ;;
  *)            GPU_ID="" ;;
esac

# NVLink present?
if nvidia-smi nvlink -s 2>/dev/null | grep -qi "Link 0"; then LINK=nvlink; else LINK=pcie; fi

mkdir -p "$(dirname "$OUT")"
if [ -n "$GPU_ID" ]; then
  GPU_ID_JSON="\"$GPU_ID\""
  CUSTOM=null
else
  GPU_ID_JSON=null
  CUSTOM=$(printf '{"name":"%s","vram_gb":%s,"bandwidth":0,"fp16":0,"fp8":0,"cc":%s,"nvlink":0,"vendor":"nvidia","arch":"detected"}' \
    "$NAME" "$VRAM_GIB" "$CC")
fi

cat > "$OUT" <<JSON
{
  "detected_at": "$(date -Iseconds)",
  "rigs": [
    {
      "id": "host-detected",
      "name": "This host - ${COUNT}x ${NAME}",
      "gpu_id": ${GPU_ID_JSON},
      "custom_gpu": ${CUSTOM},
      "gpus_per_node": ${COUNT},
      "nodes": 1,
      "host_ram_gb": ${RAM_GB},
      "os": "linux",
      "interconnect": "${LINK}",
      "driver_version": "${DRV}",
      "vram_override_gb": ${VRAM_GIB},
      "notes": "Auto-detected by detect-hardware.sh"
    }
  ]
}
JSON
echo "Wrote $OUT: ${COUNT}x ${NAME} (${VRAM_GIB} GiB each, sm_${CC//./}, driver ${DRV}, ${RAM_GB} GB RAM)"
