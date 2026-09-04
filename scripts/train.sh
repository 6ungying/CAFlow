
#!/usr/bin/env bash

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CAFLOW_PYTHON="${CAFLOW_PYTHON:-python}"
N_GPU="${N_GPU:-8}"

# JPEG restoration
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 0.3 --corrupt jpeg-5
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 0.3 --corrupt jpeg-10

# 4x super-resolution
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 0.3 --corrupt sr4x-pool
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 0.3 --corrupt sr4x-bicubic

# Deblurring
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 1.0 --ot-ode --corrupt blur-uni
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 1.0 --ot-ode --corrupt blur-gauss

# Inpainting
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 1.0 --ot-ode --corrupt inpaint-center
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 1.0 --ot-ode --corrupt inpaint-freeform1020
"$CAFLOW_PYTHON" "$PROJECT_ROOT/train.py" --n-gpu-per-node "$N_GPU" --beta-max 1.0 --ot-ode --corrupt inpaint-freeform2030
