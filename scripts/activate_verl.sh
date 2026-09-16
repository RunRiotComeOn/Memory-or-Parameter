#!/usr/bin/env bash

# Source this file from the repository root:
#   source scripts/activate_verl.sh

_VERL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_VERL_ENV_PREFIX="${_VERL_PROJECT_ROOT}/.verl-venv"
_VERL_CUDA_PREFIX="${_VERL_PROJECT_ROOT}/.cuda-toolkit"

if [[ ! -x "${_VERL_ENV_PREFIX}/bin/python" ]]; then
  echo "VERL environment not found at ${_VERL_ENV_PREFIX}" >&2
  return 1 2>/dev/null || exit 1
fi

export CONDA_PREFIX="${_VERL_ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${_VERL_ENV_PREFIX}"
export CUDA_HOME="${_VERL_CUDA_PREFIX}"
export CUDACXX="${_VERL_CUDA_PREFIX}/bin/nvcc"
export HF_HOME="/nas04/yixuh/hf_cache"
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PATH="${_VERL_ENV_PREFIX}/bin:${_VERL_CUDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${_VERL_CUDA_PREFIX}/targets/x86_64-linux/lib:${_VERL_CUDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

hash -r
unset _VERL_PROJECT_ROOT _VERL_ENV_PREFIX _VERL_CUDA_PREFIX

echo "VERL environment active: ${CONDA_PREFIX}"
python -c 'import torch, transformers, verl, veomni; print(f"torch={torch.__version__} transformers={transformers.__version__} verl={verl.__version__} veomni={veomni.__version__}")'
