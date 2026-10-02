# Sourced by the grid scripts: selects the task-agent backbone.
#
#   MODEL_PROFILE=qwen35      (default) Qwen3.5-35B-A3B -- every existing result
#   MODEL_PROFILE=gemma4      Gemma 4 26B-A4B-it
#   MODEL_PROFILE=glm47flash  GLM-4.7-Flash
#
# Defines BASE (base checkpoint), SERVED_NAME (what the clients pass as
# --model), PROFILE_TAG, and two helpers that namespace a non-default
# backbone's artifacts, so its grid never "skips" a step because the Qwen
# grid's output for it already exists:
#   backbone_out    textcraft_experiment  -> textcraft_experiment/gemma4
#   backbone_merged tc_x_merged           -> /nas04/yixuh/gemma4_tc_x_merged
# Both are the identity for qwen35, so existing paths are unchanged.
# Profiles live in src/trajectory_memory_lab/model_profiles.py.
MODEL_PROFILE="${MODEL_PROFILE:-qwen35}"
eval "$(PYTHONPATH=src .venv/bin/python -m trajectory_memory_lab.model_profiles shell "$MODEL_PROFILE")" \
  || { echo "ABORT: bad MODEL_PROFILE=$MODEL_PROFILE"; exit 1; }
[[ -f "$BASE/config.json" ]] || { echo "ABORT: $MODEL_PROFILE base checkpoint not downloaded: $BASE"; exit 1; }
backbone_out() { if [[ -n "$PROFILE_TAG" ]]; then echo "$1/$PROFILE_TAG"; else echo "$1"; fi; }
backbone_merged() { echo "/nas04/yixuh/${PROFILE_TAG:+${PROFILE_TAG}_}$1"; }
