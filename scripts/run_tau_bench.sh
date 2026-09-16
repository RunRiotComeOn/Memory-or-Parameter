#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 DOMAIN SAVE_NAME [tau2 run arguments ...]" >&2
    exit 2
fi

domain=$1
save_name=$2
shift 2

workspace=/nas04/yixuh/memory
tau_root="$workspace/third_party/tau2-bench"
llm_args='{"temperature":0.0,"max_tokens":1024,"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
agent_llm="${TAU2_AGENT_LLM:-openai/qwen35-tau}"

cd "$tau_root"
export OPENAI_API_KEY=EMPTY
export TAU2_LLM_NL_ASSERTIONS=openai/qwen35-tau
export TAU2_LLM_NL_ASSERTIONS_ARGS='{"temperature":0.0,"max_tokens":2048,"api_base":"http://127.0.0.1:8000/v1","api_key":"EMPTY","extra_body":{"chat_template_kwargs":{"enable_thinking":false}},"response_format":{"type":"json_object"}}'

exec .venv/bin/tau2 run \
    --domain "$domain" \
    --agent-llm "$agent_llm" \
    --agent-llm-args "$llm_args" \
    --user-llm openai/qwen35-tau \
    --user-llm-args "$llm_args" \
    --save-to "$save_name" \
    "$@"
