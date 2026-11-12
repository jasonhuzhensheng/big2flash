#!/bin/zsh
# Every local engine on this Mac, one at a time (each loaded alone), PASSES rounds of all of them, into OUT.
# Keeping the best of several rounds keeps one disturbed run from deciding an engine's number.
#   ENGINES="b2f_mtp mtplx"  only these   PASSES=3   CASES=short,long   RUNS=1   OUT=...   --append
cd "$(dirname "$0")/../.."
OUT=${OUT:-tests/bench/engines2.jsonl}
[ "$1" = "--append" ] || : > $OUT
PY="$HOME/Library/Application Support/big2flash/venv/bin/python"
M="$HOME/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16"   # rapid-mlx 4-bit fp16 + MTP
DD="$HOME/Library/Application Support/big2flash/models/Qwen3.8-27B-MLX-Serve-4bit"                  # mlx-serve's pack (DFlash drafter)
MX="$HOME/Library/Application Support/big2flash/models/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16"
GG="$HOME/Library/Application Support/big2flash/models/gguf/Huihui-Qwen3.8-27B-abliterated-UD-Q4_K_XL.gguf"
LMS=~/.lmstudio/bin/lms
ALL="b2f_mtp b2f_plain mlxlm ollama lms_cpp llamacpp rapid_mtp rapid_plain mlxserve_mtp mlxserve_dflash mtplx"

B() { .venv/bin/python tests/bench/engines.py "$@" --runs ${RUNS:-1} --cases ${CASES:-code,edit} --out $OUT; }
wait_up() { for i in $(seq 600); do curl -sf -m 2 "$1" >/dev/null && return 0; sleep 1; done; echo "never came up: $1"; return 1; }
serve() {  # serve LOG URL CMD... : start a server, wait for URL, leave its pid in P
  local log=$1 url=$2; shift 2
  "$@" > /tmp/bench_${CUR}_$log.log 2>&1 &
  P=$!; wait_up $url
}
stop() { kill $P 2>/dev/null; wait $P 2>/dev/null; sleep 3; }

run_one() {
  case $1 in
  b2f_mtp|b2f_plain)
    local mt=$([ $1 = b2f_mtp ] && echo 1 || echo 0)
    serve b2f http://127.0.0.1:8095/ env PYTHONUNBUFFERED=1 "$PY" -u /Applications/big2flash.app/Contents/Resources/engine/server.py --model "$M" --port 8095 --mtp $mt
    B anthropic http://127.0.0.1:8095 big2flash --label "big2flash 2.1$([ $mt = 1 ] && echo ' (MTP)' || echo ' (MTP off)')"; stop ;;
  mlxlm)
    serve mlxlm http://127.0.0.1:8096/v1/models .venv/bin/python -m mlx_lm server --model "$M" --port 8096
    B openai http://127.0.0.1:8096 "$M" --label "mlx-lm 0.32.2"; stop ;;
  ollama)
    B ollama http://127.0.0.1:11434 qwen3.8:27b-mtp-q4_K_M --label "Ollama 0.40.14"
    ollama stop qwen3.8:27b-mtp-q4_K_M ;;
  lms_cpp)
    $LMS unload --all >/dev/null 2>&1
    $LMS load huihui-qwen3.8-27b-abliterated --identifier q27 -c 16384 --gpu max -y >/dev/null 2>&1
    B openai http://127.0.0.1:1234 q27 --label "LM Studio 0.4.28 (llama.cpp 2.69)"
    $LMS unload --all >/dev/null 2>&1 ;;
  llamacpp)
    serve llamacpp http://127.0.0.1:8104/v1/models llama-server -m "$GG" --port 8104 -ngl 99 -c 16384 --jinja -fa on
    B openai http://127.0.0.1:8104 auto --label "llama.cpp 0.8.0"; stop ;;
  rapid_mtp|rapid_plain)
    local sp=$([ $1 = rapid_mtp ] && echo "" || echo "--no-spec-decode")
    serve rapid http://127.0.0.1:8102/v1/models ~/.local/bin/rapid-mlx serve "$M" --port 8102 --no-thinking --yes ${=sp}
    B openai http://127.0.0.1:8102 auto --label "rapid-mlx 1.0.1$([ $1 = rapid_mtp ] && echo ' (MTP)' || echo ' (no spec)')"; stop ;;
  mlxserve_mtp)
    serve mlxserve http://127.0.0.1:8103/v1/models /Applications/MLX-Serve.app/Contents/MacOS/mlx-serve --model "$M" --host 127.0.0.1 --port 8103
    B openai http://127.0.0.1:8103 auto --label "mlx-serve 26.11.1 (MTP)"; stop ;;
  mlxserve_dflash)
    serve mlxserve http://127.0.0.1:8103/v1/models /Applications/MLX-Serve.app/Contents/MacOS/mlx-serve --model "$DD" --host 127.0.0.1 --port 8103
    B openai http://127.0.0.1:8103 auto --label "mlx-serve 26.11.1 (DFlash, own pack)"; stop ;;
  mtplx)
    serve mtplx http://127.0.0.1:8101/v1/models ~/.local/bin/mtplx serve --model "$MX" --port 8101 --no-auth --yes
    B openai http://127.0.0.1:8101 auto --label "MTPLX 2.12.2 (own pack)"; stop ;;
  esac
}

for pass in $(seq ${PASSES:-3}); do
  for e in ${=ENGINES:-$ALL}; do
    echo "== pass $pass: $e"
    CUR=$e run_one $e
  done
done
echo done
