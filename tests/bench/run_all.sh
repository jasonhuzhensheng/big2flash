#!/bin/zsh
# Runs engines.py against every local engine, one at a time (each loaded alone), into engines.jsonl.
cd "$(dirname "$0")/../.."
OUT=tests/bench/engines.jsonl
[ "$1" = "--append" ] || : > $OUT
PY="$HOME/Library/Application Support/big2flash/venv/bin/python"
M="$HOME/Library/Application Support/big2flash/models/Qwen3.8-27B-4bit-MTP-fp16"
B() { .venv/bin/python tests/bench/engines.py "$@"; }
wait_up() { for i in $(seq 300); do curl -s -m 2 "$1" >/dev/null && return 0; sleep 1; done; return 1; }

for mt in ${=MTPS:-1 0}; do  # big2flash, MTP on / off
  PYTHONUNBUFFERED=1 "$PY" -u /Applications/big2flash.app/Contents/Resources/engine/server.py --model "$M" --port 8095 --mtp $mt > /tmp/run_all_b2f.log 2>&1 &
  P=$!; wait_up http://127.0.0.1:8095/
  B anthropic http://127.0.0.1:8095 big2flash --label "big2flash$([ $mt = 1 ] && echo ' (MTP)')" --out $OUT
  kill $P; wait $P 2>/dev/null
done

# stock mlx-lm server, same model files
.venv/bin/python -m mlx_lm server --model "$M" --port 8096 > /tmp/run_all_mlxlm.log 2>&1 &
P=$!; wait_up http://127.0.0.1:8096/v1/models
B openai http://127.0.0.1:8096 "$M" --label "mlx-lm 0.32.2" --out $OUT
kill $P; wait $P 2>/dev/null

# Ollama (llama.cpp-based), its Qwen3.8 27B q4_K_M with MTP
B ollama http://127.0.0.1:11434 qwen3.8:27b-mtp-q4_K_M --label "Ollama 0.40.9" --out $OUT
ollama stop qwen3.8:27b-mtp-q4_K_M

# LM Studio llama.cpp 2.68, Qwen3.8 27B GGUF Q4_K_XL; plain, then with its MTP option
LMS=~/.lmstudio/bin/lms
for mtp in no yes; do
  $LMS unload --all >/dev/null 2>&1
  if [ $mtp = yes ]; then $LMS load huihui-qwen3.8-27b-abliterated --identifier q27 -c 16384 --gpu max --speculative-draft-mtp -y || continue
  else $LMS load huihui-qwen3.8-27b-abliterated --identifier q27 -c 16384 --gpu max -y; fi
  B openai http://127.0.0.1:1234 q27 --label "LM Studio llama.cpp$([ $mtp = yes ] && echo ' (MTP)')" --out $OUT
done
$LMS unload --all >/dev/null 2>&1
echo done
