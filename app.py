from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import json
import time
import os
import threading
from pathlib import Path

app = Flask(__name__)
CORS(app)

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

FAST_MODEL = "meta/llama-3.1-8b-instruct"
MAIN_MODEL = "meta/llama-3.1-70b-instruct"  # более стабильный вариант

# ---------------- KEY LOADING ----------------

def load_nvidia_keys():
    keys = []
    for env_name in (
        "NVIDIA_API_KEY_1",
        "NVIDIA_API_KEY_2",
        "NVIDIA_API_KEY_3",
        "NVIDIA_API_KEY",
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            value = value.removeprefix("Bearer ").strip()
            if value:
                keys.append(value)

    return list(dict.fromkeys(keys))


NVIDIA_KEYS = load_nvidia_keys()

KEY_STATE = {
    key: {"active": 0, "last_used": 0.0}
    for key in NVIDIA_KEYS
}

CLIENT_LEASES = {}
LEASE_TTL = 20 * 60
LOCK = threading.Lock()

# ---------------- MEMORY ----------------

BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory.json"
MAX_MEMORY = 50
memory = []


def load_memory():
    global memory
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text("utf-8"))
            memory = data.get("memory", [])[-MAX_MEMORY:]
        except:
            memory = []


def save_memory_async():
    def _save():
        try:
            MEMORY_FILE.write_text(
                json.dumps({"memory": memory[-MAX_MEMORY:]}, ensure_ascii=False, indent=2),
                "utf-8"
            )
        except:
            pass

    threading.Thread(target=_save, daemon=True).start()


load_memory()

# ---------------- PROMPT ----------------

SYSTEM_PROMPT = """
Ты — персонаж в ролевом чате.
Пиши эмоционально, живо, с действиями *в звёздочках*.
Не говори что ты ИИ.
Пиши развернуто и непрерывно.
"""

# ---------------- KEY SELECTION ----------------

def choose_key():
    if not NVIDIA_KEYS:
        return None

    return min(
        NVIDIA_KEYS,
        key=lambda k: (KEY_STATE[k]["active"], KEY_STATE[k]["last_used"])
    )


def release_key(key):
    if key in KEY_STATE and KEY_STATE[key]["active"] > 0:
        KEY_STATE[key]["active"] -= 1

# ---------------- SSE HELPERS ----------------

def sse(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk(text, model, finish=None, role=False):
    delta = {"content": text}
    if role:
        delta = {"role": "assistant", "content": text}

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish
        }]
    }

# ---------------- NVIDIA STREAM ----------------

def stream_nvidia(api_key, model, messages, temperature=0.8, max_tokens=600, timeout=60):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": True
    }

    resp = requests.post(URL, headers=headers, json=payload, stream=True, timeout=(10, timeout))

    if resp.status_code != 200:
        raise RuntimeError(f"{model} HTTP {resp.status_code}: {resp.text[:200]}")

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue

        if "data:" in line:
            line = line.split("data:")[1].strip()

        if line == "[DONE]":
            break

        try:
            data = json.loads(line)
            token = (
                data.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
            )
            if token:
                yield token
        except:
            continue

# ---------------- ROUTE ----------------

@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    client = request.headers.get("X-Forwarded-For", request.remote_addr or "x")

    key = choose_key()
    if not key:
        return jsonify({"error": "no keys"}), 500

    with LOCK:
        KEY_STATE[key]["active"] += 1

    data = request.get_json(force=True)
    messages = data.get("messages", [])

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory + messages

    def generate():
        last_ping = time.time()
        output = ""

        try:
            # initial chunk (НЕ пустой)
            yield sse(chunk("...", MAIN_MODEL, role=True))

            # ---------------- MAIN MODEL ----------------
            try:
                for token in stream_nvidia(key, MAIN_MODEL, full_messages):
                    output += token
                    yield sse(chunk(token, MAIN_MODEL))

                    # heartbeat
                    if time.time() - last_ping > 10:
                        yield "data: {}\n\n"
                        last_ping = time.time()

            except Exception as e:
                yield sse(chunk(f"*fallback triggered: {str(e)}*", MAIN_MODEL))

                # ---------------- FALLBACK ----------------
                try:
                    for token in stream_nvidia(key, FAST_MODEL, full_messages):
                        output += token
                        yield sse(chunk(token, FAST_MODEL))
                except Exception as e2:
                    yield sse(chunk(f"*fatal error: {str(e2)}*", FAST_MODEL))

            if not output.strip():
                output = "*no response*"
                yield sse(chunk(output, "hybrid"))

            # memory async save
            if messages:
                memory.append(messages[-1])
            memory.append({"role": "assistant", "content": output})
            memory[:] = memory[-MAX_MEMORY:]
            save_memory_async()

            yield sse(chunk("", MAIN_MODEL, finish="stop"))
            yield "data: [DONE]\n\n"

        finally:
            release_key(key)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

# ---------------- HEALTH ----------------

@app.route("/v1/models")
def models():
    return jsonify({
        "object": "list",
        "data": [{"id": "hybrid-ai", "object": "model"}]
    })


@app.route("/")
def root():
    return "OK"

# ---------------- RUN ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, threaded=True)
