 ERROR:", str(e))


load_memory()

# создаём файл сразу, чтобы он точно появился рядом с app.py
if not MEMORY_FILE.exists():
    save_memory()


def cleanup_expired_leases(now=None):
    if now is None:
        now = time.time()

    expired_clients = []
    for client_id, lease in list(CLIENT_LEASES.items()):
        if now - lease["last_used"] > LEASE_TTL_SECONDS:
            expired_clients.append(client_id)

    for client_id in expired_clients:
        del CLIENT_LEASES[client_id]


def get_client_fingerprint():
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.remote_addr or "unknown"

    ua = request.headers.get("User-Agent", "unknown")
    return f"{ip}|{ua}"


def get_key_order_for_client(client_id):
    with STATE_LOCK:
        cleanup_expired_leases()

        if not NVIDIA_KEYS:
            return []

        sticky = CLIENT_LEASES.get(client_id, {}).get("key")
        ordered = []

        if sticky in NVIDIA_KEYS:
            ordered.append(sticky)

        remaining = [k for k in NVIDIA_KEYS if k != sticky]
        free_keys = [k for k in remaining if KEY_STATE[k]["active"] == 0]
        busy_keys = [k for k in remaining if KEY_STATE[k]["active"] > 0]

        if free_keys:
            start = abs(hash(client_id)) % len(free_keys)
            free_keys = free_keys[start:] + free_keys[:start]

        busy_keys.sort(key=lambda k: (KEY_STATE[k]["active"], KEY_STATE[k]["last_used"]))

        for key in free_keys + busy_keys:
            if key not in ordered:
                ordered.append(key)

        return ordered


def acquire_key(client_id, key):
    with STATE_LOCK:
        cleanup_expired_leases()

        if key not in KEY_STATE:
            return None

        KEY_STATE[key]["active"] += 1
        now = time.time()
        KEY_STATE[key]["last_used"] = now
        CLIENT_LEASES[client_id] = {"key": key, "last_used": now}
        return key


def release_key(client_id, key):
    with STATE_LOCK:
        if key in KEY_STATE and KEY_STATE[key]["active"] > 0:
            KEY_STATE[key]["active"] -= 1

        if client_id in CLIENT_LEASES and CLIENT_LEASES[client_id]["key"] == key:
            CLIENT_LEASES[client_id]["last_used"] = time.time()


def extract_text_from_result(result):
    if not isinstance(result, dict):
        return ""

    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        if isinstance(choice, dict):
            return (
                choice.get("delta", {}).get("content", "")
                or choice.get("message", {}).get("content", "")
                or choice.get("text", "")
            )

    if "content" in result:
        return result.get("content", "") or ""

    if "outputs" in result and result["outputs"]:
        try:
            return result["outputs"][0].get("text", "") or ""
        except Exception:
            return ""

    if "error" in result:
        msg = result["error"].get("message", "Unknown")
        return "API ERROR: " + msg

    return ""


def ask_model(api_key, model, messages, temperature, max_tokens, timeout):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens
    }

    try:
        resp = requests.post(
            URL,
            headers=headers,
            json=payload,
            timeout=(10, 25)  # connect=10s, read=25s
        )

        if resp.status_code != 200:
            return None, f"{model} HTTP {resp.status_code}: {resp.text[:300]}"

        try:
            data = resp.json()
        except Exception:
            return None, f"{model} returned non-JSON response"

        text = extract_text_from_result(data).strip()
        if not text:
            text = resp.text.strip()

        if not text:
            return None, f"{model} produced empty response"

        return text, None

    except Exception as e:
        return None, f"{model} ERROR: {str(e)}"


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    global memory

    if request.method == "OPTIONS":
        return "", 200

    client_id = get_client_fingerprint()
    leased_key = None

    try:
        data = request.get_json(force=True)
        incoming_messages = data.get("messages", [])

        memory = [m for m in memory if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        memory = memory[-MAX_MEMORY_MESSAGES:]

        full_messages = memory + incoming_messages
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + full_messages

        temperature = data.get("temperature", 0.8)

        incoming_tokens = data.get("max_tokens", 0)
        if not incoming_tokens or incoming_tokens == 0:
            max_tokens = 700
        else:
            max_tokens = min(int(incoming_tokens), 700)

        key_candidates = get_key_order_for_client(client_id)
        if not key_candidates:
            return jsonify({
                "error": {
                    "message": "No NVIDIA API keys configured"
                }
            }), 500

        final_text = None

        for api_key in key_candidates:
            leased_key = acquire_key(client_id, api_key)
            if not leased_key:
                continue

            try:
                # сначала DeepSeek
                text, err = ask_model(
                    api_key=leased_key,
                    model=MAIN_MODEL,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60
                )

                if text:
                    final_text = text
                    print("KEY CHOSEN:", leased_key[:8] + "...")
                    print("MODEL CHOSEN:", MAIN_MODEL)
                    break

                print(err)

                # fallback LLaMA на том же ключе
                text, err = ask_model(
                    api_key=leased_key,
                    model=FAST_MODEL,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 400),
                    timeout=25
                )

                if text:
                    final_text = text
                    print("KEY CHOSEN:", leased_key[:8] + "...")
                    print("MODEL CHOSEN:", FAST_MODEL)
                    break

                print(err)

            except Exception as e:
                print("REQUEST ERROR:", str(e))

            if final_text is None:
                release_key(client_id, leased_key)
                leased_key = None

        if not final_text:
            final_text = "*thinking...*"

        # обновляем память
        if incoming_messages:
            memory.append(incoming_messages[-1])
        memory.append({"role": "assistant", "content": final_text})
        memory[:] = memory[-MAX_MEMORY_MESSAGES:]
        save_memory()

        # стримим уже готовый текст в Janitor по словам
        def generate():
            try:
                words = final_text.split()
                if not words:
                    yield f"data: {json.dumps({'id': f'chatcmpl-{int(time.time())}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'hybrid-ai', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': final_text}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                first = True
                for word in words:
                    piece = word + " "
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "hybrid-ai",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant" if first else None,
                                    "content": piece
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    first = False
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    time.sleep(0.015)

                yield "data: [DONE]\n\n"

            finally:
                if leased_key:
                    release_key(client_id, leased_key)

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        if leased_key:
            release_key(client_id, leased_key)

        text = f"Server error: {str(e)}"

        def error_generate():
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "hybrid-ai",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(error_generate(), mimetype="text/event-stream")


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "hybrid-ai", "object": "model"}
        ]
    })


@app.route("/", methods=["GET"])
def root():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
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

# 🧠 модели
FAST_MODEL = "meta/llama-3.1-8b-instruct"
MAIN_MODEL = "deepseek-ai/deepseek-v4-flash"

# 🧠 ключи NVIDIA
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
            keys.append(value)

    unique_keys = []
    for key in keys:
        if key not in unique_keys:
            unique_keys.append(key)

    return unique_keys


NVIDIA_KEYS = load_nvidia_keys()

# 🔒 состояние ключей
KEY_STATE = {
    key: {"active": 0, "last_used": 0.0}
    for key in NVIDIA_KEYS
}

CLIENT_LEASES = {}
LEASE_TTL_SECONDS = 20 * 60
STATE_LOCK = threading.Lock()

# 🧠 ПАМЯТЬ
BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory.json"
MAX_MEMORY_MESSAGES = 50
memory = []

# 🎭 SYSTEM PROMPT
SYSTEM_PROMPT = """
Ты — персонаж в ролевом чате.

Правила:
- всегда пиши живо, эмоционально
- используй действия в *звёздочках*
- добавляй реакции, чувства, атмосферу
- не пиши как ИИ
- не обрывай ответы
- делай ответы длинными и насыщенными

Важно:
- не говори что ты ИИ
- не объясняй правила

Write only from the perspective of {{char}}. 
Never write dialogue or actions for {{user}}. 
Compose your responses using long, well-written sentences; 
avoid using abrupt, monosyllabic phrases. 
Focus on the external description of the characters' actions, 
feelings, and thoughts. 
Add sudden actions and elements of surprise to the narrative.
"""


def load_memory():
    global memory
    if MEMORY_FILE.exists():
        try:
            with MEMORY_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = data.get("memory", [])
            memory = [
                m for m in loaded
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            memory = memory[-MAX_MEMORY_MESSAGES:]
        except Exception as e:
            print("MEMORY LOAD ERROR:", str(e))
            memory = []


def save_memory():
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {"memory": memory[-MAX_MEMORY_MESSAGES:]},
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        print("MEMORY SAVE ERROR:", str(e))


load_memory()

# создаём файл сразу, чтобы он точно появился рядом с app.py
if not MEMORY_FILE.exists():
    save_memory()


def cleanup_expired_leases(now=None):
    if now is None:
        now = time.time()

    expired_clients = []
    for client_id, lease in list(CLIENT_LEASES.items()):
        if now - lease["last_used"] > LEASE_TTL_SECONDS:
            expired_clients.append(client_id)

    for client_id in expired_clients:
        del CLIENT_LEASES[client_id]


def get_client_fingerprint():
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.remote_addr or "unknown"

    ua = request.headers.get("User-Agent", "unknown")
    return f"{ip}|{ua}"


def get_key_order_for_client(client_id):
    with STATE_LOCK:
        cleanup_expired_leases()

        if not NVIDIA_KEYS:
            return []

        sticky = CLIENT_LEASES.get(client_id, {}).get("key")
        ordered = []

        if sticky in NVIDIA_KEYS:
            ordered.append(sticky)

        remaining = [k for k in NVIDIA_KEYS if k != sticky]
        free_keys = [k for k in remaining if KEY_STATE[k]["active"] == 0]
        busy_keys = [k for k in remaining if KEY_STATE[k]["active"] > 0]

        if free_keys:
            start = abs(hash(client_id)) % len(free_keys)
            free_keys = free_keys[start:] + free_keys[:start]

        busy_keys.sort(key=lambda k: (KEY_STATE[k]["active"], KEY_STATE[k]["last_used"]))

        for key in free_keys + busy_keys:
            if key not in ordered:
                ordered.append(key)

        return ordered


def acquire_key(client_id, key):
    with STATE_LOCK:
        cleanup_expired_leases()

        if key not in KEY_STATE:
            return None

        KEY_STATE[key]["active"] += 1
        now = time.time()
        KEY_STATE[key]["last_used"] = now
        CLIENT_LEASES[client_id] = {"key": key, "last_used": now}
        return key


def release_key(client_id, key):
    with STATE_LOCK:
        if key in KEY_STATE and KEY_STATE[key]["active"] > 0:
            KEY_STATE[key]["active"] -= 1

        if client_id in CLIENT_LEASES and CLIENT_LEASES[client_id]["key"] == key:
            CLIENT_LEASES[client_id]["last_used"] = time.time()


def extract_text_from_result(result):
    if not isinstance(result, dict):
        return ""

    if "choices" in result and result["choices"]:
        choice = result["choices"][0]
        if isinstance(choice, dict):
            return (
                choice.get("delta", {}).get("content", "")
                or choice.get("message", {}).get("content", "")
                or choice.get("text", "")
            )

    if "content" in result:
        return result.get("content", "") or ""

    if "outputs" in result and result["outputs"]:
        try:
            return result["outputs"][0].get("text", "") or ""
        except Exception:
            return ""

    if "error" in result:
        msg = result["error"].get("message", "Unknown")
        return "API ERROR: " + msg

    return ""


def ask_model(api_key, model, messages, temperature, max_tokens, timeout):
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
        "stream": False
    }

    try:
        resp = requests.post(
            URL,
            headers=headers,
            json=payload,
            timeout=(10, timeout)
        )

        if resp.status_code != 200:
            return None, f"{model} HTTP {resp.status_code}: {resp.text[:300]}"

        try:
            data = resp.json()
        except Exception:
            return None, f"{model} returned non-JSON response"

        text = extract_text_from_result(data).strip()
        if not text:
            text = resp.text.strip()

        if not text:
            return None, f"{model} produced empty response"

        return text, None

    except Exception as e:
        return None, f"{model} ERROR: {str(e)}"


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    global memory

    if request.method == "OPTIONS":
        return "", 200

    client_id = get_client_fingerprint()
    leased_key = None

    try:
        data = request.get_json(force=True)
        incoming_messages = data.get("messages", [])

        memory = [m for m in memory if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        memory = memory[-MAX_MEMORY_MESSAGES:]

        full_messages = memory + incoming_messages
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + full_messages

        temperature = data.get("temperature", 0.8)

        incoming_tokens = data.get("max_tokens", 0)
        if not incoming_tokens or incoming_tokens == 0:
            max_tokens = 700
        else:
            max_tokens = min(int(incoming_tokens), 700)

        key_candidates = get_key_order_for_client(client_id)
        if not key_candidates:
            return jsonify({
                "error": {
                    "message": "No NVIDIA API keys configured"
                }
            }), 500

        final_text = None

        for api_key in key_candidates:
            leased_key = acquire_key(client_id, api_key)
            if not leased_key:
                continue

            try:
                text, err = ask_model(
                    api_key=leased_key,
                    model=MAIN_MODEL,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=20
                )

                if text:
                    final_text = text
                    print("KEY CHOSEN:", leased_key[:8] + "...")
                    print("MODEL CHOSEN:", MAIN_MODEL)
                    break

                print(err)

                text, err = ask_model(
                    api_key=leased_key,
                    model=FAST_MODEL,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 400),
                    timeout=12
                )

                if text:
                    final_text = text
                    print("KEY CHOSEN:", leased_key[:8] + "...")
                    print("MODEL CHOSEN:", FAST_MODEL)
                    break

                print(err)

            except Exception as e:
                print("REQUEST ERROR:", str(e))

            if final_text is None:
                release_key(client_id, leased_key)
                leased_key = None

        if not final_text:
            final_text = "*thinking...*"

        if incoming_messages:
            memory.append(incoming_messages[-1])
        memory.append({"role": "assistant", "content": final_text})
        memory[:] = memory[-MAX_MEMORY_MESSAGES:]
        save_memory()

        def generate():
            try:
                words = final_text.split()
                if not words:
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "hybrid-ai",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": final_text
                                },
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                first = True
                for word in words:
                    piece = word + " "
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "hybrid-ai",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant" if first else None,
                                    "content": piece
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    first = False
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    time.sleep(0.015)

                yield "data: [DONE]\n\n"

            finally:
                if leased_key:
                    release_key(client_id, leased_key)

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
        if leased_key:
            release_key(client_id, leased_key)

        text = f"Server error: {str(e)}"

        def error_generate():
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "hybrid-ai",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": text
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return Response(error_generate(), mimetype="text/event-stream")


@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "hybrid-ai", "object": "model"}
        ]
    })


@app.route("/", methods=["GET"])
def root():
    return "OK"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
