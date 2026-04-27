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
            if value:
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


def choose_key_for_client(client_id):
    with STATE_LOCK:
        cleanup_expired_leases()

        if not NVIDIA_KEYS:
            return None

        sticky = CLIENT_LEASES.get(client_id, {}).get("key")
        ordered = []

        if sticky in NVIDIA_KEYS:
            ordered.append(sticky)

        remaining = [k for k in NVIDIA_KEYS if k != sticky]
        remaining.sort(key=lambda k: (KEY_STATE[k]["active"], KEY_STATE[k]["last_used"]))

        for key in remaining:
            if key not in ordered:
                ordered.append(key)

        for key in ordered:
            if key not in KEY_STATE:
                continue

            KEY_STATE[key]["active"] += 1
            now = time.time()
            KEY_STATE[key]["last_used"] = now
            CLIENT_LEASES[client_id] = {"key": key, "last_used": now}
            return key

        return None


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


def make_chunk(content="", finish_reason=None, role=False, model_name="hybrid-ai"):
    delta = {"content": content}
    if role:
        delta = {"role": "assistant", "content": content}

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }


def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def yield_model_stream(api_key, model, messages, temperature, max_tokens, timeout):
    """
    Streams tokens from NVIDIA directly and yields OpenAI-style SSE chunks.
    Returns the full assembled text as the generator return value.
    """
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

    resp = None
    try:
        resp = requests.post(
            URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(10, timeout)
        )

        if resp.status_code != 200:
            body = ""
            try:
                body = resp.text
            except Exception:
                body = ""
            raise RuntimeError(f"{model} HTTP {resp.status_code}: {body[:300]}")

        assembled = []
        first = True

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    payload_line = line[5:].strip()
                else:
                    payload_line = line

                if payload_line == "[DONE]":
                    break

                try:
                    data = json.loads(payload_line)
                except Exception:
                    continue

                token = extract_text_from_result(data)
                if token:
                    assembled.append(token)

                    chunk = make_chunk(
                        content=token,
                        finish_reason=None,
                        role=first,
                        model_name=model
                    )
                    first = False
                    yield sse(chunk)

        except Exception as e:
            print(f"{model} STREAM ERROR:", str(e))

        return "".join(assembled).strip()

    except Exception as e:
        print(f"{model} ERROR:", str(e))
        raise

    finally:
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


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

        memory = [
            m for m in memory
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        memory = memory[-MAX_MEMORY_MESSAGES:]

        full_messages = memory + incoming_messages
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + full_messages

        temperature = data.get("temperature", 0.8)

        incoming_tokens = data.get("max_tokens", 0)
        if not incoming_tokens or incoming_tokens == 0:
            max_tokens = 500
        else:
            max_tokens = min(int(incoming_tokens), 700)

        leased_key = choose_key_for_client(client_id)
        if not leased_key:
            return jsonify({
                "error": {
                    "message": "No NVIDIA API keys configured"
                }
            }), 500

        def generate():
            global memory

            try:
                # мгновенный стартовый chunk
                yield sse(make_chunk("", None, role=True, model_name="hybrid-ai"))

                final_text = ""

                # 1) DeepSeek first
                try:
                    main_stream = yield_model_stream(
                        api_key=leased_key,
                        model=MAIN_MODEL,
                        messages=full_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=90
                    )
                    final_text = yield from main_stream
                except Exception as e:
                    print("MAIN STREAM FAILED:", str(e))

                # 2) fallback to LLaMA if DeepSeek produced nothing
                if not final_text:
                    try:
                        fast_stream = yield_model_stream(
                            api_key=leased_key,
                            model=FAST_MODEL,
                            messages=full_messages,
                            temperature=temperature,
                            max_tokens=min(max_tokens, 400),
                            timeout=25
                        )
                        final_text = yield from fast_stream
                    except Exception as e:
                        print("FAST STREAM FAILED:", str(e))

                # 3) autocompletion if answer is short
                if final_text and len(final_text) < 400:
                    cont_messages = list(full_messages)
                    cont_messages.append({"role": "assistant", "content": final_text})
                    cont_messages.append({
                        "role": "user",
                        "content": "continue the response, make it longer and more detailed"
                    })

                    try:
                        extra_stream = yield_model_stream(
                            api_key=leased_key,
                            model=FAST_MODEL,
                            messages=cont_messages,
                            temperature=temperature,
                            max_tokens=300,
                            timeout=25
                        )
                        extra_text = yield from extra_stream
                        if extra_text:
                            final_text += ("\n" if final_text else "") + extra_text
                    except Exception as e:
                        print("CONT STREAM FAILED:", str(e))

                if not final_text:
                    final_text = "*thinking...*"
                    yield sse(make_chunk(final_text, None, role=False, model_name="hybrid-ai"))

                # memory
                if incoming_messages:
                    memory.append(incoming_messages[-1])
                memory.append({"role": "assistant", "content": final_text})
                memory[:] = memory[-MAX_MEMORY_MESSAGES:]
                save_memory()

                # final stop chunk
                yield sse(make_chunk("", "stop", role=False, model_name="hybrid-ai"))
                yield "data: [DONE]\n\n"

            finally:
                if leased_key:
                    release_key(client_id, leased_key)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        if leased_key:
            release_key(client_id, leased_key)

        return jsonify({
            "error": {
                "message": f"Server error: {str(e)}"
            }
        }), 500


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
