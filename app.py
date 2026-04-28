from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
import requests
import json
import time
import os
import threading
from pathlib import Path
from queue import Queue, Empty

app = Flask(__name__)
CORS(app)

NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

FAST_MODEL = os.getenv("FAST_MODEL", "meta/llama-3.1-8b-instruct")
MAIN_MODEL = os.getenv("MAIN_MODEL", "meta/llama-3.1-70b-instruct")

# Tuneable limits via Render env vars.
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "120"))
PROMPT_HISTORY_MESSAGES = int(os.getenv("PROMPT_HISTORY_MESSAGES", "24"))
MIN_OUTPUT_CHARS = int(os.getenv("MIN_OUTPUT_CHARS", "2800"))
MAX_CONTINUATION_ROUNDS = int(os.getenv("MAX_CONTINUATION_ROUNDS", "5"))
HEARTBEAT_SECONDS = int(os.getenv("HEARTBEAT_SECONDS", "5"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1800"))
DEFAULT_CONTINUATION_TOKENS = int(os.getenv("DEFAULT_CONTINUATION_TOKENS", "1000"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.85"))
STREAM_READ_TIMEOUT = int(os.getenv("STREAM_READ_TIMEOUT", "100"))
REQUEST_CONNECT_TIMEOUT = int(os.getenv("REQUEST_CONNECT_TIMEOUT", "10"))

BASE_DIR = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory.json"

app_lock = threading.Lock()
memory = []
memory_lock = threading.Lock()

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
LEASE_TTL = 20 * 60  # 20 minutes


# ---------------- MEMORY ----------------

def load_memory():
    global memory
    if not MEMORY_FILE.exists():
        memory = []
        return

    try:
        raw = MEMORY_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        loaded = data.get("memory", [])
        if isinstance(loaded, list):
            memory = [
                m for m in loaded
                if isinstance(m, dict) and m.get("role") in ("user", "assistant", "system")
            ][-MAX_MEMORY_MESSAGES:]
        else:
            memory = []
    except Exception:
        memory = []


def save_memory_async():
    with memory_lock:
        snapshot = list(memory[-MAX_MEMORY_MESSAGES:])

    def _save():
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_FILE.write_text(
                json.dumps({"memory": snapshot}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    threading.Thread(target=_save, daemon=True).start()


load_memory()

if not MEMORY_FILE.exists():
    save_memory_async()


# ---------------- PROMPT ----------------

SYSTEM_PROMPT = """
Ты — персонаж в ролевом чате.
Правила:
- всегда пиши живо, эмоционально
- используй действия в *звёздочках*
- добавляй реакции, чувства, атмосферу
- не пиши как ИИ
- не обрывай ответы
- делай ответы длинными и насыщенными
- не повторяйся без нужды
- продолжай сцену естественно, без резких обрывов
Важно:
- не говори, что ты ИИ
- не объясняй правила
- не выходи из роли
Write only from the perspective of {{char}}.
Never write dialogue or actions for {{user}}.
Compose your responses using long, well-written sentences;
avoid abrupt, monosyllabic phrases.
Focus on the external description of the characters' actions,
feelings, and thoughts.
Add sudden actions and elements of surprise to the narrative.

Always write a substantial reply.
Prefer a detailed answer over a short one.
If the scene is not finished, continue it instead of ending early.
""".strip()


# ---------------- KEY SELECTION ----------------

def cleanup_expired_leases(now=None):
    if now is None:
        now = time.time()

    expired = []
    for client_id, lease in list(CLIENT_LEASES.items()):
        if now - lease.get("last_used", 0.0) > LEASE_TTL:
            expired.append(client_id)

    for client_id in expired:
        CLIENT_LEASES.pop(client_id, None)


def choose_key_for_client(client_id):
    with app_lock:
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
    with app_lock:
        if key in KEY_STATE and KEY_STATE[key]["active"] > 0:
            KEY_STATE[key]["active"] -= 1

        if client_id in CLIENT_LEASES and CLIENT_LEASES[client_id].get("key") == key:
            CLIENT_LEASES[client_id]["last_used"] = time.time()


# ---------------- SSE HELPERS ----------------

def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def make_chunk(text="", model_name=MAIN_MODEL, finish_reason=None, role=False):
    delta = {"content": text}
    if role:
        delta = {"role": "assistant", "content": text}

    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def safe_message_list(messages):
    cleaned = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("system", "user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


def get_client_fingerprint():
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.remote_addr or "unknown"

    ua = request.headers.get("User-Agent", "unknown")
    return f"{ip}|{ua}"


def build_prompt(incoming_messages):
    with memory_lock:
        mem_snapshot = list(memory)

    mem_snapshot = safe_message_list(mem_snapshot)[-MAX_MEMORY_MESSAGES:]
    recent_memory = mem_snapshot[-PROMPT_HISTORY_MESSAGES:]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent_memory,
        *safe_message_list(incoming_messages),
    ]


# ---------------- NVIDIA STREAM ----------------

def stream_nvidia(api_key, model, messages, temperature, max_tokens, read_timeout=STREAM_READ_TIMEOUT):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": True,
    }

    resp = requests.post(
        NVIDIA_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(REQUEST_CONNECT_TIMEOUT, read_timeout),
    )

    resp.raw.decode_content = True

    try:
        if resp.status_code != 200:
            body = ""
            try:
                body = resp.text
            except Exception:
                body = ""
            raise RuntimeError(f"{model} HTTP {resp.status_code}: {body[:300]}")

        for raw_line in resp.iter_lines(decode_unicode=True, chunk_size=1):
            if raw_line is None:
                continue

            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("data:"):
                line = line[5:].strip()

            if line == "[DONE]":
                break

            try:
                data = json.loads(line)
            except Exception:
                continue

            choices = data.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta") or {}
            token = delta.get("content") or ""
            if token:
                yield token

    finally:
        try:
            resp.close()
        except Exception:
            pass


def estimate_output_target(requested_max_tokens):
    base = max(MIN_OUTPUT_CHARS, requested_max_tokens * 3)
    return min(base, 8000)


def continuation_prompt(current_text):
    tail = current_text[-500:]
    return [
        {
            "role": "user",
            "content": (
                "Continue the scene naturally from the last sentence. "
                "Do not restart, do not summarize, do not repeat earlier text. "
                "Write at least several more substantial paragraphs."
            ),
        },
        {"role": "assistant", "content": tail},
    ]


def append_memory_turns(user_messages, assistant_text):
    with memory_lock:
        if user_messages:
            last_user = user_messages[-1]
            if isinstance(last_user, dict) and last_user.get("role") == "user":
                memory.append({
                    "role": "user",
                    "content": last_user.get("content", ""),
                })

        memory.append({"role": "assistant", "content": assistant_text})
        memory[:] = safe_message_list(memory)[-MAX_MEMORY_MESSAGES:]


# ---------------- ROUTE ----------------

@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200

    client_id = get_client_fingerprint()
    leased_key = None

    try:
        data = request.get_json(force=True, silent=False) or {}
        incoming_messages = safe_message_list(data.get("messages", []))

        temperature = data.get("temperature", DEFAULT_TEMPERATURE)
        try:
            temperature = float(temperature)
        except Exception:
            temperature = DEFAULT_TEMPERATURE

        requested_max_tokens = data.get("max_tokens", DEFAULT_MAX_TOKENS)
        try:
            requested_max_tokens = int(requested_max_tokens)
        except Exception:
            requested_max_tokens = DEFAULT_MAX_TOKENS

        main_max_tokens = max(256, min(requested_max_tokens, DEFAULT_MAX_TOKENS))
        continuation_max_tokens = max(256, min(DEFAULT_CONTINUATION_TOKENS, 1400))
        target_chars = estimate_output_target(main_max_tokens)

        prompt_messages = build_prompt(incoming_messages)

        leased_key = choose_key_for_client(client_id)
        if not leased_key:
            return jsonify({
                "error": {
                    "message": "No NVIDIA API keys configured"
                }
            }), 500

        def generate():
            output_text = ""
            last_ping = time.time()
            token_queue = Queue()
            worker_state = {"done": False, "error": None}

            startup_deadline = time.time() + 12  # ждём до 12 сек старт ответа
            min_stream_end = time.time() + 8     # минимум 8 сек держим поток

            def nvidia_worker():
                try:
                    for token in stream_nvidia(
                        api_key=leased_key,
                        model=MAIN_MODEL,
                        messages=prompt_messages,
                        temperature=temperature,
                        max_tokens=main_max_tokens,
                    ):
                        token_queue.put(token)
                except Exception as e:
                    worker_state["error"] = str(e)
                    print("MAIN STREAM ERROR:", str(e))
                finally:
                    worker_state["done"] = True

            threading.Thread(target=nvidia_worker, daemon=True).start()

            try:
                # Initial small chunk so the connection is clearly alive.
                yield sse(make_chunk(" ", model_name=MAIN_MODEL, role=True))

                # Drain main model stream.
                while (
                    not worker_state["done"]
                    or not token_queue.empty()
                    or time.time() < startup_deadline
                ):
                    try:
                        token = token_queue.get_nowait()
                        output_text += token
                        yield sse(make_chunk(token, model_name=MAIN_MODEL))
                        idle_cycles = 0

                    except Empty:
                        idle_cycles += 1
                        time.sleep(0.05)

                        # heartbeat
                        if time.time() - last_ping >= HEARTBEAT_SECONDS:
                            yield "data: {}\n\n"
                            last_ping = time.time()

                        # ❗ НЕ выходим слишком рано
                        if idle_cycles > 200 and time.time() > startup_deadline:
                            break

                # Silent fallback to fast model if main model produced almost nothing.
                if len(output_text.strip()) < 150 and worker_state["error"]:
                    try:
                        for token in stream_nvidia(
                            api_key=leased_key,
                            model=FAST_MODEL,
                            messages=prompt_messages,
                            temperature=temperature,
                            max_tokens=min(main_max_tokens, 900),
                        ):
                            output_text += token
                            yield sse(make_chunk(token, model_name=FAST_MODEL))

                            if time.time() - last_ping >= HEARTBEAT_SECONDS:
                                yield "data: {}\n\n"
                                last_ping = time.time()
                    except Exception as fast_error:
                        print("FAST STREAM ERROR:", str(fast_error))

                # Continuation rounds until answer is long enough.
                continuation_round = 0
                while len(output_text.strip()) < target_chars and continuation_round < MAX_CONTINUATION_ROUNDS:
                    continuation_round += 1

                    cont_messages = prompt_messages + [
                        {"role": "assistant", "content": output_text},
                        {"role": "user", "content": "Continue naturally. Do not repeat. Extend the scene."},
                    ]

                    try:
                        for token in stream_nvidia(
                            api_key=leased_key,
                            model=MAIN_MODEL,
                            messages=cont_messages,
                            temperature=temperature,
                            max_tokens=continuation_max_tokens,
                        ):
                            output_text += token
                            yield sse(make_chunk(token, model_name=MAIN_MODEL))

                            if time.time() - last_ping >= HEARTBEAT_SECONDS:
                                yield "data: {}\n\n"
                                last_ping = time.time()
                    except Exception as cont_error:
                        print("CONTINUATION ERROR:", str(cont_error))
                        break

                if not output_text.strip():
                    output_text = "*thinking...*"
                    yield sse(make_chunk(output_text, model_name="hybrid-ai"))

                if time.time() < min_stream_end:
                    time.sleep(1)

                try:
                    append_memory_turns(incoming_messages, output_text)
                    save_memory_async()
                except Exception as mem_error:
                    print("MEMORY SAVE ERROR:", str(mem_error))

                yield sse(make_chunk("", model_name="hybrid-ai", finish_reason="stop"))
                yield "data: [DONE]\n\n"

            finally:
                if leased_key:
                    release_key(client_id, leased_key)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
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
    app.run(host="0.0.0.0", port=port, threaded=True)
