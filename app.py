from flask import Flask, request, jsonify
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


def call_nvidia(api_key, model, messages, temperature, max_tokens, timeout):
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
            timeout=(5, 15)  # 🔥 БЫЛО ДОЛГО → теперь максимум 15 сек
        )

        if resp.status_code != 200:
            return None, f"{model} HTTP {resp.status_code}"

        data = resp.json()

        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not text:
            return None, "empty response"

        return text, None

    except requests.exceptions.Timeout:
        return None, f"{model} TIMEOUT"

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
            max_tokens = 300
        else:
            max_tokens = min(int(incoming_tokens), 700)

        leased_key = choose_key_for_client(client_id)
        if not leased_key:
            return jsonify({
                "error": {
                    "message": "No NVIDIA API keys configured"
                }
            }), 500

        final_text = None
        chosen_model = None

        # Главная модель
        text, err = call_nvidia(
            api_key=leased_key,
            model=MAIN_MODEL,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=15
        )

        if text:
            final_text = text
            chosen_model = MAIN_MODEL
            print("KEY CHOSEN:", leased_key[:8] + "...")
            print("MODEL CHOSEN:", chosen_model)
        else:
            print(err)

            # Fallback
            text, err = call_nvidia(
                api_key=leased_key,
                model=FAST_MODEL,
                messages=full_messages,
                temperature=temperature,
                max_tokens=min(max_tokens, 400),
                timeout=10
            )

            if text:
                final_text = text
                chosen_model = FAST_MODEL
                print("KEY CHOSEN:", leased_key[:8] + "...")
                print("MODEL CHOSEN:", chosen_model)
            else:
                print(err)

        if not final_text:
            final_text = "*thinking...*"
            chosen_model = "hybrid-ai"

        # память
        if incoming_messages:
            memory.append(incoming_messages[-1])
        memory.append({"role": "assistant", "content": final_text})
        memory[:] = memory[-MAX_MEMORY_MESSAGES:]
        save_memory()

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": chosen_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": {
                "message": f"Server error: {str(e)}"
            }
        }), 500

    finally:
        if leased_key:
            release_key(client_id, leased_key)


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
