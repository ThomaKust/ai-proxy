from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import requests
import json
import time
from pathlib import Path

app = Flask(__name__)
CORS(app)

# 🔑 NVIDIA KEY
NVIDIA_API_KEY = "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"

URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# 🧠 модели
FAST_MODEL = "meta/llama-3.1-8b-instruct"
MAIN_MODEL = "deepseek-ai/deepseek-v4-flash"

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


# 🔁 запрос к модели
def ask_model(model, messages, temperature, max_tokens, timeout):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
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

    try:
        return requests.post(
            URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(30, timeout)
        )
    except Exception as e:
        print(f"{model} ERROR:", str(e))
        return None


# 🧠 извлечение текста
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


def parse_stream_token(line):
    if not line:
        return ""

    line = line.strip()
    if not line:
        return ""

    if line.startswith("data:"):
        payload = line[5:].strip()
    else:
        payload = line

    if payload == "[DONE]":
        return "__DONE__"

    try:
        obj = json.loads(payload)
    except Exception:
        return ""

    return extract_text_from_result(obj)


def extract_text_from_raw(raw):
    if not raw:
        return ""

    raw = raw.strip()
    if not raw:
        return ""

    try:
        obj = json.loads(raw)
        return extract_text_from_result(obj)
    except Exception:
        pass

    for line in raw.splitlines():
        token = parse_stream_token(line)
        if token and token != "__DONE__":
            return token

    return ""


def open_model_stream(model, messages, temperature, max_tokens, timeout):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
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

    try:
        resp = requests.post(
            URL,
            headers=headers,
            json=payload,
            stream=True,
            timeout=(30, timeout)
        )
    except Exception as e:
        return None, f"{model} ERROR: {str(e)}"

    if resp.status_code != 200:
        body = ""
        try:
            body = resp.text
        except Exception:
            body = ""
        resp.close()
        return None, f"{model} HTTP {resp.status_code}: {body[:300]}"

    iterator = resp.iter_lines(decode_unicode=True)

    first_token = None
    try:
        for line in iterator:
            token = parse_stream_token(line)
            if token == "__DONE__":
                break
            if token:
                first_token = token
                break
    except Exception as e:
        resp.close()
        return None, f"{model} STREAM ERROR: {str(e)}"

    if first_token is None:
        raw = ""
        try:
            raw = resp.text
        except Exception:
            raw = ""
        fallback_text = extract_text_from_raw(raw)
        resp.close()

        if fallback_text:
            def one_shot():
                yield fallback_text
            return one_shot(), None

        return None, f"{model} produced no stream content"

    def token_generator():
        try:
            yield first_token
            for line in iterator:
                token = parse_stream_token(line)
                if token == "__DONE__":
                    break
                if token:
                    yield token
        finally:
            resp.close()

    return token_generator(), None


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
def chat():
    global memory

    if request.method == "OPTIONS":
        return "", 200

    text = ""

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
            max_tokens = 900
        else:
            max_tokens = min(int(incoming_tokens), 900)

        stream_iter = None
        chosen_model = None

        for model, timeout, tokens in (
            (MAIN_MODEL, 90, max_tokens),
            (FAST_MODEL, 25, min(max_tokens, 400)),
        ):
            stream_iter, err = open_model_stream(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=tokens,
                timeout=timeout
            )

            if stream_iter is not None:
                chosen_model = model
                print("MODEL CHOSEN:", chosen_model)
                break
            else:
                print(err)

        if stream_iter is None:
            text = "*thinking...*"

            def fallback_generate():
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

            return Response(fallback_generate(), mimetype="text/event-stream")

        def generate():
            nonlocal text
            assembled = []

            try:
                for piece in stream_iter:
                    if piece:
                        assembled.append(piece)
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
                                        "content": piece
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                final_text = "".join(assembled).strip()
                if not final_text:
                    final_text = "*thinking...*"

                text = final_text

                if incoming_messages:
                    memory.append(incoming_messages[-1])
                memory.append({"role": "assistant", "content": final_text})
                memory = memory[-MAX_MEMORY_MESSAGES:]
                save_memory()

                yield "data: [DONE]\n\n"

            except Exception as e:
                err_text = f"Server error: {str(e)}"
                error_chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "hybrid-ai",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": err_text
                            },
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")

    except Exception as e:
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


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)