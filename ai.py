import os
import re
import uuid
import time
import sqlite3
import asyncio
import urllib.parse
import torch
import requests
import gradio as gr
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# =========================================================================
# 1. CORE DATABASE & SECURITY SYSTEM (Safe Multithreading & WAL Mode)
# =========================================================================
DB_NAME = "mani_ai_master.db"
MODEL_NAME = "Mani-AI-Ultra-v8-Local"

def get_db_connection():
    """सुरक्षित SQLite कनेक्शन (Database Locking से बचाव)"""
    conn = sqlite3.connect(DB_NAME, timeout=20)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS api_keys (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key_value TEXT UNIQUE NOT NULL,
                        client_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS conversation_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_input TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def generate_mani_api_key(client_name: str) -> str:
    if not client_name.strip():
        client_name = "Default Client"
    conn = get_db_connection()
    cursor = conn.cursor()
    new_key = f"sk-mani-{uuid.uuid4().hex[:16]}"
    cursor.execute("INSERT INTO api_keys (key_value, client_name) VALUES (?, ?)", (new_key, client_name))
    conn.commit()
    conn.close()
    return new_key

def verify_api_key(key: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM api_keys WHERE key_value = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def get_all_keys():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT client_name, key_value, created_at FROM api_keys ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_memory(user_text: str, ai_text: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversation_memory (user_input, ai_response) VALUES (?, ?)", (user_text, ai_text))
    conn.commit()
    conn.close()

def get_recent_memory_turns(limit: int = 2) -> list:
    """चैट हिस्ट्री को सही ChatML फॉर्मेट में वापस लाता है"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_input, ai_response FROM conversation_memory ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    turns = []
    for u, a in reversed(rows):
        turns.append({"role": "user", "content": u})
        turns.append({"role": "assistant", "content": a})
    return turns

init_db()

# =========================================================================
# 2. ADVANCED DATA SCRAPING & IMAGE GENERATION ENGINE
# =========================================================================
def generate_image_url(prompt: str) -> str:
    """FLUX/Pollinations इंजन से तुरंत HD इमेज जनरेट करने का फंक्शन"""
    clean_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip()
    if not clean_prompt:
        clean_prompt = "hd creative digital art"
    encoded_prompt = urllib.parse.quote(clean_prompt)
    seed = uuid.uuid4().int % 1000000
    return f"https://pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&model=flux&seed={seed}&nologo=true"

def analyze_external_data(query: str) -> str:
    urls = re.findall(r'(https?://\S+)', query)
    context = ""
    if urls:
        for url in urls[:1]: 
            try:
                headers = {"User-Agent": "Mani-AI-Data-Analyzer/8.0"}
                response = requests.get(url, headers=headers, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    clean_text = soup.get_text(separator=' ', strip=True)
                    context += f"\n[Extracted Data from {url}]:\n{clean_text[:1200]}\n"
            except Exception: pass
            
    search_keywords = ["weather", "news", "today", "latest", "price", "who is", "what is", "आज", "मौसम", "खबर"]
    if any(kw in query.lower() for kw in search_keywords):
        try:
            results = list(DDGS().text(query, max_results=3))
            if results:
                context += "\n[Live Internet Search Results]: " + " | ".join([r['body'] for r in results if 'body' in r]) + "\n"
        except Exception: pass
    return context

# =========================================================================
# 3. MANI AI - HIGH-INTELLIGENCE NEURAL ENGINE (LOCAL)
# =========================================================================
print("⚡ Booting up Mani AI Ultra Engine (Local Smart Mode)...")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct" 
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,   # CPU पर भी RAM आधी कर देगा, जिससे स्पीड 10x बढ़ेगी
        low_cpu_mem_usage=True,      # लैपटॉप को फ्रीज होने से रोकेगा
        device_map="auto"
    )
    ai_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"✅ Mani AI Neural Core Online! ({device.upper()})")
except Exception as e:
    print(f"⚠️ Core Engine Warning: {e}")
    tokenizer = None
    ai_pipeline = None

MASTER_SYSTEM_PROMPT = f"""You are {MODEL_NAME}, a highly intelligent, logical, and friendly AI created by Mani Singh.
CRITICAL RULES:
1. Always reply in natural Hindi or Hinglish.
2. Be highly logical, direct, and conversational. Give accurate, detailed answers.
3. NEVER repeat yourself. NEVER output weird characters, loops, or broken math. 
4. If you don't know the answer, say "मुझे इसकी पक्की जानकारी नहीं है भाई।" Do not invent facts."""

def run_mani_core(user_query: str) -> str:
    img_triggers = ["image of", "photo of", "picture of", "draw", "फोटो बनाओ", "इमेज बनाओ", "चित्र बनाओ", "तस्वीर बनाओ", "generate image"]
    if any(trig in user_query.lower() for trig in img_triggers):
        img_url = generate_image_url(user_query)
        reply = f"लो भाई, आपके लिए यह फोटो जनरेट कर दी है:\n\n![Generated Image]({img_url})\n\n**डायरेक्ट लिंक:** {img_url}"
        save_memory(user_query, reply)
        return reply

    if not ai_pipeline:
        return "सिस्टम एरर: Mani AI कोर मॉडल लोड नहीं हुआ है। कृपया टर्मिनल चेक करें।"

    clean_q = user_query.strip().lower()
    if clean_q in ["hi", "hello", "hey", "नमस्ते", "हेलो", "हाय", "कैसे हो", "kaise ho", "hi brother", "hi bro"]:
        reply = "नमस्ते भाई! मैं Mani AI हूँ। बताइए, आज मैं आपकी क्या मदद कर सकता हूँ?"
        save_memory(user_query, reply)
        return reply

    external_data = analyze_external_data(user_query)
    messages = [{"role": "system", "content": MASTER_SYSTEM_PROMPT}]
    
    memory_turns = get_recent_memory_turns(limit=2)
    messages.extend(memory_turns)

    full_user_input = f"{external_data}\n{user_query}".strip() if external_data else user_query
    messages.append({"role": "user", "content": full_user_input})
    
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # यहाँ Indentation ठीक कर दी गई है और पैरामीटर्स को स्मार्ट बना दिया है
        outputs = ai_pipeline(
            formatted, 
            max_new_tokens=512,         # 150 से बढ़ाकर 512 कर दिया ताकि जवाब लंबे आएं
            do_sample=False,            # फालतू की प्रोबेबिलिटी कैलकुलेशन बंद (Direct Fast Answer)
            repetition_penalty=1.15,    # अनाप-शनाप लूप्स और रिपीटेशन रोकने के लिए 
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id
        )

        ai_reply = outputs[0]["generated_text"].strip()
        
        ai_reply = re.sub(r'<\|im_end\|>|<\|im_start\|>.*', '', ai_reply).strip()
        
        if not ai_reply:
            ai_reply = "नमस्ते भाई! मैं समझ नहीं पाया, कृपया दोबारा साफ़-साफ़ लिखें।"
            
        save_memory(user_query, ai_reply)
        return ai_reply
    except Exception as e:
        return f"सिस्टम एरर: {str(e)}"

# =========================================================================
# 4. FASTAPI - OPENAI COMPATIBLE REST API PROVIDER
# =========================================================================
app = FastAPI(title="Mani AI Universal Provider")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[ChatMessage]

class ImageGenRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list", 
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "Mani Singh"}]
    }

@app.post("/v1/chat/completions")
async def api_completions(request: Request, body: ChatRequest):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Access Denied: Missing Bearer API Key."})
    
    api_key = auth_header.split("Bearer ")[1].strip()
    if not verify_api_key(api_key):
        return JSONResponse(status_code=403, content={"error": "Access Denied: Invalid Mani AI Key."})
    
    user_msg = body.messages[-1].content if body.messages else ""
    
    ai_reply = await asyncio.to_thread(run_mani_core, user_msg)
    
    p_tokens = len(tokenizer.encode(user_msg)) if tokenizer else len(user_msg)
    c_tokens = len(tokenizer.encode(ai_reply)) if tokenizer else len(ai_reply)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model or MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ai_reply}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens, "total_tokens": p_tokens + c_tokens}
    }

@app.post("/v1/images/generations")
async def api_generate_images(request: Request, body: ImageGenRequest):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Access Denied: Missing Bearer API Key."})
    
    api_key = auth_header.split("Bearer ")[1].strip()
    if not verify_api_key(api_key):
        return JSONResponse(status_code=403, content={"error": "Access Denied: Invalid Mani AI Key."})
    
    img_url = generate_image_url(body.prompt)
    return {
        "created": int(time.time()),
        "data": [{"url": img_url}]
    }

# =========================================================================
# 5. MANI STUDIO UI & FRONTEND CODE GENERATOR
# =========================================================================
def studio_chat(user_input, history):
    text_query = user_input.get("text", "") if isinstance(user_input, dict) else str(user_input)
    files = user_input.get("files", []) if isinstance(user_input, dict) else []
    
    if files:
        text_query += f" [User attached {len(files)} file(s). System log: Vision processing.]"
        
    reply = run_mani_core(text_query)
    out = ""
    words = reply.split(' ')
    for word in words:
        out += word + " "
        yield out.strip()

sample_html_code = """<!-- Mani AI Integration Code for index.html -->
<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <title>Mani AI Client Interface</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: white; padding: 20px; }
        .chat-box { background: #1e293b; padding: 15px; border-radius: 8px; max-width: 600px; margin: auto; }
        input, button { padding: 10px; border-radius: 5px; border: none; margin-top: 10px; }
        input { width: 70%; }
        button { background: #ff3366; color: white; cursor: pointer; }
        #response { margin-top: 15px; white-space: pre-wrap; background: #0f172a; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="chat-box">
        <h2>🚀 Mani AI Live Client</h2>
        <input type="text" id="userQuery" placeholder="Ask anything in Hindi or Hinglish...">
        <button onclick="askManiAI()">Send</button>
        <div id="response">जवाब यहाँ दिखेगा...</div>
    </div>

    <script>
        async function askManiAI() {
            const query = document.getElementById('userQuery').value;
            const resDiv = document.getElementById('response');
            resDiv.innerText = "सोच रहा हूँ...";

            const response = await fetch('http://localhost:8000/v1/chat/completions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer YOUR_GENERATED_API_KEY_HERE'
                },
                body: JSON.stringify({
                    model: 'Mani-AI-Ultra-v8-Local',
                    messages: [{ role: 'user', content: query }]
                })
            });

            const data = await response.json();
            if(data.choices && data.choices[0]) {
                resDiv.innerText = data.choices[0].message.content;
            } else {
                resDiv.innerText = "Error: " + JSON.stringify(data);
            }
        }
    </script>
</body>
</html>
"""

custom_css = """
h1 {text-align: center; color: #ff3366; font-family: 'Courier New', monospace;}
.gradio-container {background-color: #0f172a;}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=custom_css) as studio:
    gr.Markdown(f"# 🚀 {MODEL_NAME} COMMAND CENTER")
    gr.Markdown("**Developed by Mani Singh. Fully Local, Smart & Secure AI Provider.**")
    
    with gr.Tabs():
        with gr.TabItem("💬 Neural Interface (Chat & Image)"):
            gr.ChatInterface(
                fn=studio_chat,
                multimodal=True,
                chatbot=gr.Chatbot(height=550),
                textbox=gr.MultimodalTextbox(placeholder="हिंदी में कुछ भी पूछो, 'फोटो बनाओ', API लिंक दो या फाइल अटैच करो...", file_types=["image", "video", "text"])
            )
            
        with gr.TabItem("🔑 API Key Generator & Credentials"):
            gr.Markdown("### 🌍 यहाँ से मॉडल नाम, यूआरएल और सीक्रेट की जनरेट करके कॉपी करो")
            
            with gr.Row():
                model_box = gr.Textbox(value=MODEL_NAME, label="Model Name (Click to Copy)", interactive=False)
                url_box = gr.Textbox(value="http://localhost:8000/v1/chat/completions", label="Chat API Endpoint", interactive=False)
                img_url_box = gr.Textbox(value="http://localhost:8000/v1/images/generations", label="Image API Endpoint", interactive=False)

            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    client_in = gr.Textbox(label="Enter Client / Website Name (e.g., My Portfolio Website)")
                    gen_btn = gr.Button("Create Secret Key", variant="primary")
                    key_out = gr.Textbox(label="Your Secret API Key (Copy this into index.html)", interactive=False)
                with gr.Column():
                    db_table = gr.Dataframe(headers=["Client Name", "API Key", "Created At"], value=get_all_keys)
                    ref_btn = gr.Button("Refresh Database")
            
            gen_btn.click(fn=generate_mani_api_key, inputs=client_in, outputs=key_out)
            ref_btn.click(fn=get_all_keys, outputs=db_table)

        with gr.TabItem("🌐 Web / Index.html Integration Code"):
            gr.Markdown("### 💻 इसे अपने किसी भी `index.html` या फ्रंटएंड प्रोजेक्ट में पेस्ट करो")
            gr.Code(value=sample_html_code, language="html", interactive=False)

app = gr.mount_gradio_app(app, studio, path="/")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*65)
    print(f"🔥 {MODEL_NAME} IS LIVE AND FULLY ARMED 🔥")
    print("💻 Local Studio UI : http://localhost:8000")
    print("🔌 Chat API URL    : http://localhost:8000/v1/chat/completions")
    print("🎨 Image API URL   : http://localhost:8000/v1/images/generations")
    print("="*65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)