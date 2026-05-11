# Qwen3 Chat — A Simple Local AI Chat App

A tiny web app that lets you chat with the **Qwen3** AI model running on your own computer. No internet needed after setup. Built with Python + Flask + Hugging Face Transformers.

This guide is written for **beginners**. If you have never built an AI app before, you are in the right place.

---

## What this app does (in plain English)

1. You open a web page in your browser.
2. You type a question into a text box and click **Send**.
3. Your question is sent to a small Python server running on your computer.
4. The server feeds the question to the **Qwen3** AI model.
5. The model generates an answer.
6. The answer appears on the web page.

That's it. It's a mini version of ChatGPT, except the "brain" lives on your own machine.

---

## The parts of the project

The whole app is in one file: [app.py](app.py). There are really only **three big pieces**:

| Piece | What it does | Tool used |
|-------|--------------|-----------|
| The **AI model** | Reads your prompt and writes an answer | `transformers` + `torch` |
| The **web server** | Listens for requests from the browser | `Flask` |
| The **web page** | The box where you type and read answers | HTML + a bit of JavaScript |

---

## How I built it — step by step

### Step 1: Install the tools

```bash
pip install flask transformers torch
```

- **Flask** — turns Python into a tiny website.
- **transformers** — Hugging Face's library for loading AI models.
- **torch** — PyTorch, the math engine the model runs on.

### Step 2: Download the Qwen3 model

The model files live in a folder called `./models/qwen3`. You download it once from Hugging Face and point the code at that folder:

```python
MODEL_PATH = "./models/qwen3"
```

### Step 3: Load the model into memory

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()
```

- The **tokenizer** turns words into numbers the model understands (and back again).
- The **model** is the actual AI brain.
- `device_map="auto"` tells it to use your GPU if you have one, otherwise the CPU.
- `model.eval()` means "we are just using it, not training it."

This happens **once**, when the app starts. That's why the first launch takes a minute.

### Step 4: Start a web server with Flask

```python
app = Flask(__name__)
```

Flask is like a receptionist — it waits for visitors (web requests) and decides what to do with each one.

### Step 5: Make a simple web page

The variable `PAGE` is just a string of HTML. It has:
- A `<textarea>` where you type your prompt.
- A **Send** button.
- A `<pre id=out>` block where the answer is shown.
- A few lines of JavaScript that send your prompt to the server and display the answer.

When you visit `/`, Flask returns this page:

```python
@app.route("/")
def index():
    return render_template_string(PAGE)
```

### Step 6: Handle the chat request

When you click **Send**, the JavaScript sends a `POST` request to `/chat` with your prompt as JSON. Flask catches it here:

```python
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    prompt = (data or {}).get("prompt", "").strip()
```

Then the prompt is wrapped in a chat format the model expects:

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

This adds special markers like "the user said..." / "now the assistant should answer" so the model knows its job.

### Step 7: Generate the answer

```python
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
```

What these settings mean:
- **max_new_tokens=512** — answer can be up to ~512 word-pieces long.
- **do_sample=True** — pick words a bit randomly instead of always the most likely one (makes replies feel more natural).
- **temperature=0.7** — creativity dial. Lower = safer, higher = wilder.
- **top_p=0.9** — only consider the top 90% most likely next words.
- **torch.no_grad()** — "don't track gradients," saves memory since we aren't training.

### Step 8: Decode and send back

```python
new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
return jsonify({"response": response})
```

We slice off the part of the output that corresponds to **only the new answer** (not your original prompt), turn the numbers back into text, and ship it as JSON. The browser then paints it into the page.

### Step 9: Run the server

```python
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
```

This starts Flask on your own machine (`127.0.0.1` = "this computer only") at port `5000`.

---

## How to run it

1. Make sure the model is in `./models/qwen3`.
2. Open a terminal in the project folder.
3. Run:
   ```bash
   python app.py
   ```
4. Wait until you see `Model loaded.`
5. Open your browser and go to **http://127.0.0.1:5000**
6. Type something, click **Send**, and wait for the answer.

---

## The full flow in one picture

```
[ You type a prompt in the browser ]
              |
              v
   JavaScript sends JSON to /chat
              |
              v
       Flask receives the request
              |
              v
  Tokenizer turns text -> numbers
              |
              v
      Qwen3 model generates tokens
              |
              v
  Tokenizer turns numbers -> text
              |
              v
    Flask returns JSON response
              |
              v
  JavaScript shows the answer on page
```

---

## Things you can try changing

- **`temperature`** — try `0.3` for more focused answers, `1.0` for more creative ones.
- **`max_new_tokens`** — raise it for longer replies.
- **`PAGE` HTML** — make the UI prettier with CSS.
- **Chat history** — right now every message is independent. You could keep a list of past messages and send them all each time so the model "remembers."

---

## Common problems

- **"Model loading" takes forever** — normal the first time. It's reading gigabytes from disk.
- **Out of memory** — your GPU is too small. The model will try CPU, which is slower but works.
- **Port 5000 already in use** — change the port in the last line of `app.py`.

---

That's the whole project. One file, three ideas: **load a model, serve a page, answer prompts.** Everything else is details.
