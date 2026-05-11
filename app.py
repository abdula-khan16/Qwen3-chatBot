from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "./models/qwen3"

print("Loading model... (first load can take a minute)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()
print("Model loaded.")

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Qwen3 Chat</title>
<h2>Qwen3 Chat</h2>
<form id=f>
  <textarea name=prompt rows=4 cols=60 placeholder="Ask something..."></textarea><br>
  <button>Send</button>
</form>
<pre id=out></pre>
<script>
f.onsubmit = async (e) => {
  e.preventDefault();
  out.textContent = "Thinking...";
  const r = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({prompt: f.prompt.value})
  });
  const j = await r.json();
  out.textContent = j.response || j.error;
};
</script>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    prompt = (data or {}).get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
