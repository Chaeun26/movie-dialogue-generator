from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained("./movie_dialogue_model_final/movie_dialogue_model_final")
tokenizer = AutoTokenizer.from_pretrained("./movie_dialogue_model_final/movie_dialogue_model_final")
model.eval()
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "What’s your plan?")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

@app.route("/uptime", methods=["GET"])
def uptime():
    current_time = time.time()
    uptime_seconds = current_time - start_time
    uptime_minutes = uptime_seconds / 60
    uptime_hours = uptime_minutes / 60
    return jsonify({
        "uptime_seconds": uptime_seconds,
        "uptime_minutes": uptime_minutes,
        "uptime_hours": uptime_hours
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
