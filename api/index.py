from flask import Flask, request, jsonify
from model_utils import load_or_train_model, generate_smart_response
from utils import log_event, sanitize_input

app = Flask(__name__)
tokenizer, model = load_or_train_model()

@app.route("/chat", methods=["GET"])
def chat():
    user_input = request.args.get("text", "").strip()
    if not user_input:
        return jsonify({"error": "Parameter 'text' wajib diisi"}), 400

    user_input = sanitize_input(user_input)
    log_event("input", user_input)

    try:
        response = generate_smart_response(tokenizer, model, user_input)
        log_event("response", response)
        return jsonify({"response": response})
    except Exception as e:
        log_event("error", str(e))
        return jsonify({"error": "Terjadi kesalahan pada sistem."}), 500
