from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)

model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route('/', methods=['GET', 'POST'])
def home():
    summary = ""  # Default empty summary
    if request.method == "POST":
        input_text = request.form.get("inputtext_", "").strip()

        if input_text:  # Check if input text is not empty
            input_text = "summarize: " + input_text
            try:
                tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True).to(device)
                summary_ids = model.generate(tokenized_text, min_length=30, max_length=300)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(e)
                summary = "An error occurred during summarization. Please try again."

    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
