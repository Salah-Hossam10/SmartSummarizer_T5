# T5-Base Text Summarization

This project implements a text summarization model using the **T5-base** transformer from Hugging Face. The model is trained to generate concise summaries for given input texts using an optimized training pipeline.

## Features
- Uses **T5-base** model for abstractive text summarization.
- Implements **gradient accumulation**, **mixed precision training**, and **gradient checkpointing** for efficient training.
- Supports **batch processing** for better performance.
- Handles **tokenization, padding, and truncation** using Hugging Face's tokenizer.
- Evaluates performance using standard NLP metrics.

## Dataset
The model is trained on **CNN/DailyMail** dataset (or any other dataset containing long-form text and summaries).

## Installation
Ensure you have the required dependencies installed:
```bash
pip install torch transformers datasets accelerate
```

## Training
Run the following script to train the model:
```bash
python train.py
```
Training includes:
- Tokenizing input and output sequences.
- Using **AdamW optimizer** with **learning rate scheduling**.
- Logging losses and evaluation metrics.

## Evaluation
To evaluate the trained model, run:
```bash
python evaluate.py
```
This calculates **ROUGE**, **BLEU**, and **other summarization metrics**.

## Usage
After training, use the model for summarization:
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("./saved_model")

def summarize(text):
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=128)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

## Known Issues & Improvements
- `as_target_tokenizer()` is deprecated; update tokenizer usage.
- Optimize dataset processing using `datasets` library.

## License
This project is licensed under the MIT License.

