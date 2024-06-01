## 필요한 모듈 설치
! pip install transformers
! pip install datasets
! pip install sentencepiece
! pip install rouge_score
! pip install wandb
! pip install accelerate>=0.21.0

import torch
import numpy as np
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from tabulate import tabulate
import nltk
from datetime import datetime

model_name = "google-t5/t5-base"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set model parameters or use the default
#print(model.config)

# tokenization
encoder_max_length = 128  # considering the distribution of token number
decoder_max_length = 2   # # of token : 2

### 데이터셋 load
data = datasets.load_dataset("commonsense_qa", split="train")

# Take a look at the data
print(data)

# Function to format each entry
def flatten(example):
    return {
        "choice_text": [item['text'] for item in data['choices']],
        "choice_label": [item['label'] for item in data['choices']]
    }
batch_size = 10000
dataset = data.map(flatten, remove_columns=["id", "question_concept", "choices"], batched=True, batch_size=batch_size)

def format_data(entries):
    formatted_entries = []
    for entry in entries:
        question = entry['question']
        choice_labels = entry['choice_label']
        choice_texts = entry['choice_text']
        # Format choices
        formatted_choices = ",  ".join(f"{label}. {text}" for label, text in zip(choice_labels, choice_texts))
        # Format entire entry
        formatted_question = f"'question': '{question}' 'choice': {formatted_choices}"
        formatted_entries.append(formatted_question)
    return formatted_entries

# Call the function and print each formatted entry
formatted_question = format_data(dataset)

dataset = datasets.Dataset.from_dict({"question" : formatted_question, "answerKey" : data["answerKey"]})

train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()

dataset[5]

### 토큰 수 분포 확인
import matplotlib.pyplot as plt

# Initialize list to store token lengths
token_lengths = []

# Iterate through test dataset and calculate token length for each sample
for example in train_data_txt:
    # Tokenize text
    tokens = tokenizer(example["question"], return_tensors="pt")["input_ids"]
    # Get number of tokens
    num_tokens = len(tokens[0])
    # Append token length to list
    token_lengths.append(num_tokens)

# Plot histogram of token lengths
plt.hist(token_lengths, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.show()

### 토큰 수 분포 확인
import matplotlib.pyplot as plt

# Initialize list to store token lengths
token_lengths = []

# Iterate through test dataset and calculate token length for each sample
for example in dataset:
    # Tokenize text
    tokens = tokenizer(example["answerKey"], return_tensors="pt")["input_ids"]
    # Get number of tokens
    num_tokens = len(tokens[0])
    # Append token length to list
    token_lengths.append(num_tokens)

max_token_length = max(token_lengths)
print("가장 긴 토큰 수:", max_token_length)

# Plot histogram of token lengths
plt.hist(token_lengths, bins=20, color='green', edgecolor='black')
plt.title('Distribution of Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.show()

### preprocess 및 tokenize
def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["question"], batch["answerKey"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}

    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

### 학습

nltk.download("punkt", quiet=True)

from datasets import load_metric

# Load accuracy metric from datasets

# accuracy 메트릭 로드
metric = datasets.load_metric("accuracy")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def exact_match_metric(predictions, references):
    exact_matches = []
    for pred, ref in zip(predictions, references):
        if pred == ref:
            exact_matches.append(1)
        else:
            exact_matches.append(0)
    return np.mean(exact_matches) * 100

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute exact match
    exact_match_result = exact_match_metric(decoded_preds, decoded_labels)

    # Compute accuracy
    total_examples = len(decoded_preds)
    correct_predictions = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
    accuracy = correct_predictions / total_examples * 100

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    gen_len = np.mean(prediction_lens)

    result = {
        #"exact_match": round(exact_match_result, 4),
        "accuracy": round(accuracy, 4),
        "gen_len": round(gen_len, 4)
    }
    return result

#from transformers import GradientAccumulationCallback

### Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,  # demo
    per_device_eval_batch_size=8,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    #callbacks=[GradientAccumulationCallback(logging_dir="logs")],
)

import wandb
wandb.login()

  wandb.init(
      # Set the project where this run will be logged
      project="multiple_choice_Answering",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"experiment_2",
      # Track hyperparameters and run metadata
      config={
      "batchsize": 8,
      "encoder_max_length" : 128,
      "dataset": "commonsense_qa",
      "epochs": 10,
      })

### 본격적으로 학습을 시작하기전, evaluation 진행
# Wandb 로깅 콜백 추가

trainer.evaluate()

trainer.train()

eval_metrics = trainer.evaluate()
wandb.log(eval_metrics)

eval_metrics

### 학습한 모델과 학습하지 않은 모델로부터 각각 생성결과를 얻고 비교
def generate_Ans(test_samples, model):
    inputs = tokenizer(
        test_samples["question"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


model_before_tuning = AutoModelForSeq2SeqLM.from_pretrained(model_name)

test_samples = validation_data_txt.select(range(16))

ans_before_tuning = generate_Ans(test_samples, model_before_tuning)[1]
ans_after_tuning = generate_Ans(test_samples, model)[1]

wandb.log({"ans_before_tuning": ans_before_tuning, "ans_after_tuning": ans_after_tuning})

print(
    tabulate(
        zip(
            range(len(ans_after_tuning)),
            ans_after_tuning,
            ans_before_tuning,
        ),
        headers=["Id", "answer after", "answer before"],
    )
)
print("\nTarget answerKey:\n")
print(
    tabulate(list(enumerate(test_samples["answerKey"])), headers=["#Q", "Target answerKey"])
)
print("\nSource questions:\n")
print(tabulate(list(enumerate(test_samples["question"])), headers=["#Q", "question"]))

results = list(zip(range(len(ans_after_tuning)), ans_after_tuning, ans_before_tuning))

# Wandb에 결과 기록
wandb_table = wandb.Table(data=results, columns=["Id", "answer after", "answer before"])
wandb.log({"answers_comparison": wandb_table})
