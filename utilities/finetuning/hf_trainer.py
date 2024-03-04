# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from datasets import load_dataset
# import torch


# # Check GPU availability
# print("Available GPU devices:", torch.cuda.device_count())
# print("Name of the first available GPU:", torch.cuda.get_device_name(0))

# # Load model and tokenizer
# model_name = "TheBloke/Llama-2-7B-chat-GPTQ"

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Move the model to GPU
# model.to('cuda')

# # Load training and validation data
# train_data = load_dataset('json', data_files='train_data.jsonl')
# val_data = load_dataset('json', data_files='val_data.jsonl')

# # Function to format the data
# def formatting_func(example):
#     return tokenizer(example['input'], example.get('output', ''), truncation=True, padding='max_length')

# # Prepare training and validation data
# train_data = train_data.map(formatting_func)
# val_data = val_data.map(formatting_func)

# # Set training arguments
# training_args = TrainingArguments(
#     output_dir="./output",
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=64,
#     save_steps=10_000,
#     save_total_limit=2,
# )

# # Create trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=val_data,
# )

# # Start training
# trainer.train()

# # Save the model
# model.save_pretrained("./output")
