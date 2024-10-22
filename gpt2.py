from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

tokenizer = AutoTokenizer.from_pretrained("pierreguillou/gpt2-small-portuguese")
model = AutoModelWithLMHead.from_pretrained("pierreguillou/gpt2-small-portuguese")

# Get sequence length max of 1024
tokenizer.model_max_length=1024 

model.eval()  # disable dropout (or leave in train mode to finetune)

# input sequence
text = "Quem era o Pelé? Pelé era um"
inputs = tokenizer(text, return_tensors="pt")

# model output
outputs = model(**inputs, labels=inputs["input_ids"])
loss, logits = outputs[:2]
predicted_index = torch.argmax(logits[0, -1, :]).item()
predicted_text = tokenizer.decode([predicted_index])

# results
print('input text:', text)
print('predicted text:', predicted_text)

# input sequence
text = "Quem era Pelé? Pelé era um"
inputs = tokenizer(text, return_tensors="pt")

# model output using Top-k sampling text generation method
sample_outputs = model.generate(inputs.input_ids,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=True, 
                                max_length=100, # put the token number you want
                                top_k=30,
                                top_p=0.9,
                                temperature=0.6,
                                num_return_sequences=3)

# generated sequence
for i, sample_output in enumerate(sample_outputs):
    print(f">> Generated text {i+1}\n\n{tokenizer.decode(sample_output.tolist(), skip_special_tokens=True)}")

