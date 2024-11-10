from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2")

if tokenizer_gpt.pad_token is None:
    tokenizer_gpt.add_special_tokens({'pad_token': '[PAD]'})

model_gpt.config.pad_token_id = tokenizer_gpt.pad_token_id

def generate_insights(query):
    inputs = tokenizer_gpt(query, return_tensors='pt', padding=True, truncation=True)

    outputs = model_gpt.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=150,
        pad_token_id=tokenizer_gpt.pad_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    result = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True).strip()
    
    if result:
        result = result.capitalize()
        if not result.endswith('.'):
            result += '.'

    return result
