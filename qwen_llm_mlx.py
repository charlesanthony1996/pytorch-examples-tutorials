from mlx_lm import load, generate

model_name = "mlx-community/Qwen3.5-4B-4bit"

model, tokenizer = load(model_name)

# print(tokenizer)
# print(model)

messages = [
    {"role": "system", "content": "you are a concise assistant"},
    # {"role": "user", "content": "what is the capital of france"}
    {"role": "user", "content": "who is michael jackson"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generate_prompt=True,
    enable_thinking=False,
    tokenize=False
)

# print(prompt)

answer = generate(model, tokenizer, prompt = prompt, max_tokens=100)

print("Answer: ", answer)