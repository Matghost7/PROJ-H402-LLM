import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

prompts = [
    "Complete: 1,2,3,...",
    "Continue the sequence: 5,6,7,...",
    "Next number: 10,11,12,...",
    "Finish counting: 20,21,22,..."
]


import random

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_iterations = 30  # tu peux augmenter après test
max_new_tokens = 1  # court pour CPU

def expected_next(prompt):
    try:
        # récupère la partie après ":" si elle existe
        numbers_part = prompt.split(":")[1]
        # supprime les "..." et espaces
        numbers_clean = numbers_part.replace("...", "").strip()
        # split par "," et ignore les chaînes vides
        numbers = [int(x.strip()) for x in numbers_clean.split(",") if x.strip() != '']
        return str(numbers[-1] + 1)
    except Exception as e:
        print("Error parsing prompt:", prompt)
        raise e

for it in range(num_iterations):

    prompt = random.choice(prompts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].unsqueeze(0)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    expected = expected_next(prompt)

    reward = 1.0 if expected in generated_text else -1.0
    reward = torch.tensor(reward, device=model.device)

    print(f"\nIter {it+1}")
    print("Prompt:", prompt)
    print("Generated:", generated_text)
    print("Expected:", expected)

    # ----- REINFORCE -----
    labels = torch.full_like(inputs['input_ids'], -100)
    labels = torch.cat([labels, generated_ids], dim=1)

    input_ids_full = torch.cat([inputs['input_ids'], generated_ids], dim=1)

    outputs_for_loss = model(
        input_ids=input_ids_full,
        labels=labels
    )

    log_probs = -outputs_for_loss.loss
    loss = -log_probs * reward

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()


for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs, skip_special_tokens=True)
    print(text)
