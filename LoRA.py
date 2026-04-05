import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random

# ------------------- modèle -------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

# ------------------- LoRA -------------------
lora_config = LoraConfig(
    r=8,                # rang faible, faible nombre de paramètres
    lora_alpha=16,      # facteur de scaling
    target_modules=["q_proj","v_proj"],  # couches à adapter
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ------------------- prompts -------------------
prompts = [
    "Complete: 1,2,3,...",
    "Continue the sequence: 5,6,7,...",
    "Next number: 10,11,12,...",
    "Finish counting: 20,21,22,..."
]

# ------------------- hyperparam -------------------
learning_rate = 1e-4  # pour LoRA, plus grand que full fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_iterations = 70
max_new_tokens = 5

# ------------------- fonction reward -------------------
def expected_next(prompt):
    try:
        numbers_part = prompt.split(":")[1]
        numbers_clean = numbers_part.replace("...", "").strip()
        numbers = [int(x.strip()) for x in numbers_clean.split(",") if x.strip() != '']
        return str(numbers[-1] + 1)
    except Exception as e:
        print("Error parsing prompt:", prompt)
        raise e

# ------------------- boucle REINFORCE -------------------
for it in range(num_iterations):

    prompt = random.choice(prompts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].unsqueeze(0)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    expected = expected_next(prompt)
    reward = 1.0 if expected in generated_text else -1.0

    """
    generated_ids_seq = generated_ids[0]
    prev_num = None
    for tok_id in generated_ids_seq:
        tok_text = tokenizer.decode(tok_id, skip_special_tokens=True).strip()
        if tok_text == ",":
            continue
        try:
            num = int(tok_text)
        except ValueError:
            prev_num = None
            continue
        if prev_num is not None and num == prev_num + 1:
            reward += 1.0
        prev_num = num
    reward = reward / max_new_tokens
    reward = torch.tensor(reward, device=model.device)
    """

    print(f"\nIter {it+1}")
    print("Prompt:", prompt)
    print("Generated:", generated_text)
    print("Expected:", expected)

    # ---- REINFORCE ----
    labels = torch.full_like(inputs['input_ids'], -100)
    labels = torch.cat([labels, generated_ids], dim=1)

    input_ids_full = torch.cat([inputs['input_ids'], generated_ids], dim=1)

    outputs_for_loss = model(input_ids=input_ids_full)
    logits = outputs_for_loss.logits
    logits = logits[:, :-1]
    targets = input_ids_full[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(
        -1,
        targets.unsqueeze(-1)
    ).squeeze(-1)

    mask = (labels[:, 1:] != -100).float()
    log_prob = (selected_log_probs * mask).sum()
    loss = -log_prob * reward


    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ------------------- test -------------------
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"{prompt} → {text}")
