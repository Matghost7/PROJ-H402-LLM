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
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ------------------- hyperparam -------------------
learning_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_iterations = 200
max_new_tokens = 8

running_reward = 0

# ------------------- fonction prompt -------------------
def generate_prompt():
    start = random.randint(1, 50)
    prompt = f"Continue the sequence: {start},{start+1},{start+2},..."
    expected = start + 3
    return prompt, expected

# ------------------- boucle REINFORCE -------------------
for it in range(num_iterations):

    prompt, expected = generate_prompt()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].unsqueeze(0)

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    # ------------------- nettoyer texte -------------------
    allowed = set("0123456789, ")
    generated_text = "".join(c for c in generated_text if c in allowed)

    # ------------------- reward -------------------
    reward = 0.0
    numbers = []

    for part in generated_text.replace(" ", "").split(","):
        try:
            num = int(part)
            numbers.append(num)
            reward += 0.3
        except:
            reward -= 0.3

    if len(numbers) > 0:
        if numbers[0] == expected:
            reward += 2.0
        else:
            reward -= 1.0
    else:
        reward -= 3.0

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            reward += 1.0
        else:
            reward -= 0.5

    reward = reward / (max_new_tokens/2)
    reward = torch.tanh(torch.tensor(reward, device=model.device))

    # ------------------- baseline RL -------------------
    running_reward = 0.9 * running_reward + 0.1 * reward
    advantage = reward - running_reward

    # ------------------- calcul log prob -------------------
    labels = torch.full_like(inputs['input_ids'], -100)
    labels = torch.cat([labels, generated_ids], dim=1)

    input_ids_full = torch.cat(
        [inputs['input_ids'], generated_ids],
        dim=1
    )

    outputs_for_loss = model(input_ids=input_ids_full)

    logits = outputs_for_loss.logits[:, :-1]
    targets = input_ids_full[:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    selected_log_probs = log_probs.gather(
        -1,
        targets.unsqueeze(-1)
    ).squeeze(-1)

    mask = (labels[:, 1:] != -100).float()

    log_prob = (selected_log_probs * mask).sum()

    loss = -log_prob * advantage

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # ------------------- logs -------------------
    if it % 20 == 0:
        print("\nIter:", it)
        print("Prompt:", prompt)
        print("Generated:", generated_text)
        print("Expected:", expected)
        print("Reward:", reward.item())

# ------------------- test -------------------
print("\n------ TEST ------\n")

for start in [1, 5, 51, 80]:

    prompt = f"Continue the sequence: {start},{start+1},{start+2},..."

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=6,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(prompt, "→", text)