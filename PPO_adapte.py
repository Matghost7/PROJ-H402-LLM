import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import torch.nn.functional as F
from copy import deepcopy
import time
import numpy as np
import pickle
import csv
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------- modèle -------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)

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

# ------------------- old policy -------------------
old_model = deepcopy(model)
old_model.eval()

for p in old_model.parameters():
    p.requires_grad = False

# ------------------- hyperparam -------------------
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_iterations = 300
max_new_tokens = 8
running_reward = 0
clip_epsilon = 0.2

# ------------------- prompt -------------------



def generate_prompt():
    start = random.randint(1, 20)
    choice = random.choice(["arithmetic", "geometric", "fibonacci"])

    if choice == "arithmetic":
        step = random.randint(1, 10)
        numbers = [start + i * step for i in range(3)]
        expected_first = numbers[-1] + step

    elif choice == "geometric":
        step = random.randint(2, 4)
        numbers = [start * (step ** i) for i in range(3)]
        expected_first = numbers[-1] * step

    else:

        a = 1
        b = random.randint(1, 10)
        numbers = [a, b, a + b]
        step = None
        expected_first = None



    prompt = f"Continue the {choice} sequence: {','.join(map(str, numbers))},..."
    return prompt, numbers, step, choice, expected_first

# ------------------- log prob -------------------
def compute_log_prob(model, input_ids, prompt_length):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits

    # shift standard
    logits = logits[:, :-1]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)

    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 👉 ne garder que les tokens générés
    selected = selected[:, prompt_length-1:]

    return selected.sum()

# ------------------- stockage métriques -------------------
history = {
    "iter": [],
    "reward": [],
    "reward_avg": [],
    "advantage": [],
    "loss": [],
    "ratio": [],
    "error_tokens": [],
    "time": []
}

# ------------------- boucle PPO -------------------

for it in range(num_iterations):
    start_time = time.time()

    prompt, sequence,  step, choice, expected_first = generate_prompt()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # -------- génération --------
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = outputs[0, inputs["input_ids"].shape[1]:].unsqueeze(0)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # -------- nettoyage --------
    allowed = set("0123456789, ")
    generated_text = "".join(c for c in generated_text if c in allowed)

    if len(generated_text.strip()) == 0:
        generated_text = "0"

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

    # premier nombre
    if len(numbers) > 0:
        if numbers[0] == expected_first:
            reward += 3.0
        else:
            reward -= 1.0
    else:
        reward -= 3.0

    token_errors = 0
    if choice == "arithmetic":
    # progression arithmétique
        prev_num = expected_first - step


        for num in numbers:
            if num == prev_num + step:
                reward += 2.0
            else:
                reward -= 0.5
                token_errors += 1
            prev_num = num

    elif choice == "geometric":
        prev = expected_first/step
        for num in numbers:
            if prev != 0 and abs(num / prev - step) < 0.1:
                reward += 2.0
            else:
                reward -= 0.5
                token_errors += 1
            prev = num

    else:
        a, b = sequence[-2], sequence[-1]
        for num in numbers:
            if num == a + b:
                reward += 2.0
            else:
                reward -= 0.5
                token_errors += 1
            a, b = b, num

    missing = (max_new_tokens // 2) - len(numbers)

    if missing > 0:
        token_errors += missing
        reward -= missing
    reward = reward / (max_new_tokens / 2)
    reward = torch.tensor(reward, device=device)

    # ------------------- baseline -------------------
    running_reward = 0.9 * running_reward + 0.1 * reward
    advantage = reward - running_reward

    # ------------------- inputs full -------------------
    labels = torch.full_like(inputs["input_ids"], -100)
    labels = torch.cat([labels, generated_ids], dim=1)

    input_ids_full = torch.cat([inputs["input_ids"], generated_ids], dim=1)

    prompt_length = inputs["input_ids"].shape[1]

    log_prob_new = compute_log_prob(model, input_ids_full, prompt_length)

    with torch.no_grad():
        log_prob_old = compute_log_prob(old_model, input_ids_full, prompt_length)

    # ------------------- PPO loss -------------------
    ratio = torch.exp(log_prob_new - log_prob_old)

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage

    loss = -torch.min(unclipped, clipped)



    # ------------------- update -------------------
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # -------- update old policy --------

    if it%10 == 0:

        old_model.load_state_dict(model.state_dict())

    # ------------------- logs -------------------
    elapsed_time = time.time() - start_time

    history["iter"].append(it)
    history["reward"].append(reward.item())
    history["reward_avg"].append(running_reward.item())
    history["advantage"].append(advantage.item())
    history["loss"].append(loss.item())
    history["ratio"].append(ratio.item())
    history["error_tokens"].append(token_errors)
    history["time"].append(elapsed_time)

    if it % 20 == 0:
        print(f"\nIter: {it}")
        print("Prompt:", prompt)
        print("Generated:", generated_text)
        print("Reward:", reward.item())
        print("Advantage:", advantage.item())
        print("Ratio:", ratio.item())
        print("Loss:", loss.item())
        print("Token errors:", token_errors)
        print("Time:", round(elapsed_time, 3), "s")

# ------------------- test -------------------
print("\n------ TEST ------\n")

for _ in range(10):
    prompt, sequence,  step, choice, expected_first = generate_prompt()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(prompt, "→", text)

# ------------------- sauvegarde -------------------
os.makedirs("ppo_training_history", exist_ok=True)


# CSV
csv_file = "ppo_training_history/PPO_data.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(history.keys())

    for row in zip(*history.values()):
        writer.writerow(row)

print(f"Historic saved ({csv_file}).")