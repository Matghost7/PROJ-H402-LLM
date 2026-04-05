import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

# ------------------- SEED -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------- MODEL -------------------
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

# ------------------- OLD POLICY -------------------
old_model = deepcopy(model)
old_model.eval()
for p in old_model.parameters():
    p.requires_grad = False

# ------------------- OPTIM -------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# ------------------- PPO PARAMS -------------------
num_iterations = 200
batch_size = 2
clip_epsilon = 0.2
entropy_coef = 0.01

running_reward = 0

# ------------------- UTILS -------------------
def parse_numbers(text):
    allowed = set("0123456789, ")
    text = "".join(c for c in text if c in allowed)
    nums = []
    for part in text.replace(" ", "").split(","):
        try:
            nums.append(int(part))
        except:
            pass
    return nums


def generate_prompt():
    start = random.randint(1, 20)
    choice = random.choice(["arith", "geom", "fibb"])

    if choice == "arith":
        step = random.randint(1, 5)
        numbers = [start + i * step for i in range(3)]

    elif choice == "geom":
        step = random.randint(2, 4)
        numbers = [start * (step ** i) for i in range(3)]

    else:

        a = 1
        b = random.randint(1, 10)
        numbers = [a, b, a + b]
        step = None

    prompt = f"Continue the {choice} sequence: {','.join(map(str, numbers))},..."
    return prompt, numbers, step, choice


def compute_log_prob(model, input_ids, labels):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    mask = (labels[:, 1:] != -100).float()
    return (selected * mask).sum() / mask.sum()


def compute_reward(pred_numbers, numbers, step, choice):
    if len(pred_numbers) == 0:
        return -2.0

    reward = 0.0

    if choice == "arith":
        prev = numbers[-1]
        for num in pred_numbers:
            if num == prev + step:
                reward += 2.0
            else:
                reward -= 1.0
            prev = num

    elif choice == "geom":
        prev = numbers[-1]
        for num in pred_numbers:
            if prev != 0 and abs(num / prev - step) < 0.1:
                reward += 2.0
            else:
                reward -= 1.0
            prev = num

    else:
        a, b = numbers[-2], numbers[-1]
        for num in pred_numbers:
            if num == a + b:
                reward += 2.0
            else:
                reward -= 1.0
            a, b = b, num

    return reward / len(pred_numbers)


# ------------------- TRAIN -------------------
for it in range(num_iterations):

    total_loss = 0

    for _ in range(batch_size):

        prompt, numbers, step, choice = generate_prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=True,
            temperature=0.3,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].unsqueeze(0)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        pred_numbers = parse_numbers(generated_text)

        reward = compute_reward(pred_numbers, numbers, step, choice)
        reward = torch.tensor(reward, device=device)

        # -------- advantage --------
        advantage = reward - running_reward
        running_reward = 0.9 * running_reward + 0.1 * reward.item()

        # -------- PPO --------
        labels = torch.full_like(inputs['input_ids'], -100)
        labels = torch.cat([labels, generated_ids], dim=1)
        input_ids_full = torch.cat([inputs['input_ids'], generated_ids], dim=1)

        log_prob_new = compute_log_prob(model, input_ids_full, labels)

        with torch.no_grad():
            log_prob_old = compute_log_prob(old_model, input_ids_full, labels)

        ratio = torch.exp(log_prob_new - log_prob_old)

        clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

        ppo_loss = -torch.min(ratio * advantage, clipped * advantage)

        # -------- entropy bonus --------
        logits = model(input_ids_full).logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).mean()

        loss = ppo_loss - entropy_coef * entropy

        total_loss += loss

    total_loss /= batch_size

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # -------- update old model --------
    if it % 50 == 0:
        old_model.load_state_dict(model.state_dict())

    # -------- logging --------
    if it % 10 == 0:
        print(f"\nITER {it}")
        print(f"Loss: {total_loss.item():.4f}")
        print(f"Reward: {reward.item():.4f}")
        print(f"Generated: {generated_text}")

        print(f"Type: {choice}")

        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        print(f"Reward: {reward.item():.4f}")