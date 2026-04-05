import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import random
import torch.nn.functional as F

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

ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
).to(device)
ref_model.eval()

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
lr = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
beta = 0.1  # température DPO
num_iterations = 200
max_new_tokens = 9

# ------------------- prompt -------------------
def generate_prompt():
    start = random.randint(1, 50)
    step = random.randint(1, 3)
    numbers = [start + i*step for i in range(3)]
    prompt = f"Continue the sequence: {','.join(map(str, numbers))},..."
    return prompt, numbers, step

# ------------------- oracle correct -------------------
def build_correct_sequence(numbers, step, length=5):
    seq = []
    current = numbers[-1]
    for _ in range(length):
        current += step
        seq.append(current)
    return ",".join(map(str, seq))

# ------------------- log prob -------------------
def compute_log_prob(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return selected.sum(dim=1)

# ------------------- boucle DPO -------------------
for it in range(num_iterations):

    prompt, numbers, step = generate_prompt()

    # -------- chosen (correct) --------
    correct = build_correct_sequence(numbers, step)

    # -------- rejected (modèle actuel) --------
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    rejected_text = tokenizer.decode(
        gen[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # nettoyer
    allowed = set("0123456789, ")
    rejected_text = "".join(c for c in rejected_text if c in allowed)

    # fallback si vide
    if len(rejected_text.strip()) == 0:
        rejected_text = "0"

    # -------- tokenizer --------
    chosen_full = prompt + correct
    rejected_full = prompt + rejected_text

    chosen_inputs = tokenizer(chosen_full, return_tensors="pt", padding=True).to(model.device)
    rejected_inputs = tokenizer(rejected_full, return_tensors="pt", padding=True).to(model.device)

    # -------- log probs --------
    logp_chosen = compute_log_prob(model, chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
    logp_rejected = compute_log_prob(model, rejected_inputs['input_ids'], rejected_inputs['attention_mask'])

    with torch.no_grad():
        ref_logp_chosen = compute_log_prob(ref_model, chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
        ref_logp_rejected = compute_log_prob(ref_model, rejected_inputs['input_ids'], rejected_inputs['attention_mask'])

    # -------- DPO loss --------
    pi_logratios = logp_chosen - logp_rejected
    ref_logratios = ref_logp_chosen - ref_logp_rejected

    loss = -torch.log(torch.sigmoid(beta * (pi_logratios - ref_logratios))).mean()

    # -------- update --------
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # -------- logs --------
    if it % 20 == 0:
        print("\nIter:", it)
        print("Prompt:", prompt)
        print("Chosen:", correct)
        print("Rejected:", rejected_text)
        print("Loss:", loss.item())

# ------------------- test -------------------
print("\n------ TEST ------\n")
for _ in range(10):
    prompt, _, _ = generate_prompt()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print(prompt, "→", tokenizer.decode(outputs[0], skip_special_tokens=True))