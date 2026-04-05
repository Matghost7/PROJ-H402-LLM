import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

prompt ="Complete the sequence: 1, 2, 3, ...  "



# ======== Hyperparamètres ========
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_iterations = 20  # tu peux augmenter après test
max_new_tokens = 1  # court pour CPU

# ======== Boucle REINFORCE ========

for it in range(num_iterations):
    # inputs pour la génération
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Génération
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].unsqueeze(0)  # [1, gen_len]

    # Reward simple
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    reward = 1.0 if "4" in generated_text else -1
    reward = torch.tensor(reward, device=model.device)

    print(f"\nIteration {it + 1}")
    print("Generated:", generated_text)

    # Préparer labels pour teacher forcing : on met -100 pour les tokens d'entrée
    labels = torch.full_like(inputs['input_ids'], -100)  # ignore input
    labels = torch.cat([labels, generated_ids], dim=1)

    # Préparer input_ids complets (inputs + generated_ids) pour calculer log_probs
    input_ids_full = torch.cat([inputs['input_ids'], generated_ids], dim=1)

    outputs_for_loss = model(input_ids=input_ids_full, labels=labels)
    log_probs = -outputs_for_loss.loss
    #calculate the log of the probability that he would naturally output this considering the context

    # REINFORCE update
    loss = -log_probs * reward
    optimizer.zero_grad()
    loss.backward() #Change theta maximising the reward
    optimizer.step() #modify theta

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
text = tokenizer.decode(outputs, skip_special_tokens=True)
print(text)
