from huggingface_hub import login

from huggingface_hub import whoami
login(token="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")
print(whoami())