import torch 
with open('dataset/shakespeare/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
data=text
num=len(data)*0.7
data=data[:int(num)]
len(data)

chars = sorted(list(set(data)))
vocab_size = len(chars)
print("Unique chars:", vocab_size)

# Map chars to int and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode entire dataset
encoded = torch.tensor(encode(data), dtype=torch.long)
