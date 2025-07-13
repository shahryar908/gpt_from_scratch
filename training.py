import torch 
import torch.nn as nn
import torch.nn.functional as F
from Utils.dataloader import loader
from Utils.tokenizer import vocab_size
from src.model import Transformer 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    embed_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=4,
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0

    for batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # Forward pass (teacher forcing)
        out = model(x, x)

        # Reshape for loss: (B*T, vocab)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch} | Loss {loss.item():.4f}")

    print(f">>> Epoch {epoch} avg loss: {total_loss / len(loader):.4f}")
     ## if you are facing in the training phase you should need to save the model and then save after every epoch 
     