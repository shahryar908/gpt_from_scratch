import torch 
import torch.nn as nn
import torch.nn.functional as F
from Utils.dataloader import loader
from Utils.tokenizer import vocab_size
from src.model import Transformer 
import os
device = "cuda" if torch.cuda.is_available() else "cpu"


#  Your Transformer model setup
model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    embed_dim=64,      # Reduced from 512
    num_heads=4,       # Reduced from 8
    ff_dim=256,        # Reduced from 2048
    num_layers=4,
    dropout=0.0,       # Match first code
    max_len=32         # Match block_size
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

#  Dataset splitting and loading
n = int(0.9 * len(encoded))
train_data = encoded[:n]
val_data = encoded[n:]
train_dataset = CharDataset(train_data, block_size=32)
val_dataset = CharDataset(val_data, block_size=32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

#  Function to estimate train/val loss
def estimate_loss():
    model.eval()
    losses = {'train': 0.0, 'val': 0.0}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            if i >= 200:
                break
            x, y = x.to(device), y.to(device)
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(device)
            out = model(x, x, tgt_mask=mask)
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
        losses[split] = total_loss / min(i + 1, 200)
    model.train()
    return losses

#  Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f" Saved checkpoint: {path}")

#  Training loop with checkpoint saving
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(device)
        out = model(x, x, tgt_mask=mask)
        loss = criterion(out.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch} | Loss {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f">>> Epoch {epoch} avg loss: {avg_loss:.4f}")
    
    #  Estimate loss
    losses = estimate_loss()
    print(f"Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    #  Save checkpoint after each epoch
    checkpoint_path = f"checkpoints/transformer_epoch_{epoch+1}.pt"
    save_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_path)


model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=2000, block_size=32)
print(decode(generated[0].tolist()))

     ## if you are facing in the training phase you should need to save the model and then save after every epoch 
     
