import urllib.request
import os

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
save_dir = "dataset/shakespeare"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "input.txt")
urllib.request.urlretrieve(url, save_path)

print(f"Saved to {save_path}")
