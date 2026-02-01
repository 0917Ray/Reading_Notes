# NanoGPT with no attention implementation
import requests, time, os
import torch
import torch.nn as nn
from torch.nn import functional as F
from utilits import plot_loss, COLORS

# hyperparameters
batch_size = 32
block_size = 32
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embed_dim = 64
dropout = 0.2
# ---------- ----------

torch.manual_seed(1337) # set random seed for reproducibility

# download dataset  (tiny shakespeare dataset)
if '__file__' in globals(): # only works if run as a python script
    base_dir = os.path.dirname(__file__) # for python script
else:
    base_dir = os.getcwd() # for jupyter notebook
input_file_path = os.path.join(base_dir, 'input.txt') # build local path
if not os.path.exists(input_file_path):
    text_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(text_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# build character level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars) # 65 characters in total

# create character to integer mapping and vice versa
cha2ite = { ch:i for i,ch in enumerate(chars) }
ite2cha = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [cha2ite[c] for c in s]        # encoder: take a String, outout the integer of the Characters in the String
decode = lambda l: "".join(ite2cha[i] for i in l) # decoder: take a list of integers, output the corresponding String

# separate train and validation dataset
data = torch.tensor(encode(text), dtype=torch.long) # torch.Size([1115394])
n = int(0.9*len(data)) 
train_data = data[:n] # first 90% for training
val_data = data[n:]   # last 10% for validation

# data loading function
def get_batch(data_source='train'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if data_source == 'train' else val_data
    idx = torch.randint(len(data)-block_size, (batch_size,)) # randomly pick batch_size starting points for the chunks
    x = torch.stack([data[i:i+block_size] for i in idx]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # (batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y

# estimate loss on train and val sets, to avoid noisy estimates
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for data_source in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_source)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[data_source] = losses.mean()
    model.train()
    return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# define the super simple bigram language model
class NanoGPT(nn.Module):
    def __init__(self, embed_dim): # initialize the model
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim=embed_dim) # not consider the absolute position of the token
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.ffwd = FeedForward(embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim) # final layer norm
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, target=None): # index and target are both (batch_size, block_size) == (B, T)
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx) # (batch_size, block_size, embed_size) == (B, T, C)
        position_emb = self.position_embedding(torch.arange(T, device=device)) # (block_size, embed_size) == (T, C)
        x = token_emb + position_emb # (B, T, C), using broadcasting
        x = self.ffwd(x)   # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        # to get logits, we need to project the embeddings to the vocabulary size
        logits = self.lm_head(x)  # (batch_size, block_size (time), volcabulary_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T) # or target = target.reshape(-1)

            loss = F.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, block_size) == (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # (B, T), crop the context to the last block_size tokens
            logits, _ = self(idx_cond) # (B, T, C), get the predictions
            logits = logits[:, -1, :] # (B, C), focus only on the last time step in the chunk
            probs = F.softmax(logits, dim=-1) # (B, C), dim=-1 means along the class dimension
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = NanoGPT(embed_dim)
m = model.to(device)
# print the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f'The model has {num_params:,} parameters')

# define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

loss_list = []
start_time = time.time()
for iter in range(max_iters):
    # every eval_interval steps, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    if iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())
end_time = time.time()
print(f'\nTraining completed in {end_time - start_time:.2f} seconds on {device}')
print(f'Final training loss: {loss.item():.4f}\n')
save_path = os.path.join(base_dir, 'results', 'training_loss_no_attention.pdf')
plot_loss(loss_list, save_path=save_path, figsize=(18, 6), color = COLORS()['green'], show_plot=True)
# save the model

# generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device) # starting
generated_idx = model.generate(context, max_new_tokens=500)
generated_text = decode(generated_idx[0].tolist())
print(generated_text)