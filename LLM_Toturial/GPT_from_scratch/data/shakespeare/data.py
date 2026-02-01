# ------------------------------ required libraries ------------------------------ #
import os
import requests
import torch
# ------------------------------ required libraries ------------------------------ #

# ----------------- download dataset  (tiny shakespeare dataset) ----------------- #
if '__file__' in globals():
    base_dir = os.path.dirname(__file__)
else:
    base_dir = os.getcwd()

input_file_path = os.path.join(base_dir, 'input.txt') # build local path
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
# ----------------- download dataset  (tiny shakespeare dataset) ----------------- #

# ---------------------- separate train & validation dataset --------------------- #
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data) # data length = 1115394
train_data = data[:int(n*0.9)] # first 90% for training
val_data = data[int(n*0.9):]   # last 10% for validation
# ---------------------- separate train & validation dataset --------------------- #

# ----------------------- build character level vocabulary ----------------------- #
chars = list(sorted(set(data)))
volca_size = len(chars) # 65 characters in total
# print(''.join(chars))
# print(f'{len(chars)} unique characters found')
# ----------------------- build character level vocabulary ----------------------- #

# -------------- create character to integer mapping and vice versa -------------- #
cha2ite = { ch:i for i,ch in enumerate(chars) }
ite2cha = { i:ch for i,ch in enumerate(chars) }
# print(cha2ite)
encode = lambda s: [cha2ite[c] for c in s]        # encoder: take a String, outout the integer of the Characters in the String
decode = lambda l: "".join(ite2cha[i] for i in l) # decoder: take a list of integers, output the corresponding String

data = torch.tensor(encode(data), dtype=torch.long)             # torch.Size([1115394])
train_data = torch.tensor(encode(train_data), dtype=torch.long) # torch.Size([1003854])
val_data = torch.tensor(encode(val_data), dtype=torch.long)     #  torch.Size([111540])
# print(data.shape, train_data.shape, val_data.shape)

# test the encoder and decoder
# print(encode("Hello World!"))
# print(decode([20, 43, 50, 50, 53, 1, 35, 53, 56, 50, 42, 2]))
# -------------- create character to integer mapping and vice versa -------------- #

# ---------------------------- time dimension:  chunk ---------------------------- #
block_size = 4
x = data[:block_size]
y = data[1:block_size+1]

# for t in range(block_size):
#     content = x[:t+1]
#     target = y[t]
#     print(f"based on content {content}, want to predict {target}")
# ---------------------------- time dimension:  chunk ---------------------------- #

# ---------------------------- generate mini-batches ---------------------------- #
torch.manual_seed(1337)
block_size = 8
batch_size = 4

def get_batch(data_source='train'):
    # generate a small batch of data of inputs x and targets y
    data = train_data if data_source == 'train' else val_data
    idx = torch.randint(len(data)-block_size, (batch_size,)) # randomly pick batch_size starting points for the chunks
    x = torch.stack([data[i:i+block_size] for i in idx]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in idx]) # (batch_size, block_size)
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb)
# print('targets:')
# print(yb)

# for i in range(batch_size):
#     for t in range(block_size):
#         context = xb[i,:t+1]
#         target = yb[i,t]
#         print(f'for batch {i} at time {t}, based on context {context} want to predict {target}')







