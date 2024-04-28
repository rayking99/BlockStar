import ast
import torch
import torch.nn as nn
from torch.nn import functional as F

# fmt: off
torch.manual_seed(1337)

batch_size = 8
block_size = 64
max_iters = 5000
eval_interval = 200
learning_rate = 0.0001
device = torch.device("mps")
eval_iters = 200
n_embd = 384
dropout = 0.2
n_head = 6
n_layer = 6

replace_dict = {"right": "r", "left": "l", "up": "u", "down": "d"}

def open_text():
    f = open("/Users/jso/code/projects/results.txt", "r").readlines()
    f = list(set(f))
    f = [ast.literal_eval(x) for x in f]
    f_ = []
    for i in f:
        f_.append([x[0] + replace_dict[x[1]] + str(x[2]) for x in i])
    strings = ["|" + " ".join(i) + "." for i in f_]
    data_chars = " ".join(strings)
    move_vocab = list(set([item for sublist in f_ for item in sublist])) + [' ', '|', '.']
    return data_chars, move_vocab


data_chars, moves = open_text()
chars = list(set(data_chars))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


data = torch.tensor(encode(data_chars), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)  
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # [B, T, C]
        q = self.query(x) # [B, T, C]
        
        wei = q @ k.transpose(-2, -1) * C ** (-0.5) # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # [B, T, C]
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class AgentFormer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # [B, T, C]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [T, C]
        x = tok_emb + pos_emb # [B, T, C]
        x = self.blocks(x) # [B, T, C]
        logits = self.lm_head(x) # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


model = AgentFormer()
m = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"iter {iter} train loss: {losses['train']:.4f} eval loss: {losses['eval']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.tensor(encode("|Cr1"), dtype=torch.long, device=device).unsqueeze(0)
moves_out = decode(m.generate(context, max_new_tokens=500)[0].tolist())

print(moves_out)

moves_out_ = moves_out.split('.')[0].replace('|', '').split(' ')
moves_out_not = [x for x in moves_out_ if x not in moves]


print(moves_out_not)


moves_converted = []
for i in moves_out_:
    # char 0 is the letter, char 1 is the direction, and the rest is the number
    try:
        s = [i[0], i[1], i[2:]]
        s[2] = int(s[2])
        s[1] = list(replace_dict.keys())[list(replace_dict.values()).index(s[1])]
    except:
        moves_out_.remove(i)
    # convert back to original
    moves_converted.append(s)

print(moves_converted)
