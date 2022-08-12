from numpy import size
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, device) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads # split to heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divided by heads"

        self.values_net = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.keys_net = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)
        self.queries_net = nn.Linear(self.head_dim, self.head_dim, bias=False).to(device)

        # linear mapping from embed_size -> embed_size before output to next attention block
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to(device) # head_dim * heads IS embed size actually


    def forward(self, q, k, v, mask):
        # this is the most important part !!!
        # the first dim is the batch_size, e.g: # of sentences
        # the second dim is word size in one batch, like 128 words.
        # the third dim is embed size = self.heads_dim * heads  
        N = q.shape[0] #  q,k,v len should be the same
        value_len, key_len, query_len = v.shape[1], k.shape[1], q.shape[1]
        
        # split embeddings into the heads, N and value_len remain the same
        values = v.reshape(N, value_len, self.heads, self.head_dim) 
        keys =  k.reshape(N, key_len, self.heads, self.head_dim)       
        queries = q.reshape(N, query_len, self.heads, self.head_dim)
 
        # go through linear mapping
        values = self.values_net(values)
        keys = self.keys_net(keys)
        queries = self.queries_net(queries)

        # einsum calculates the sum of products based on einstein convention 
        energy =  torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        #  queries shape (N, query_len, heads, heads_dim)
        
        if mask is not None:
            # replace the masked value with very small value to avoid numeric overflow
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        # embed_size = heads * heads_dim, so why not use heads_dim ? --> since they will be concatenate laters
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3) # nhqk / normalized
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]) # nqhd
        out = out.reshape(N, query_len, self.head_dim * self.heads) # concatenation over heads to self.embed_dims
        # attention shape (N, heads, query_len, key_len)
        # value shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, heads_dim)

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, device):
        super().__init__() 
        self.attention = SelfAttention(embed_size, heads, device)
        self.norm1 = nn.LayerNorm(embed_size).to(device)
        self.norm2 = nn.LayerNorm(embed_size).to(device)
        
        # forward_expansion to improve the FFN encoding ability
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        ).to(device)

        self.dropout = nn.Dropout(dropout).to(device) # dropout inside the transformersblock


    def forward(self, value, key, query, mask):
        # sequentially chain the attention block -> add_norm -> FFN -> add_norm
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query)) # why not value ?
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size, 
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.device = device

        # embedding layer will take an integer, make it one-hot, then linear mapping
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size).to(device)
        self.position_embedding = nn.Embedding(max_length, embed_size).to(device)

        self.word_embedding.to(device)
        self.position_embedding.to(device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    device=device
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length


    def forward(self, x, mask):
        N, seq_length = x.shape[0], x.shape[1]
        # arange(0, N) -> [0, 1, 2, ..., N]
        # expand -> enlarge any dim with one length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # the integer input will be converted to one-hot inside the embedding module
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) 
         
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device) -> None:
        super().__init__()
        self.attention = SelfAttention(embed_size, heads, device)
        self.norm = nn.LayerNorm(embed_size).to(device)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion, device
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, trg_mask):
        # the first masked multiheads attention using raw x as the QKV
        attention = self.attention(x, x, x, trg_mask)  
        # take the output sentence's query to attend the input sentences value and keys
        query = self.dropout(self.norm(attention + x))
        # value and key are from encoder, query is generated from self-attention of the decoder last outputs
        # we use every query from decoder to attend keys in encoder output, times values then normalized
        # so every query in the decoder last output will produce a value-shape-like output from transformers
        out = self.transformer_block(value, key, query, src_mask) # shape of (N_dc, q_dc, embed_size)
        return out


class Decoder(nn.Module):
    def __init__(self, 
                    trg_vocab_size, 
                    embed_size,
                    num_layers,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                    max_length, 
    ) -> None:
        super().__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size).to(device)
        self.position_embedding = nn.Embedding(max_length, embed_size).to(device)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device=self.device
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape[0], x.shape[1]
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # enc_out is in shape of (N, l, d) serves as value and keys
            x = layer(x, enc_out, enc_out, src_mask, trg_mask) # values and keys are just features from encoder
            # output will be (N, l, embed_size)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx, 
                 trg_pad_idx, 
                 embed_size=256, 
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0.0,
                 device="cpu",
                 max_length=128,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len) set all pad token to 0 and non-pad token to 1
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):   
        N, trg_len = trg.shape[0], trg.shape[1]
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out 

