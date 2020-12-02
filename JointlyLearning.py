import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset
from torchtext.data import BucketIterator, Field


import random, math, os, time

SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)

BATCH_SIZE = 32
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

N_EPOCHS = 50
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'JointlyAlign.pt')


data_path_inp = 'enghin/train.en'
data_path_inp_val = 'enghin/dev.en'
data_path_tar = 'enghin/train.hi'
data_path_tar_val = 'enghin/dev.hi'
data_path_inp_test = 'enghin/test.en'
data_path_tar_test = 'enghin/test.hi'

torch.backends.cudnn.deterministic = True

def tokenize(text):
    return text.split()

src_field = Field(tokenize=tokenize, lower=True, init_token='<SOL>', eos_token='<EOL>')
trg_field = Field(tokenize=tokenize, lower=True, init_token='<SOL>', eos_token='<EOL>')

train_data, valid_data, test_data = TranslationDataset.splits(exts=(".en",".hi"), fields=(src_field, trg_field), path="", train="train_med", validation="dev", test="test")

src_field.build_vocab(train_data, min_freq=2, max_size=10000)
trg_field.build_vocab(train_data, min_freq=2, max_size=10000)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)


device = 'cuda'

class Encoder(nn.Module):
    def __init__(self, inp_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        
        self.inp_dim = inp_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(inp_dim, embed_dim)
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.rnn = nn.GRU(embed_dim, encoder_hidden_dim, bidirectional=True)
        
        self.decoder_hidden_dim = decoder_hidden_dim
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, source):
        embedded = self.dropout(self.embedding(source))
        
        outputs, hidden_st = self.rnn(embedded)
        forw = hidden[-2,:,:]
        back = hidden[-1,:,:]
        hidden_st = torch.tanh(self.fc(torch.cat((forw, back), dim=1)))
        
        return outputs, hidden_st

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.vcoeff = nn.Parameter(torch.rand(decoder_hidden_dim))
        self.attention = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        
    def forward(self, hidden, encoder_outputs):
        shape = encoder_outputs.shape
        srcLen = shape[0]
        batchSize = shape[1]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        hidden = hidden.unsqueeze(1).repeat(1, srcLen, 1)
        
        vcoeff = self.vcoeff.repeat(batchSize, 1).unsqueeze(1)
        
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2))) 
        energy = energy.permute(0, 2, 1)
        
        attention = torch.bmm(vcoeff, energy).squeeze(1)
        
        output = F.softmax(attention, dim=1)

        return output

class Decoder(nn.Module):
    def __init__(self, out_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout, attention):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.embedding = nn.Embedding(out_dim, embed_dim)
        
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embed_dim, decoder_hidden_dim)
        self.out = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim + embed_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        
    def forward(self, input, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        embed = self.dropout(self.embedding(input.unsqueeze(0)))
        a = self.attention(hidden, encoder_outputs)
        
        weight = torch.bmm(a.unsqueeze(1), encoder_outputs).permute(1, 0, 2)

        output, hidden = self.rnn(torch.cat((embed, weight), dim=2), hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        output = self.out(torch.cat((output.squeeze(0), weight.squeeze(0), embed.squeeze(0)), dim=1))
        
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = 'cuda'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        
        outputs = torch.zeros(trg.shape[0], src.shape[1], self.decoder.output_dim).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        
        output = trg[0,:]
        
        for t in range(1, trg.shape[0]):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.max(1)[1]
            isTeacherForce = random.random() < teacher_forcing_ratio

            output = (trg[t] if isTeacherForce else top1)

        return outputs



INPUT_DIM = len(src_field.vocab)
OUTPUT_DIM = len(trg_field.vocab)

attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attention)

model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=trg_field.vocab.stoi['<pad>'])

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg[1:].view(-1)
        optimizer.zero_grad()
        output = model(src, trg)[1:].view(-1, output.shape[-1])
        
        loss = criterion(output, trg).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

def evaluate(model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)
            output = [max(j) for torch.nn.Softmax()(output)]
            if iterator == test_iterator:
              print ("Batch Number: " + str(i))
              for idx in range(batch.batch_size):
                print ("Source Sentence")
                print (" ".join([SRC.vocab.itos[n] for n in src[:,idx]]))
                print ("Actual Translation")
                print (" ".join([TRG.vocab.itos[n] for n in trg[:,idx]]))

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f}')