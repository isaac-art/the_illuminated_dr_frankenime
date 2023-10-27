import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.5, bidirectional=True):
        super(Seq2SeqBiLSTM, self).__init__()
        self.bidi=bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=self.bidi, batch_first=True)
        self.encoder_bn = nn.BatchNorm1d(hidden_size * 2 if self.bidi else hidden_size)  # Batch Norm for encoder
        self.encoder_drop = nn.Dropout(dropout)
        self.decoder = nn.LSTM(embed_size, 2 * hidden_size if self.bidi else hidden_size, batch_first=True)
        self.decoder_bn = nn.BatchNorm1d(hidden_size * 2 if self.bidi else hidden_size)  # Batch Norm for decoder
        self.decoder_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size if self.bidi else hidden_size, vocab_size)
        self.relu = nn.ReLU() #added
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        x_embed = self.embedding(x)
        _, (hn, _) = self.encoder(x_embed)
        if self.bidi: hn = torch.cat((hn[0:1], hn[1:2]), dim=2) # because bidi we concat

        hn = self.encoder_bn(hn.squeeze(0)).unsqueeze(0)
        hn = self.encoder_drop(hn)
        hn = self.relu(hn)

        batch_size, seq_len = x.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(x.device)
        dec_input = x[:, 0].unsqueeze(1)
        hidden_state = (hn, torch.zeros_like(hn))
        for t in range(1, seq_len):
            dec_embed = self.embedding(dec_input)
            output, hidden_state = self.decoder(dec_embed, hidden_state)
            
            output = self.decoder_bn(output.squeeze(1)).unsqueeze(1)
            output = self.decoder_drop(output)
            output = self.relu(output)

            output = self.fc(output)
            # output = self.softmax(output)
            outputs[:, t] = output.squeeze(1)

            use_teacher_forcing = True if torch.rand(1).item() < teacher_forcing_ratio else False
            if use_teacher_forcing and target is not None: dec_input = target[:, t].unsqueeze(1)
            else: dec_input = output.argmax(dim=2)
        return outputs