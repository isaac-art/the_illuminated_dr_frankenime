import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2SeqBiLSTM(nn.Module):
    def __init__(self, vocab_size:int, embed_size:int, hidden_size:int, dropout:float=0.5, bidirectional:bool=True):
        super(Seq2SeqBiLSTM, self).__init__()
        self.bidirectional=bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        lstm_hidden_size = 2 * hidden_size if self.bidirectional else hidden_size

        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=self.bidirectional, batch_first=True)
        self.encoder_layers = self._build_layers(hidden_size, lstm_hidden_size, dropout)

        self.decoder = nn.LSTM(embed_size, lstm_hidden_size, batch_first=True)
        self.decoder_layers = self._build_layers(hidden_size, lstm_hidden_size, dropout)

        self.fc = nn.Linear(lstm_hidden_size, vocab_size)
        self.relu = nn.ReLU()

        self.softmax = nn.LogSoftmax(dim=2)

    def _build_layers(self, hidden_size: int, lstm_hidden_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size),
            nn.Dropout(dropout),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, target: torch.Tensor = None, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size, seq_len = x.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(x.device)

        x_embed = self.embedding(x)
        _, (hn, _) = self.encoder(x_embed)
        hn = self._handle_bidirectional(hn)

        dec_input = x[:, 0].unsqueeze(1)
        hidden_state = (hn, torch.zeros_like(hn))
        
        for t in range(1, seq_len):
            dec_input, hidden_state = self._decode_step(dec_input, hidden_state, outputs, t)
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing and target is not None:
                dec_input = target[:, t].unsqueeze(1)
            else:
                dec_input = outputs[:, t].argmax(dim=1).unsqueeze(1)
        return outputs
    
    def _handle_bidirectional(self, hn: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            hn = torch.cat((hn[0:1], hn[1:2]), dim=2)
        return self.encoder_layers(hn.squeeze(0)).unsqueeze(0)

    def _decode_step(self, dec_input: torch.Tensor, hidden_state: torch.Tensor, outputs: torch.Tensor, t: int) -> tuple:
        dec_embed = self.embedding(dec_input)
        output, hidden_state = self.decoder(dec_embed, hidden_state)
        output = self.decoder_layers(output.squeeze(1)).unsqueeze(1)
        output = self.fc(self.relu(output))
        outputs[:, t] = output.squeeze(1)
        return dec_input, hidden_state