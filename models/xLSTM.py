import torch
import torch.nn as nn
import torch.nn.functional as F

class xLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        if lstm_type == "slstm":
            from layers.slstm import sLSTM
            self.lstm = sLSTM(input_size, hidden_size, num_layers, dropout)
        elif lstm_type == "mlstm":
            from layers.mlstm import mLSTM
            self.lstm = mLSTM(input_size, hidden_size, num_layers, dropout)
        else:
            raise ValueError(f"Invalid LSTM type: {lstm_type}")
        
        self.norm = nn.LayerNorm(input_size)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout)

        if bidirectional:
            self.proj = nn.Linear(2 * hidden_size, input_size)
        else:
            if lstm_type == "mlstm":
                self.up_proj = nn.Sequential(
                    nn.Linear(input_size, 4 * input_size), 
                    nn.GELU(),
                    nn.Linear(4 * input_size, input_size)
                )
            self.proj = nn.Linear(hidden_size, input_size)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "up_proj"):
            nn.init.xavier_uniform_(self.up_proj[0].weight)
            nn.init.zeros_(self.up_proj[0].bias)
            nn.init.xavier_uniform_(self.up_proj[2].weight)
            nn.init.zeros_(self.up_proj[2].bias)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_seq, hidden_state=None):
        if hasattr(self, "up_proj"):
            input_seq = self.up_proj(input_seq)

        lstm_output, hidden_state = self.lstm(input_seq, hidden_state)
        if self.lstm_type == "slstm":
            hidden_state = [[hidden_state[i][0], hidden_state[i][1]] for i in range(len(hidden_state))]

        if self.bidirectional:
            lstm_output = torch.cat((lstm_output[:, :, :self.hidden_size], lstm_output[:, :, self.hidden_size:]), dim=-1)

        output = self.activation(self.proj(lstm_output))
        output = self.norm(output + input_seq)
        output = self.dropout_layer(output)

        return output, hidden_state
    

class xLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm_type = lstm_type

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([xLSTMBlock(embedding_size if i == 0 else hidden_size,
                                                hidden_size, num_layers, dropout, bidirectional, lstm_type)
                                     for i in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_seq, hidden_states=None):
        embedded_seq = self.embedding(input_seq)

        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_state = block(output_seq, hidden_states[i])
            if self.lstm_type == "slstm":
                hidden_states[i] = [[hidden_state[j][0], hidden_state[j][1]] for j in range(len(hidden_state))]
            else:
                hidden_states[i] = hidden_state

        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states