import torch
import torch.nn as nn
import torch.nn.functional as F

class StackMemory(nn.Module):
    def __init__(self, hidden_dim=400, stack_size=32):
        super(StackMemory, self).__init__()
        self.action_predictor = nn.Linear(hidden_dim, 3)  # Push, Pop, or No-Op
        self.D = nn.Parameter(torch.randn(1, hidden_dim))
        self.stack = None
        self.stack_depth = stack_size
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1) 

    def forward(self, hidden_state):
        device = hidden_state.device
        batch_size, seq_len, _ = hidden_state.size()
        if self.stack is None:
            self.stack = torch.zeros(batch_size, self.stack_depth, self.hidden_dim).to(device)
        for t in range(seq_len):
            # At each timestep, the RNN reads an element from the stack
            ht = hidden_state[:, t, :] 
            # with attention over all the stack elements
            attention_weights = F.softmax(self.attention(self.stack), dim=1)
            context = torch.sum(attention_weights * self.stack, dim=1)
            # predicts the next token
            action_logits = self.action_predictor(ht + context)
            action_prob = F.softmax(action_logits, dim=1)
            action = torch.argmax(action_prob, dim=1)
            # ??????????????
            # and then takes an action (push, pop, no action or a combination of these) to update the content of the stack
            new_stack = self.stack.clone()
            for i, act in enumerate(action): # ??????????????
                if act == 0:  # Push
                    new_stack[i, 1:] = self.stack[i, :-1]
                    new_stack[i, 0] = ht[i]
                elif act == 1:  # Pop # ??????????????
                    new_stack[i, :-1] = self.stack[i, 1:]
                    new_stack[i, -1] = self.D
                else:  # No-Op
                    new_stack[i] = self.stack[i]
            self.stack = new_stack
        # The final token predictor consists of a fully connected layer 
        # print("stack",self.stack.shape) #stack ([32, 32, 400])
        # output = self.fc(self.stack)
        # print("output",output.shape) # ([32, 32, 400]
        return self.stack