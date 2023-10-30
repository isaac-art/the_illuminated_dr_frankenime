import torch
import torch.nn as nn

# We use 2 layers of 400 Long Short-Term Memory (LSTM) units each for the RNN. 
# Using a variation of the architecture proposed in [8], we augment the RNN by 
# an external stack structure, which serves as an additional memory unit. The 
# stack stores 32 400-d vectors output from the second layer of the RNN. At each 
# timestep, the RNN reads an element from the stack (with attention over all the 
# stack elements), predicts the next token, and then takes an action (push, pop, 
# no action or a combination of these) to update the content of the stack. The 
# final token predictor consists of a fully connected layer and a softmax layer.

class Stack(nn.Module):
    def __init__(self, stack_dim, stack_size):
        super(Stack, self).__init__()
        self.stack_dim = stack_dim
        self.stack_size = stack_size
        self.stack = torch.zeros(stack_size, stack_dim)
        self.pointer = 0
    
    def push(self, x):
        self.stack[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.stack_size
        return self.stack[self.pointer]
    
    def pop(self):
        self.pointer = (self.pointer - 1) % self.stack_size
        return self.stack[self.pointer]

    def peek(self):
        return self.stack[self.pointer]    


class LSTMWithStack(nn.Module):
    def __init__(self, input_dim=400, n_layers=2, hidden_dim=400, stack_dim=400, stack_size=32, output_dim=135):
        super(LSTMWithStack, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.stack = Stack(stack_dim, stack_size)
        

