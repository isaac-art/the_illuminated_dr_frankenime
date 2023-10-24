import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim

from utils import p_
from utils.training import make_deterministic
from models.papers.faitas_2019 import ConvincingHarmony

p_()
make_deterministic()


