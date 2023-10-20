import sys

def make_deterministic(seed:int=1337) -> None:
    """
    Make the execution deterministic by setting the seed value for available modules.
    
    Parameters:
    - seed (int): The seed value to set.
    """
    modules_set = []
    if 'torch' in sys.modules:
        torch = sys.modules['torch']
        torch.manual_seed(seed)
        modules_set.append('torch')
    
    if 'numpy' in sys.modules:
        np = sys.modules['numpy']
        np.random.seed(seed)
        modules_set.append('numpy')

    if 'random' in sys.modules:
        random = sys.modules['random']
        random.seed(seed)
        modules_set.append('random')

    if modules_set:
        print(f"Set seed for {', '.join(modules_set)}")
    else:
        print("No known modules loaded.")