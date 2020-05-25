from torch.utils.data import Dataset

class BaseDataset(Dataset):
	def __init__(self, opt):
		super().__init__()

	def __getitem__(self, index):
		pass

	def __len__(self): pass
