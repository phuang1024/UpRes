import torch


SCALE_FAC = 4
# Output
OUT_SIZE = (540, 960)
# Input
IN_SIZE = (OUT_SIZE[0]//SCALE_FAC, OUT_SIZE[1]//SCALE_FAC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
