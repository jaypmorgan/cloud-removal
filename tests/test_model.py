import dfp
from cloudremoval.model import CloudRemover
from cloudremoval.dataset import SyntheticClouds, CloudsTransform

# create a model
model = CloudRemover()

# create a model using the existing weights
model = CloudRemover(pretrained=True)

# create a model using a different wavelength
model = CloudRemover(wavelength="H-alpha", pretrained=True)

# test making of predictions
dataset = SyntheticClouds(download=True, transform=CloudsTransform())
model = CloudRemover(pretrained=True)
out = model(dataset[0].input[None,...])*dataset[0].mask[None,...]

import matplotlib.pyplot as plt
plt.imshow(out[0,0].detach().cpu().numpy(), cmap="Greys_r")
