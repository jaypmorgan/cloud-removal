<p align="center"><img src="./images/overview.png" width="700px"></p>

# Removal of Cloud Shadows from Ground-based Solar Imagary with Deep Learning

Deep Neural Networks for the removal of cloud contamination in
ground-based observations. These codes were presented in the article
'Removing cloud shadows from ground-based solar imagery, Chaoui et
al.'

## Install

You can pip install directly from this github repo:

```bash
pip install git+https://github.com/jaypmorgan/cloud-removal.git
```

or if you've cloned the repo to a local directory:

```bash
cd cloudremoval
pip install ./
```

## Usage

Using the existing synthetic clouds dataset:

```python
import dfp
from cloudremoval.dataset import SyntheticClouds, CloudsTransform

# download the data
dataset = SyntheticClouds(download=True)

# get only a single wavelength from the data
caii = dataset.filter(lambda row: dfp.has_props(row, {"type": "Ca II"}))

# split into train and test
train, test = caii.split(lambda row: dfp.has_props(row, {"subset": "train"}))

# get the first instance:
item = train[0]
inp1 = item.input
item.target
item.mask

# Add a transform
train.transform = CloudsTransform(hflip_p=0.5, vflip_p=0.5)
item = train[0]
inp2 = item.input
```

To create a model, or load one using existing model weights:

```python
from cloudremoval.model import CloudRemover

# create a new model from scratch (i.e. random model weights)
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
```
