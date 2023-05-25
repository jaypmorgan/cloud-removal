import dfp
from src.dataset import SyntheticClouds, CloudsTransform

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
