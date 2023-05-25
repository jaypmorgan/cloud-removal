from src.model import CloudRemover

# create a model
model = CloudRemover()

# create a model using the existing weights
model = CloudRemover(pretrained=True)

# create a model using a different wavelength
model = CloudRemover(wavelength="H-alpha", pretrained=True)
