"""
Clouds dataset. A dataset of labels comprising the dates to which
cloud coverage is present within the Ca II images. This dataset
assumes that there is only a single sample per day, and this sample
was the one with cloud coverage on.
"""
# internal imports
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional, Union
from collections import namedtuple

# external imports
import dfp
import sunpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tv


Filepath = Union[Path, str]
item = namedtuple("item", ["input", "mask", "target", "uid"])


class SyntheticClouds(Dataset):
    """Dataset that facilitates the training of ML methods to remove clouds.

    This PyTorch dataset consists of cloudy images for Ca II and
    H-alpha wavelengths. Each of these cloudy images are paired with
    an approximate 'un-cloudy' version. These pairs then allows for
    the training of ML algorithms to produce a un-cloudy version given
    the cloudy image.

    Parameters
    ----------
    catalogue : pathlib.Path
        The path to the location of the dataset catalogue (the CSV file).
    train : bool
        Toggle to use the train or test subset of the data. By
        default this is set to True.
    transform : Callable, Optional
        The transformation function to apply to the data before
        passing it to the Neural Network.

    Examples
    --------
    >>> train_data = CloudsDataset()  # training data only
    >>> test_data  = CloudsDataset(train=False)  # testing data only
    >>> train_data[0]  # get the first sample of data.

    If you only want to use a single wavelength (i.e. only Ca II),
    then you can use the filter method.

    >>> train_data.filter(lambda sample: dfp.has_props(sample, {"type": "Ca II"}))
    >>> test_data.filter(lambda sample: dfp.has_props(sample, {"type": "Ca II"}))

    These two subsets of the data could be combined using the addition operator:

    >>> full_data = train_data + test_data

    """
    def __init__(
            self,
            catalogue: Filepath,
            transform: Optional[Callable] = None,
            download: bool = False,
    ):
        catalogue = Path(catalogue)
        if catalogue.is_dir():
            catalogue = catalogue/"synthetic-catalogue.csv"
        self.root = Path(catalogue).parent
        self.download = download
        self.transform = transform
        # load the data, downloading if necessary
        if not Path(catalogue).exists():
            self._download_data()
        self.data: pd.DataFrame = pd.read_csv(catalogue)

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def citation() -> str:
        """Get the article citation.

        Returns
        -------
        str
            Citation string

        """
        return "Morgan, Jay Paul, Paiement, Adeline, & Aboudarham, Jean. (2023). Synthetically generated clouds on ground-based solar observations [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7684200"

    def _download_data(self):
        import urllib.request
        import zipfile
        
        answer = input(f"Data cannot be found at '{self.root}'. Do you want to download it? [y/n] ")
        status_ok = False
        while answer not in ["y", "n"]:
            answer = input(f"Please answer with 'y' or 'n', you entered {answer}")
        if answer == "y":
            def progress(block_num, block_size, total_size):
                if block_num % 10 == 0:
                    down_size = f"{(block_num*block_size)/1000000000:.2f}/{total_size/1000000000:.2f} GB"
                    perc_size = f"{round(100*((block_num*block_size)/total_size), 2):.2f}"
                    print(f"Downloading archive: {down_size} ({perc_size}%)", end="\r", flush=True)
            download_path = "https://zenodo.org/record/7684201/files/synthetic-clouds.zip?download=1"
            urllib.request.urlretrieve(download_path, self.root/"synthetic-clouds.zip", progress)
            with zipfile.ZipFile(self.root/"synthetic-clouds.zip", "r") as f:
                f.extractall(self.root)
            if (self.root/"synthetic-catalogue.csv").exists():
                status_ok = True
        return status_ok

    def __add__(self, other):
        new_cls = deepcopy(self)
        new_cls.data = pd.concat([self.data, other.data])
        return new_cls

    def filter(self, condition: Callable[[dfp.Record], bool]):
        """Filter data for condition

        Filter the data for a condition being true. One can use this
        method to, say, filter for only one particular wavelength. The
        condition you pass should be a callable function with one
        argument (a unary function). This one argument is the single
        sample (i.e. Tuple of tuples). If the result of your passed
        function is True, then the sample will be kept.

        We decided for the condition to be a function as it allows you
        the ultimate flexibility on how you wish to filter the data.

        Parameters
        ----------
        condition : Callable
            The condition that should be met for the data to be used.

        Examples
        --------
        >>> data = CloudsDataset()
        >>> data.filter(lambda s: dfp.has_props(s, {"type": "Ca II"}))
        >>> # data now only contains Ca II examples

        If you have many conditions you want to place on the dataset,
        then I suggest you chain these rules. For example:

        >>> (data.filter(lambda s: dfp.has_props(s, {"type": "Ca II"}))
                 .filter(lambda s: dfp.has_props(s, {"score": lambda v: v < 0.7}))

        Or because `dfp.has_props` can take multiple conditions in one
        statement, we could also have:

        >>> data.filter(lambda s: dfp.has_props(
                 s, {"type": "Ca II", "score": lambda v: v < 0.7}))
        """
        filtered_data = dfp.records_to_dataframe(dfp.lfilter(
            condition, dfp.dataframe_to_records(self.data)))
        new_cls = deepcopy(self)
        new_cls.data = filtered_data
        return new_cls

    def split(self, condition):
        """Split dataset into two by condition

        Split the dataset into two parts by a boolean condition. If
        the condition is True then the sample will be included in the
        `left` part of the split. If the condition is False, the
        sample will be included in the `right`.

        Parameters
        ----------
        condition : Callable
            The condition to place on each sample in the dataset.

        Returns
        -------
        tuple[Dataset, Dataset]
            Two datasets where the left dataset conditions all samples
            for which the condition was True, the right contains all
            samples for which the condition was False.

        Examples
        --------
        >>> s = CloudsDataset()
        >>> l, r = s.split(lambda sample: dfp.has_props(sample, {"type":  "Ca II"}))

        The `l` or left in this example contains all data of the type
        "Ca II", whereas `r` or right, contains all data that is _not_
        "Ca II".

        This could be used to split by train and test, if it was wasn't already split.

        >>> train, test = s.split(lambda sample: dfp.has_props(sample, {"subset": "train"}))

        """
        left = self.filter(condition)
        right = self.filter(dfp.inverse(condition))
        cls_left = deepcopy(self)
        cls_left.data = left
        cls_right = deepcopy(self)
        cls_right.data = right
        return cls_left, cls_right

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"{type(self).__name__}"
                f"\n\t- Number of samples: {len(self)}"
                f"\n\t- Subset: {'train' if self.train else 'test'}"
                f"\n\t- Columns: {self.data.columns.tolist()}"
                f"\n\n{self.data.head(5)}")

    def __getitem__(self, index) -> Union[list, item]:
        if isinstance(index, slice):
            step = index.step if index.step is not None else 1
            start = index.start if index.start is not None else 0
            stop = min(index.stop, len(self)) if index.stop is not None else -1
            return [self[idx] for idx in range(start, stop, step)]
        row = self.data.iloc[index]
        inp = sunpy.map.Map(row["input"])
        oup = sunpy.map.Map(row["target"])
        msk = np.load(row["disk_mask"], allow_pickle=True)
        msk = (msk[None, ...]==False).astype(np.float32)  # inverse the mask to outside disk
        uid = row["uid"]
        if self.transform:
            inp, oup, msk = self.transform(inp, oup, msk)
        return item(input=inp, mask=msk, target=oup, uid=uid)


class CloudsTransform:
    """Basic Transformations for the cloud dataset.

    PyTorch transformations to apply to the CloudsDataset class. This
    transformation pipeline should be used to get the data ready to
    pass to a Neural Network.

    Parameters
    ----------
    hflip_p : float
        The probability of horizontal flipping (0.0 no flipping, 1.0 always flip).
    vflip_p : float
        The probability of vertical flipping (0.0 no flipping, 1.0 always flip).
    tensor_wrapper : Callable[np.ndarray]
        (default: torch.FloatTensor) The function to wrap a numpy array into a
        tensor. This allows you to convert the data into pytorch's tensor, or
        tensorflow's tensor, etc.

    Examples
    --------
    >>> clouds = CloudsDataset()   # create a dataset
    >>> trans  = CloudsTransform() # create the transform
    >>> trans(*clouds[0])  # apply to both input and target

    Or, you can pass this class directly to the dataset upon it's
    construction.

    >>> clouds = CloudsDataset(transform=CloudsTransform())
    >>> clouds[0]  # transform has already been applied

    By default, horizontal and vertical image flipping is turned
    off. By supplying these arguments > 0.0, then flipping can be
    applied to both input and target (such that both input and it's
    target still match after flipping). This value is a percentage,
    thus 0.5 means that flipping will occur 50% of the time.

    >>> clouds = CloudsDataset(
            transform=CloudsTransform(
                hflip_p=0.5,
                vflip_p=0.5))
    >>> clouds[0]   # images are horizontal/vertically flipped 50% of the time.

    If you didn't want to use pytorch, but something else, then you can
    specify the wrapping function in the tensor_wrapper argument:

    >>> clouds = CloudsDataset(
            transform=CloudsTransform(tensor_wrapper=lambda x: x))
    >>> clouds[0]  # images are simply numpy arrays
    >>> def to_tensorflow_tensor(x):
          ...
    >>> clouds = CloudsDataset(
             transform=CloudsTransform(tensor_wrapper=to_tensorflow_tensor))
    >>> clouds[0]  # now they are ready for tensorflow.
    
    """

    def __init__(
            self,
            hflip_p: float = 0.0,
            vflip_p: float = 0.0,
            tensor_wrapper = torch.FloatTensor
    ):
        assert 0.0 <= hflip_p <= 1.0
        assert 0.0 <= vflip_p <= 1.0

        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.tensor_wrapper: Callable[np.ndarray] = tensor_wrapper

    def _basic_transform(self, x):
        x = x.data.astype(np.float32)
        return self.tensor_wrapper(x[None,...])  # add batch dimension

    def __apply_both(self, *args):
        return [f(x) for x in args]

    def __call__(self, *args):
        hflip = random.random() < self.hflip_p
        vflip = random.random() < self.vflip_p
        args = self.__apply_many(self._basic_transform, *args)
        if hflip: args = self.__apply_many(tv.functional.hflip, *args)
        if vflip: args = self.__apply_many(tv.functional.vflip, *args)
        return args
