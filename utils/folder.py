# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:04:56 2022

@author: Zhuohui Chen
"""

from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import torch ###
import random ###
import librosa ###
import os.path
import numpy as np ###
import torchvision ###
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    frames_per_clip: int = None, ###
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    ###
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        if frames_per_clip is not None:
            instance = []
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    if frames_per_clip is not None:
                        instance.append(item)
                    else:
                        instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)
        if frames_per_clip is not None:
            instances.append(instance)
    ###

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances



class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        extractor (callable, optional): A function to extract a sample features.
        word_method (callable, optional): The function used to cut strings
            into sequential examples.
        stop_words (str): Tokens to discard during the preprocessing step.
        lower (bool): Whether to lowercase the text.
        tokenizer (callable, optional): The function used to tokenize strings
            into sequential examples.
        text_len (int): length of word vector.
        audio_len (int): length of audio feature.
        frames_per_clip (int): number of frames in a clip.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            extractor: Optional[Callable] = None, ###
            word_method: Optional[Callable] = None, ###
            stop_words: str = None, ###
            lower: bool = False, ###
            tokenizer: Optional[Callable] = None, ###
            text_len: int = None, ###
            audio_len: int = None, ###
            frames_per_clip: int = None, ###
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, frames_per_clip) ###

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.extractor = extractor ###
        self.word_method = word_method ###
        self.stop_words = stop_words ###
        self.lower = lower ###
        self.tokenizer = tokenizer ###
        self.text_len = text_len ###
        self.audio_len = audio_len ###
        self.frames_per_clip = frames_per_clip ###


    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        frames_per_clip: int = None, ###
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): Root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            frames_per_clip (int): number of frames in a clip.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, frames_per_clip=frames_per_clip) ###



    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        ###
        samples = self.samples[index]
        if self.loader.__name__ == 'default_loader':
            path, target = samples
            sample = self.loader(path, self.extractor)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(path)
        elif self.loader.__name__ == 'text_loader':
            path, target = samples
            sample = self.loader(path, word_method=self.word_method, stop_words=self.stop_words,
                                 lower=self.lower, text_len=self.text_len, extractor=self.tokenizer)
            if self.stop_words != '' and self.tokenizer is not None:
                sample = self.tokenizer.encode(sample, ).int().numpy()
                if len(sample) < self.text_len:
                    sample = np.pad(sample, (0, self.text_len - len(sample))).reshape(1, -1)
                else:
                    sample = sample[:self.text_len].reshape(1, -1)
                if self.transform is not None:
                    sample = self.transform(sample)
                    sample = torch.squeeze(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
        elif self.loader.__name__ == 'audio_loader':
            path, target = samples
            sample = self.loader(path, self.audio_len, self.extractor)
            if self.transform is not None:
                sample = self.transform(sample)
                sample = torch.squeeze(sample, axis=0)
            if self.target_transform is not None:
                target = self.target_transform(target)
        elif self.loader.__name__ == 'video_loader':
            idx = random.randint(0, len(samples) - self.frames_per_clip)
            for i in range(self.frames_per_clip):
                path, _ = samples[idx]
                samplei = self.loader(path)
                if self.transform is not None:
                    samplei = self.transform(samplei)
                    if i == 0:
                        sample = torch.unsqueeze(samplei, 1)
                    else:
                        sample = torch.cat([sample, torch.unsqueeze(samplei, 1)], axis=1)
                else:
                    if i == 0:
                        sample = np.expand_dims(samplei, 1)
                    else:
                        sample = np.concatenate([sample, np.expand_dims(samplei, 1)], axis=1)
                if self.target_transform is not None:
                    targeti = torch.tensor([self.target_transform(path)])
                    if i == 0:
                        target = targeti
                    else:
                        target = torch.cat([target, targeti], axis=0)
                idx += 1
        ###

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)



###
def make_combine_dataset(directory: str, frames_per_clip: int = None) -> List[Tuple[str, str]]:
    """Generates a list of samples of a form (path_to_sample, path_to_target).

    See :class:`CombineDatasetFolder` for details.
    """

    t = os.listdir(directory+'/labels/')  # label files
    if frames_per_clip is not None:
        sample_files = []
        target_files = []
        for i in t:
            target_file = os.listdir(directory+'/labels/'+i)
            target_files.append([directory + '/labels/' + i + '/' + j for j in target_file])
            sample_files.append([directory + '/samples/' + j.rsplit('.', 1)[0] + '.jpg' for j in target_file])
    else:
        target_files = [directory + '/labels/' + x for x in t if '.' + x.rsplit('.', 1)[-1].lower() in IMG_EXTENSIONS]
        sample_files = [directory + '/samples/' + x.rsplit('.', 1)[0] + '.jpg' for x in t if '.' + x.rsplit('.', 1)[-1].lower() in IMG_EXTENSIONS]

    return list(zip(sample_files, target_files))



class RandomCrop():
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(image, output_size):
        _, h, w = image.shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:

            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()

        return i, j, th, tw

    def __call__(self, image, label):
        i, j, h, w = self.get_params(image, self.size)

        return torchvision.transforms.functional.crop(image, i, j, h, w), torchvision.transforms.functional.crop(label, i, j, h, w)



class CombineDatasetFolder(VisionDataset):
    """A generic combine data loader.

    Args:
        root (string): Root directory path of samples.
        images_loader (callable, optional): A function to load a sample given its path.
        labels_loader (callable, optional): A function to load a label given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        images_extractor (callable, optional): A function to extract a sample features.
        labels_extractor (callable, optional): A function to extract a label features.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        crop_size (int): Desired output size of the crop.
        text_len (int): length of word vector.
        audio_len (int): length of audio feature.
        frames_per_clip (int): number of frames in a clip.

     Attributes:
        samples (list): List of (image path, label path) tuples
    """

    def __init__(
            self,
            root: str,
            images_loader: Callable[[str], Any],
            labels_loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            images_extractor: Optional[Callable] = None,
            labels_extractor: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            crop_size: int = None,
            text_len: int = None,
            audio_len: int = None,
            frames_per_clip: int = None,
    ) -> None:
        super(CombineDatasetFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        samples = self.make_combine_dataset(self.root, frames_per_clip)

        self.images_loader = images_loader
        self.labels_loader = labels_loader
        self.images_extractor = images_extractor
        self.labels_extractor = labels_extractor
        self.extensions = extensions

        self.samples = samples
        self.crop_size = crop_size
        self.text_len = text_len
        self.audio_len = audio_len
        self.frames_per_clip = frames_per_clip


    @staticmethod
    def make_combine_dataset(directory: str, frames_per_clip: int = None) -> List[Tuple[str, str]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (list): Root dataset directory of images, corresponding to ``self.root``.
            frames_per_clip (int): number of frames in a clip.

        Returns:
            List[Tuple[str, str]]: samples of a form (path_to_sample, path_to_target)
        """
        return make_combine_dataset(directory, frames_per_clip=frames_per_clip)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        samples = self.samples[index]
        if self.images_loader.__name__ == 'default_loader':
            path, target = samples
            sample = self.images_loader(path, self.images_extractor)
            target = self.labels_loader(target, self.labels_extractor, image=False)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.text_len is not None:
                if len(target) < self.text_len:
                    target = torch.nn.functional.pad(target, (0, self.text_len - len(target)))
                elif len(sample) > self.text_len:
                    target = target[: self.text_len]
                if self.target_transform is not None:
                    target = self.target_transform(target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.crop_size is not None:
                sample, target = RandomCrop((self.crop_size, self.crop_size))(sample, target)
        elif self.images_loader.__name__ == 'audio_loader':
            path, target = samples
            sample = self.images_loader(path, self.images_extractor, self.audio_len)
            target = self.labels_loader(target, self.labels_extractor, self.text_len)
            if self.transform is not None:
                sample = self.transform(sample)
        elif self.images_loader.__name__ == 'video_loader':
            idx = random.randint(0, len(samples) - self.frames_per_clip)
            for i in range(self.frames_per_clip):
                path, _ = samples[idx]
                samplei = self.images_loader(path, self.images_extractor)
                if self.transform is not None:
                    samplei = self.transform(samplei)
                    if i == 0:
                        sample = torch.unsqueeze(samplei, 1)
                    else:
                        sample = torch.cat([sample, torch.unsqueeze(samplei, 1)], axis=1)
                else:
                    if i == 0:
                        sample = np.expand_dims(samplei, 1)
                    else:
                        sample = np.concatenate([sample, np.expand_dims(samplei, 1)], axis=1)
                if self.target_transform is not None:
                    targeti = torch.tensor([self.target_transform(path)])
                    if i == 0:
                        target = targeti
                    else:
                        target = torch.cat([target, targeti], axis=0)
                if self.crop_size is not None:
                    sample, target = RandomCrop((self.crop_size, self.crop_size))(sample, target)
                idx += 1

        return sample, target


    def __len__(self) -> int:
        return len(self.samples)
###



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', 'jfif', '.txt', '.wav', '.avi', '.au') ###


def pil_loader(path: str, image: bool) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if image:
            return img.convert('RGB')
        else:
            return img.convert('P')


# TODO: specify the return type
def accimage_loader(path: str, image: bool) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path, image)


def default_loader(path: str, extractor: Optional[Callable] = None, image: bool = True) -> Any: ###
    from torchvision import get_image_backend
    if extractor is not None:
        img = pil_loader(path, image)
        return image_extractor(img, extractor)
    else:
        if get_image_backend() == 'accimage':
            return accimage_loader(path)
        else:
            return pil_loader(path, image)


###
def text_loader(path: str, word_method: Optional[Callable], stop_words: str, lower: bool, text_len: int, extractor: Optional[Callable]) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        if stop_words != '':
            stop_words = list(map(lambda i: i.rstrip('\n'), stop_words))
            if word_method is not None:
                text = word_method(text)
                if lower == False or lower is None:
                    return ' '.join([i for i in text if i not in stop_words])
                elif lower == True:
                    return ' '.join([i.lower() for i in text if i not in stop_words])
            else:
                if lower == False or lower is None:
                    return ' '.join([i for i in text.split(' ') if i not in stop_words])
                elif lower == True:
                    return ' '.join([i.lower() for i in text.split(' ') if i not in stop_words])
        else:
            return text_extractor(text, extractor, text_len)


def audio_loader(path: str, audio_len: int, extractor: Optional[Callable]) -> Any:
    audio, _ = librosa.load(path)
    if extractor is not None:
        return audio_extractor(audio, extractor, audio_len)
    else:
        return np.mean(librosa.feature.mfcc(audio, n_mfcc=audio_len).T, axis=0).astype(np.float32).reshape(1, -1)


def image_extractor(img: Optional[Callable], extractor: Optional[Callable]) -> Any:
    return extractor(images=img, return_tensors='pt')


def text_extractor(text: str, extractor: Optional[Callable], text_len: int) -> Any:
    return extractor(text, return_tensors='pt', truncation=True, padding='max_length', max_length=text_len)


def audio_extractor(audio: Optional[Callable], extractor: Optional[Callable], audio_len: int) -> Any:
    return extractor(audio, return_tensors='pt', truncation=True, max_length=audio_len, sampling_rate=16000)
###



class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        extractor (callable, optional): A function to extract an sample features.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            extractor: Optional[Callable] = None, ###
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          extractor=extractor, ###
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples



###
class TextFolder(DatasetFolder):
    """A generic data loader where the texts are arranged in this way by default: ::

        root/neg/xxx.txt
        root/neg/xxy.txt
        root/neg/[...]/xxz.txt

        root/pos/123.txt
        root/pos/nsdf3.txt
        root/pos/[...]/asd932_.txt

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an text
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an text given its path.
        is_valid_file (callable, optional): A function that takes path of an txt file
            and check if the file is a valid file (used to check of corrupt files)
        word_method (callable, optional): The function used to cut strings
            into sequential examples.
        stop_words (str): Tokens to discard during the preprocessing step.
        lower (bool): Whether to lowercase the text.
        tokenizer (callable, optional): The function used to tokenize strings
            into sequential examples.
        text_len (int): length of word vector.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (text path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = text_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            word_method: Optional[Callable] = None,
            stop_words: str = '',
            lower: bool = None,
            tokenizer: Optional[Callable] = None,
            text_len: int = None,
    ):
        super(TextFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                         transform=transform,
                                         target_transform=target_transform,
                                         is_valid_file=is_valid_file,
                                         word_method=word_method,
                                         stop_words=stop_words,
                                         lower=lower,
                                         tokenizer=tokenizer,
                                         text_len=text_len)
        self.texts = self.samples



class AudioFolder(DatasetFolder):
    """A generic data loader where the audios are arranged in this way by default: ::

        root/genres/xxx.wav
        root/genres/xxy.wav
        root/genres/[...]/xxz.wav

        root/rock/123.wav
        root/rock/nsdf3.wav
        root/rock/[...]/asd932_.wav

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        audio_len (int): length of audio feature.
        extractor (callable, optional): A function to extract an sample features.
        transform (callable, optional): A function/transform that takes in an array
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an audio given its path.
        is_valid_file (callable, optional): A function that takes path of an audio file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (text path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            audio_len: int = None,
            extractor: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = audio_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(AudioFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          extractor=extractor,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          audio_len=audio_len)
        self.audios = self.samples



class VideoFolder(DatasetFolder):
    """A generic data loader where the video frames are arranged in this way by default: ::

        root/abseiling/xxx.png
        root/abseiling/xxy.png
        root/abseiling/[...]/xxz.png

        root/zumba/123.png
        root/zumba/nsdf3.png
        root/zumba/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        frames_per_clip (int): number of frames in a clip.
        transform (callable, optional): A function/transform that takes in an PIL video frame
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video frame given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (video frame path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            frames_per_clip: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(VideoFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          frames_per_clip=frames_per_clip)
        self.videos = self.samples



class ImageAndImageFolder(CombineDatasetFolder):
    """A generic combine data loader where the images and labels are arranged in this way by default: ::

        root/samples/xxx.jpg
        root/samples/xxy.jpg
        root/samples/[...]/xxz.jpg

        root/labels/xxx.png
        root/labels/xxy.png
        root/labels/[...]/xxz.png

    This class inherits from :class:`~torchvision.datasets.CombineDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path of images.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        crop_size (int): Desired output size of the crop.

     Attributes:
        samples (list): List of (image path, label path) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            crop_size: int = None,
    ):
        super(ImageAndImageFolder, self).__init__(root, loader, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file,
                                                  crop_size=crop_size)



class ImageAndTextFolder(CombineDatasetFolder):
    """A generic combine data loader where the images and labels are arranged in this way by default: ::

        root/samples/xxx.png
        root/samples/xxy.png
        root/samples/[...]/xxz.png

        root/labels/xxx.txt
        root/labels/xxy.txt
        root/labels/[...]/xxz.txt

    This class inherits from :class:`~torchvision.datasets.CombineDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path of images.
        text_len (int): length of word vector.
        images_extractor (callable, optional): A function to extract an image features.
        labels_extractor (callable, optional): A function to extract an text features.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        sample_loader (callable, optional): A function to load an image given its path.
        label_loader (callable, optional): A function to load an text given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        samples (list): List of (image path, label path) tuples
    """

    def __init__(
            self,
            root: str,
            text_len: int,
            images_extractor: Optional[Callable],
            labels_extractor: Optional[Callable],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            sample_loader: Callable[[str], Any] = default_loader,
            label_loader: Callable[[str], Any] = text_extractor,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageAndTextFolder, self).__init__(root, sample_loader, label_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 images_extractor=images_extractor,
                                                 labels_extractor=labels_extractor,
                                                 is_valid_file=is_valid_file,
                                                 text_len=text_len)



class VideoAndVideoFolder(CombineDatasetFolder):
    """A generic combine data loader where the videos and labels are arranged in this way by default: ::

        root/samples/xxx.png
        root/samples/xxy.png
        root/samples/[...]/xxz.png

        root/labels/xxx.png
        root/labels/xxy.png
        root/labels/[...]/xxz.png

    This class inherits from :class:`~torchvision.datasets.CombineDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path of videos.
        frames_per_clip (int): number of frames in a clip.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        crop_size (int): Desired output size of the crop.

     Attributes:
        samples (list): List of (image path, label path) tuples
    """

    def __init__(
            self,
            root: str,
            frames_per_clip: int,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            crop_size: int = None,
    ):
        super(VideoAndVideoFolder, self).__init__(root, loader, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                  transform=transform,
                                                  target_transform=target_transform,
                                                  is_valid_file=is_valid_file,
                                                  frames_per_clip=frames_per_clip,
                                                  crop_size=crop_size)



class AudioAndTextFolder(CombineDatasetFolder):
    """A generic combine data loader where the audios and labels are arranged in this way by default: ::

        root/samples/xxx.wav
        root/samples/xxy.wav
        root/samples/[...]/xxz.wav

        root/labels/xxx.txt
        root/labels/xxy.txt
        root/labels/[...]/xxz.txt

    This class inherits from :class:`~torchvision.datasets.CombineDatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path of audios.
        audio_len (int): length of audio feature.
        text_len (int): length of word vector.
        images_extractor (callable, optional): A function to extract an audio features.
        labels_extractor (callable, optional): A function to extract an text features.
        transform (callable, optional): A function/transform that takes in an array
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        sample_loader (callable, optional): A function to load an audio given its path.
        label_loader (callable, optional): A function to load an text given its path.
        is_valid_file (callable, optional): A function that takes path of an audio file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        samples (list): List of (audio path, label path) tuples
    """

    def __init__(
            self,
            root: str,
            audio_len: int,
            text_len: int,
            labels_extractor: Optional[Callable],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            sample_loader: Callable[[str], Any] = audio_loader,
            label_loader: Callable[[str], Any] = text_extractor,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(AudioAndTextFolder, self).__init__(root, sample_loader, label_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 labels_extractor=labels_extractor,
                                                 is_valid_file=is_valid_file,
                                                 text_len=text_len,
                                                 audio_len=audio_len)
###