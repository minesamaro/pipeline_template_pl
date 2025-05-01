from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch
import torchvision
import pydicom
import os
import matplotlib.pyplot as plt

from src.modules.data.data_augmentation.ct_image_augmenter \
    import CTImageAugmenter


class NLSTPreprocessedKFoldDataLoader:
    def __init__(
            self,
            config,
            lung_metadataframe,
            load_data_name=False
    ):
        self.config = config

        self.dataloaders = None
        self.dataloaders_by_subset = None
        self.data_names_by_subset = None
        self.data_splits = None
        self.load_data_name = None
        self.torch_generator = None

        self.dataloaders = defaultdict(list)
        self.dataloaders_by_subset = defaultdict(list)
        self.data_names_by_subset = defaultdict(list)
        self.data_splits = defaultdict(lambda: defaultdict(list))
        self.load_data_name = load_data_name
        self.torch_generator = torch.Generator()

        self.lung_metadataframe = lung_metadataframe

        self.torch_generator.manual_seed(self.config.seed_value)
        self._set_data_splits(self.lung_metadataframe)
        self._set_dataloaders()

    def get_data_names(self):
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(self.config.number_of_k_folds)
        ] for subset_type in ["train", "validation", "test"]}
        return data_names

    def get_dataloaders(self):
        return self.dataloaders

    def _get_torch_dataloader(
            self,
            file_names,
            labels,
            subset_type,
            torch_dataloader_kwargs
    ):
        torch_dataloader = TorchDataLoader(
            dataset=NLSTPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type,
                lung_metadataframe=self.lung_metadataframe
            ),
            generator=self.torch_generator,
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,
            **torch_dataloader_kwargs
        )
        return torch_dataloader

    def _get_torch_dataloader_worker_init_fn(self, worker_id):
        numpy.random.seed(self.config.seed_value + worker_id)
        random.seed(self.config.seed_value + worker_id)

    def _set_dataloaders(self):
        for subset_type in ["train", "validation", "test"]:
            for datafold_id in range(self.config.number_of_k_folds):
                self.dataloaders[subset_type].append(
                    self._get_torch_dataloader(
                        file_names=self.data_splits[subset_type] \
                            ['file_names'][datafold_id],
                        labels=self.data_splits[subset_type] \
                            ['labels'][datafold_id],
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
                )

    def _set_data_splits(self, lung_metadataframe):
        if not self.config.number_of_k_folds: #TODO: Fix problem with K = 0
            train_and_validation_file_name_column, test_file_name_column = \
                train_test_split(
                    lung_metadataframe,
                    test_size=self.config.test_fraction_of_entire_dataset,
                    random_state=self.config.seed_value,
                    stratify=lung_metadataframe['label']
                )
            train_file_name_column, validation_file_name_column = \
                train_test_split(
                    train_and_validation_file_name_column,
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_metadataframe[
                        lung_metadataframe['path'].isin(
                            train_and_validation_file_name_column
                        )
                    ]['label']
                )

            self.data_names_by_subset['train'] = \
                train_file_name_column.tolist()
            self.data_names_by_subset['validation'] = \
                validation_file_name_column.tolist()
            self.data_names_by_subset['test'] = \
                test_file_name_column.tolist()

            for subset_type in ["train", "validation", "test"]: #TODO: Fix this
                self.dataloaders_by_subset[subset_type] = \
                    self._get_torch_dataloader(
                        file_names=self.data_names_by_subset[subset_type],
                        label_dataframe=lung_metadataframe,
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
        else:
            skf_cross_validator = StratifiedKFold(
                n_splits=self.config.number_of_k_folds,
                shuffle=True,
                random_state=self.config.seed_value
            )
            skf_split_generator = skf_cross_validator.split(
                X=lung_metadataframe,
                y=lung_metadataframe['label']
            )

            for datafold_id, (train_and_validation_indexes, test_indexes) \
                    in enumerate(skf_split_generator, 1):
                test_lung_metadataframe = \
                    lung_metadataframe.iloc[test_indexes]
                (
                    train_lung_metadataframe,
                    validation_lung_metadataframe
                ) = train_test_split(
                    lung_metadataframe \
                        .iloc[train_and_validation_indexes],
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_metadataframe['label'] \
                        .iloc[train_and_validation_indexes]
                )

                self.data_splits['train']['file_names'].append(
                    train_lung_metadataframe['path'].tolist()
                )
                self.data_splits['train']['labels'].append(
                    train_lung_metadataframe['label'].tolist()
                )
                self.data_splits['validation']['file_names'].append(
                    validation_lung_metadataframe['path'].tolist()
                )
                self.data_splits['validation']['labels'].append(
                    validation_lung_metadataframe['label'].tolist()
                )
                self.data_splits['test']['file_names'].append(
                    test_lung_metadataframe['path'].tolist()
                )
                self.data_splits['test']['labels'].append(
                    test_lung_metadataframe['label'].tolist()
                )


class NLSTPreprocessedDataLoader(Dataset):
    def __init__(
            self,
            config,
            file_names,
            labels,
            load_data_name,
            subset_type,
            lung_metadataframe
    ):
        self.config = config
        if config.data_augmentation.apply and subset_type == "train":
            self.file_names = (
                file_names
                + config.data_augmentation.augmented_to_original_data_ratio
                * file_names
            )  #TODO: Fix this
        else:
            self.file_names = file_names
        self.labels = labels
        self.load_data_name = load_data_name
        self.subset_type = subset_type
        self.lung_metadataframe = lung_metadataframe

        self.augmented_to_original_data_ratio = \
            config.data_augmentation.augmented_to_original_data_ratio
        self.apply_data_augmentations = config.data_augmentation.apply
        if config.data_augmentation.apply and subset_type == "train":
            self.data_augmenter = CTImageAugmenter(
                parameters=config.data_augmentation.parameters
            )
        self.image_transformer = torchvision.transforms.Compose([
            lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ]) #TODO: Check this transformer

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, data_index):
        data = self._get_data(data_index)
        label = self._get_label(data_index)
        if not self.load_data_name:
            return data, label
        else:
            return self.file_names[data_index], data, label

    def _get_data(self, data_index): #TODO: CHange this to DICOM load

        if int(self.config.dimension) == 2:
            image = self._get_slice(data_index)
        elif int(self.config.dimension) == 3:
            image = self._get_scan(data_index)
        elif int(self.config.dimension) == 2.5:
            image = self._get_2_5(data_index)
        
        if image is None:
            raise ValueError(f"[ERROR] Image is None at index {data_index}. File info: {self.lung_metadataframe.loc[self.lung_metadataframe['path'] == self.file_names[data_index]]}")

        # TODO: Do the same for lung roi and 2.5D and resample

        image = image.astype(numpy.float32)

        if self.apply_data_augmentations and data_index >= (
                len(self.file_names)
                / (self.augmented_to_original_data_ratio + 1)
        ) and self.subset_type == "train":
            image = self.data_augmenter(image=image) #TODO: Figure this out - Change
        image = self.image_transformer(image)
        data = dict(image=image)

        if data_index %50 == 0:
            print(f"[INFO] Data index: {data_index}, Image shape: {image.shape}")
            # Save the image or middle slice of volume to path
            save_path = f"/nas-ctm01/homes/mipaiva/experiment_figures/slice{data_index}.png"

            plt.imsave(save_path, image[-1], cmap='gray')


        return data
    
    def _get_slice(self, data_index):
        try:
            # Load the DICOM file
            dicom_file_path = self.file_names[data_index]
            
            # Go to the metadataframe and get the slice number of the path == dicom_file_path
            slice_number = self.lung_metadataframe.loc[
                self.lung_metadataframe['path'] == dicom_file_path,
                'sct_slice_num'
            ].values[0]
            
            # List CT slices files
            ct_dcms = os.listdir(dicom_file_path)

            # List the DICOM slice files that are read with pydicom.read_file()
            slices = [pydicom.dcmread(os.path.join(dicom_file_path, dcm)) for dcm in ct_dcms]

            if len(slices) < slice_number:
                print('HERE')
                slice_number = len(slices) // 2 # TODO: Fix this about the missing data

            # Order list of slices in an ascendant way by the position z of the slice
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            image = numpy.stack([s.pixel_array for s in slices])
            image = image.astype(numpy.int16)
            image[image == -2000] = 0
                
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(numpy.float64)
                image = image.astype(numpy.int16)
                        
            image += numpy.int16(intercept)
            dicom_image = numpy.array(image, dtype=numpy.int16)

            # Extract the slice from the DICOM image
            slice_image = dicom_image[slice_number - 1]

            return slice_image
        except Exception as e:
            print(f"Error loading slice {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            print(f"Slice number: {slice_number}")
            print(f"Image shape: {dicom_image.shape}")
            return None
    

    def _get_scan(self, data_index):
        if self.config.resample_z == False:
            dicom_image = numpy.load() #TODO: Insert path to the numpy file
        else:
            # Load the DICOM file
            dicom_file_path = self.file_names[data_index]
            
            # List CT slices files
            ct_dcms = os.listdir(dicom_file_path)

            # List the DICOM slice files that are read with pydicom.read_file()
            slices = [pydicom.dcmread(os.path.join(dicom_file_path, dcm)) for dcm in ct_dcms]

            # Order list of slices in an ascendant way by the position z of the slice
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            image = numpy.stack([s.pixel_array for s in slices])
            image = image.astype(numpy.int16)
            image[image == -2000] = 0
                
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(numpy.float64)
                image = image.astype(numpy.int16)
                        
            image += numpy.int16(intercept)
            dicom_image = numpy.array(image, dtype=numpy.int16)

        return dicom_image
    
    def _get_2_5(self, data_index):

        if self.config.n_slices_2_5 != None:
            n_slices = self.config.n_slices_2_5
        else:
            n_slices = 11

        # Go to the metadataframe and get the slice number of the path == dicom_file_path
        slice_number = self.lung_metadataframe.loc[
                self.lung_metadataframe['path'] == dicom_file_path,
                'sct_slice_num'
            ].values[0]

        # Load the DICOM file
        dicom_file_path = self.file_names[data_index]
        
        # List CT slices files
        ct_dcms = os.listdir(dicom_file_path)

        # List the DICOM slice files that are read with pydicom.read_file()
        slices = [pydicom.dcmread(os.path.join(dicom_file_path, dcm)) for dcm in ct_dcms]

        # Order list of slices in an ascendant way by the position z of the slice
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        image = numpy.stack([s.pixel_array for s in slices])
        image = image.astype(numpy.int16)
        image[image == -2000] = 0
            
        intercept = slices[0].RescaleIntercept
        slope = slices[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(numpy.float64)
            image = image.astype(numpy.int16)
                    
        image += numpy.int16(intercept)
        dicom_image = numpy.array(image, dtype=numpy.int16)

        dicom_image = dicom_image[slice_number - n_slices // 2: slice_number + n_slices // 2]

        return dicom_image


    def _get_label(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])

        return labels
