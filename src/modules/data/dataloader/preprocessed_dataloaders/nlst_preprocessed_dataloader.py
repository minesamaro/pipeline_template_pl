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
        folds = self.config.number_of_k_folds
        if folds == 0:
            folds = 1
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(folds)
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
        folds = self.config.number_of_k_folds
        if folds == 0: # Works for no cross validation and cross validation 
            # When n_folds is 0 it works as if it was just one data split, adding another diemnsion to the datalaoder dictionary
            folds = 1
        for subset_type in ["train", "validation", "test"]:
            for datafold_id in range(folds):
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
                # print the label distribution of each subset_type
                labels, counts = numpy.unique(self.data_splits[subset_type]['labels'][datafold_id], return_counts=True)
                label_dist_str = ", ".join([f"{label}: {count}" for label, count in zip(labels, counts)])
                print(f"[INFO] {subset_type.capitalize()} dataloader {datafold_id} label distribution: {label_dist_str}")


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
                    stratify=train_and_validation_file_name_column['label']
                )

            self.data_names_by_subset['train'] = \
                train_file_name_column['path'].tolist()
            self.data_names_by_subset['validation'] = \
                validation_file_name_column['path'].tolist()
            self.data_names_by_subset['test'] = \
                test_file_name_column['path'].tolist()

            self.data_splits['train']['file_names'].append(
                train_file_name_column['path'].tolist()
            )
            self.data_splits['train']['labels'].append(
                train_file_name_column['label'].tolist()
            )
            self.data_splits['validation']['file_names'].append(
                validation_file_name_column['path'].tolist()
            )
            self.data_splits['validation']['labels'].append(
                validation_file_name_column['label'].tolist()
            )
            self.data_splits['test']['file_names'].append(
                test_file_name_column['path'].tolist()
            )
            self.data_splits['test']['labels'].append(
                test_file_name_column['label'].tolist()
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

    def _transform(self, image):
        if image.ndim == 3:
            image_transformer = torchvision.transforms.Compose([
                lambda x: numpy.transpose(x, axes=(1, 2, 0)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
            ])
        else:
            image_transformer = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
            ])
        return image

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, data_index):
        try:
            data = self._get_data(data_index)
            label = self._get_label(data_index)
            if not self.load_data_name:
                return data, label
            else:
                return self.file_names[data_index], data, label
        except Exception as e:
            print(f"[ERROR] Error in __getitem__ at index {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            print(f"Label: {self.labels[data_index]}")
            raise e

    def _get_data(self, data_index):
        if getattr(self.config, "random", False):  # If config.random exists and is True
            if self.config.dimension == 2:
                image = numpy.random.rand(512, 512).astype(numpy.float32)
            elif self.config.dimension == 2.5:
                image = numpy.random.rand(10, 512, 512).astype(numpy.float32)
            elif self.config.dimension == 3:
                image = numpy.random.rand(512, 512, 32).astype(numpy.float32)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
        else:
            if self.config.dimension == 2:
                image = self._get_slice(data_index)
            elif self.config.dimension == 3:
                image = self._get_scan(data_index)
            elif self.config.dimension == 2.5:
                image = self._get_2_5(data_index)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
             
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

        # Get the metadata row corresponding to the current file
        meta_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]

#        if not meta_row.empty:
#            pid = meta_row['pid'].values[0]
#            study = meta_row['study_yr'].values[0]
#            
#            filename = f"slice_{pid}_{study}.png"
#            save_path = f"/nas-ctm01/homes/mipaiva/experiment_figures/{filename}"
#            plt.imsave(save_path, image[-1], cmap='gray')
#        else:
#            print(f"[WARNING] No metadata found for file: {self.file_names[data_index]}")

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

            # Order list of slices in an ascendant way by the position z of the slice
            slices.sort(key = lambda x: float(x.InstanceNumber))
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

            # Normalize the slice
            slice_image = self._normalize(slice_image)

            if self.config.resize:
                slice_image = numpy.resize(slice_image, (224, 224))

            return slice_image
        except Exception as e:
            print(f"Error loading slice {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            print(f"Slice number: {slice_number}")
            print(f"Image shape: {dicom_image.shape}")
            return None
    

    def _get_scan(self, data_index):
        if self.config.resample_z:
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

            # keep only the 10 middle slices
            n_slices = 32
            slices = slices[len(slices) // 2 - n_slices // 2: len(slices) // 2 + n_slices // 2]

            image = numpy.stack([s.pixel_array for s in slices])
            image = image.astype(numpy.int16)
            image[image == -2000] = 0
                
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(numpy.float64)
                image = image.astype(numpy.int16)
                        
            image += numpy.int16(intercept)
            # Normalization
            image = self._normalize(image)
            
            dicom_image = numpy.array(image, dtype=numpy.int16)

            if self.config.resize:
                dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image
    
    def _get_2_5(self, data_index):
        if 'n_slices_2_5' in self.config:
            n_slices = self.config.n_slices_2_5
        else:
            n_slices = 10
        

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

        # Normalize the slice
        dicom_image = self._normalize(dicom_image)

        if self.config.resize:
            dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image


    def _get_label(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])

        return labels
    
    def _normalize(self, scan, minimum=-1000, maximum=400):
        scan = scan.astype(numpy.float32)  # ensure float
        scan = (scan - minimum) / (maximum - minimum)
        scan = numpy.clip(scan, 0, 1)      # clip values between 0 and 1
        return scan
