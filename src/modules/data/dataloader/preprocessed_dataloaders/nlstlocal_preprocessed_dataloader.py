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
from collections import defaultdict
import random

from src.modules.data.data_augmentation.ct_image_augmenter \
    import CTImageAugmenter


class NLSTLocalPreprocessedKFoldDataLoader:
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
        if not self.config.number_of_k_folds:
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
        self.load_data_name = load_data_name
        self.subset_type = subset_type
        self.lung_metadataframe = lung_metadataframe
        self.augmented_to_original_data_ratio = config.data_augmentation.augmented_to_original_data_ratio
        self.apply_data_augmentations = config.data_augmentation.apply

        
        if self.apply_data_augmentations and subset_type == "train":
            label_to_files = defaultdict(list)
            for file, label in zip(file_names, labels):
                label_to_files[label].append(file)

            # Identify majority and minority class
            class_counts = {k: len(v) for k, v in label_to_files.items()}
            max_class = max(class_counts, key=class_counts.get)
            min_class = min(class_counts, key=class_counts.get)

            diff = class_counts[max_class] - class_counts[min_class]

            # Sample (with replacement) from the minority class to balance
            additional_files = random.choices(label_to_files[min_class], k=diff)
            self.file_names = file_names + additional_files
            self.labels = labels + [min_class] * diff

            # Track which indices are duplicates/augmented
            original_count = len(file_names)
            self.augmented_indices = set(range(original_count, len(self.file_names)))
        else:
            self.file_names = file_names
            self.labels = labels
            self.augmented_indices = set()

        if config.data_augmentation.apply and subset_type == "train":
            self.data_augmenter = CTImageAugmenter(
                parameters=config.data_augmentation.parameters
            )
        self.image_transformer = torchvision.transforms.Compose([
            lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
            torchvision.transforms.ToTensor(),
        ])

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
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        pid = dataframe_row['pid'].values[0]
        study_yr = dataframe_row['study_yr'].values[0]

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
                data_path = 'C:\\Users\\HP\\OneDrive - Universidade do Porto\\Documentos\\UNIVERSIDADE\\Tese\\PhantomDataset\\2d' #TODO: Change this to the correct path
                image = self._get_slice(data_index, data_path, pid, study_yr)
            elif self.config.dimension == 3:
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/3d'
                image = self._get_scan(data_index, data_path, pid, study_yr)
            elif self.config.dimension == 2.5:
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/25d'
                image = self._get_2_5(data_index, data_path, pid, study_yr)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
             
        if image is None:
            raise ValueError(f"[ERROR] Image is None at index {data_index}. File info: {self.lung_metadataframe.loc[self.lung_metadataframe['path'] == self.file_names[data_index]]}")

        image = image.astype(numpy.float32)

        # Apply augmentation only to duplicated/repeated images
        if (self.apply_data_augmentations and 
            #data_index in self.augmented_indices and  #TODO Change when not local
            self.subset_type == "train"):
            filename = f"augmented_slice_{data_index}.png"
            filename_og = f"original_slice_{data_index}.png"

            save_path = f"C:\\Users\\HP\\OneDrive - Universidade do Porto\\Documentos\\UNIVERSIDADE\\Tese\\PhantomDataset\\dataaug\\{filename_og}"

            plt.imsave(save_path, image, cmap='gray')
            image = self.data_augmenter(image=image)

            # Save image to disk for debugging purposes
            
            save_path = f"C:\\Users\\HP\\OneDrive - Universidade do Porto\\Documentos\\UNIVERSIDADE\\Tese\\PhantomDataset\\dataaug\\{filename}"
            plt.imsave(save_path, image[-1], cmap='gray')

        
        image = self.image_transformer(image)
        data = dict(image=image)

        # Get the metadata row corresponding to the current file
        #meta_row = self.lung_metadataframe.loc[
        #    self.lung_metadataframe['path'] == self.file_names[data_index]
        #]

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
    
    def _get_slice(self, data_index, data_path, pid, study_yr):
        try:
            slice_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

            #if self.config.resize:
            #    slice_image = numpy.resize(slice_image, (224, 224))

            return slice_image
        except Exception as e:
            print(f"Error loading slice {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            return None
    

    def _get_scan(self, data_index, data_path, pid, study_yr):
        if self.config.resample_z:
            dicom_image = numpy.load() #TODO: Insert path to the numpy file
        else:
            dicom_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

            n_slices = 32

            # Compute start and end slice indices
            center = dicom_image.shape[0] // 2
            start = max(center - n_slices // 2, 0)
            end = min(center + n_slices // 2, dicom_image.shape[0])

            # Extract the central volume
            dicom_image = dicom_image[start:end, :, :]

            if self.config.resize:
                dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image
    
    def _get_2_5(self, data_index, data_path, pid, study_yr):
        # if different n_slices, read from 3D and change the number TODO
        
        dicom_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

        if self.config.resize:
            dicom_image = numpy.resize(dicom_image, (dicom_image.shape[0], 224, 224))

        return dicom_image


    def _get_label(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])

        return labels
    
