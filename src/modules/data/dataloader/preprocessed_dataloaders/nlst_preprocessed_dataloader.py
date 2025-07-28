from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader, WeightedRandomSampler
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
    import CTImageAugmenter, CTImageAugmenter3D

from src.modules.data.dataloader.visualizationuploader \
    import VisualizationUploader


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
        if self.config.weighted_random_sampler:
            print(f"\nUsing WeightedRandomSampler for {subset_type} subset")
            dataset = NLSTPreprocessedDataLoader(
                config=self.config,
                file_names=file_names,
                labels=labels,
                load_data_name=self.load_data_name,
                subset_type=subset_type,
                lung_metadataframe=self.lung_metadataframe
            )

            if subset_type == "train":
                # Convert labels to numpy for processing
                labels_np = numpy.array(labels)
                class_counts = numpy.bincount(labels_np)
                class_weights = 1. / class_counts
                # Assign weight to each sample
                sample_weights = class_weights[labels_np]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False  # Disable shuffle when using sampler
                torch_dataloader = TorchDataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    generator=self.torch_generator,
                    worker_init_fn=self._get_torch_dataloader_worker_init_fn,
                    **torch_dataloader_kwargs
                )
            else:
                torch_dataloader = TorchDataLoader(
                    dataset=dataset,
                    shuffle=False,  # Validation/test can be shuffled normally or not
                    generator=self.torch_generator,
                    worker_init_fn=self._get_torch_dataloader_worker_init_fn,
                    **torch_dataloader_kwargs
                )
        else:
            print(f"\nUsing regular DataLoader for {subset_type} subset")
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
            print(f"\n✅ Set {subset_type} dataloaders with {len(self.dataloaders[subset_type])} folds")
            # Print the number of samples in batches for each fold
            for fold_id, dataloader in enumerate(self.dataloaders[subset_type], 1):
                num_samples = len(dataloader.dataset)
                print(f"  - Fold {fold_id}: {num_samples} samples")
                
                # Print the distribution of labels in the fold
                
                labels = dataloader.dataset.labels
                unique_labels, counts = numpy.unique(labels, return_counts=True)
                label_distribution = dict(zip(unique_labels, counts))
                print(f"    Label distribution: {label_distribution}")

                # Print the distribution of the batches
                for batch in dataloader:
                    # Print the distribution of 0 and 1 labels in the batch
                    labels = batch[1]
                    unique_labels, counts = numpy.unique(labels, return_counts=True)
                    label_distribution = dict(zip(unique_labels, counts))
                    print(f"    Batch label distribution: {label_distribution}")




    def _set_data_splits(self, lung_metadataframe):
        metadata_with_splits = lung_metadataframe.copy()

        # === Step 1: Fixed stratified test split ===
        train_val_df, test_df = train_test_split(
            lung_metadataframe,
            test_size=self.config.test_fraction_of_entire_dataset,
            random_state=self.config.seed_value,
            stratify=lung_metadataframe['label']
        )

        if not self.config.number_of_k_folds:
            self.config.number_of_k_folds = 1 #Doesnt work

        # Assign test split once (same for all folds)
        for fold_id in range(1, self.config.number_of_k_folds + 1):
            metadata_with_splits.loc[test_df.index, f'split_fold_{fold_id}'] = 'test'

        # === Step 2: Stratified K-Fold on train_val_df only ===
        skf = StratifiedKFold(
            n_splits=self.config.number_of_k_folds,
            shuffle=True,
            random_state=self.config.seed_value
        )
        # Missing a generator, no??

        for fold_id, (train_idx, val_idx) in enumerate(
            skf.split(train_val_df, train_val_df['label']), 1
        ):
            train_split = train_val_df.iloc[train_idx]
            val_split = train_val_df.iloc[val_idx]

            # Assign to DataFrame
            metadata_with_splits.loc[train_split.index, f'split_fold_{fold_id}'] = 'train'
            metadata_with_splits.loc[val_split.index, f'split_fold_{fold_id}'] = 'val'

            # Save in internal structure if needed
            self.data_splits['train']['file_names'].append(train_split['path'].tolist())
            self.data_splits['train']['labels'].append(train_split['label'].tolist())
            self.data_splits['validation']['file_names'].append(val_split['path'].tolist())
            self.data_splits['validation']['labels'].append(val_split['label'].tolist())
            self.data_splits['test']['file_names'].append(test_df['path'].tolist())
            self.data_splits['test']['labels'].append(test_df['label'].tolist())

        # === Save annotated DataFrame to CSV ===
        metadata_with_splits.to_csv(
            '/nas-ctm01/homes/mipaiva/small_scripts/lung_metadata_with_splits.csv',
            index=False
        )
        print("\n✅ Saved split assignments to 'lung_metadata_with_splits.csv'")


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
        
        # Check if there is a roi parameter in config
        if 'roi' in config:
            self.roi = config.roi
            if self.roi not in ['lung', 'masked']:
                if self.config.dimension == 2:
                    self.roi = 'ws'
                elif self.config.dimension == 3:
                    self.roi = 'ws'
            print(f"Using ROI: {self.roi}")
        else:
            self.roi = 'ws'

        self.visualization = config.visualize_imgur
        if self.visualization:
            print("\n✅ Visualization enabled for NLSTLocalPreprocessedDataLoader")
            self.visualization_uploader = VisualizationUploader(
                client_id='f5a89997db63c60',
                album_id='WC0PErb6jxLRWHt'
            )

        if self.apply_data_augmentations and subset_type == "train":
            print("Data Aug")
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
            print(f"Original shape: {original_count}, New Shape: {len(self.file_names)}")
        else:
            self.file_names = file_names
            self.labels = labels
            self.augmented_indices = set()

        if config.data_augmentation.apply and subset_type == "train":
            if config.dimension == 3:
                self.data_augmenter = CTImageAugmenter3D(
                    parameters=config.data_augmentation.parameters
                )
            else:
                # Use CTImageAugmenter for 2D and 2.5D
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

    def get_slice_range_3d(self, total_slices, slice_idx, n_slices):
        """
        Calculates a robust slice range around slice_idx, always returning n_slices if possible.
        If slice_idx is None, it defaults to the center of the volume.
        """
        if slice_idx is None:
            slice_idx = total_slices // 2

        half = n_slices // 2

        # Initial guess
        start = slice_idx - half
        end = slice_idx + half + (0 if n_slices % 2 == 0 else 1)

        # Clamp to volume bounds
        if start < 0:
            end += abs(start)
            start = 0
        if end > total_slices:
            excess = end - total_slices
            start = max(0, start - excess)
            end = total_slices

        # Final adjustment to ensure exactly n_slices
        current_len = end - start
        if current_len < n_slices:
            if start > 0:
                missing = n_slices - current_len
                shift = min(missing, start)
                start -= shift
                current_len = end - start
            if current_len < n_slices and end < total_slices:
                missing = n_slices - current_len
                shift = min(missing, total_slices - end)
                end += shift

        return start, end
    
    def _get_data(self, data_index):
        dataframe_row = self.lung_metadataframe.loc[
            self.lung_metadataframe['path'] == self.file_names[data_index]
        ]
        pid = dataframe_row['pid'].values[0]
        study_yr = dataframe_row['study_yr'].values[0]
        reversed = dataframe_row['reversed'].values[0]
        if dataframe_row['sct_slice_num'].values[0] is None:
            slice_idx = None
            print(f"[WARNING] No slice index found for {self.file_names[data_index]}. Using None.")
        else:
            slice_idx = int(dataframe_row['sct_slice_num'].values[0])


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
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/2d'
                data_path = os.path.join(data_path, self.roi) if self.roi else data_path
                image = self._get_slice(data_index, data_path, pid, study_yr)

            elif self.config.dimension == 3:
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/general_shift_crop'
                data_path = os.path.join(data_path, self.roi) if self.roi else data_path
                image = self._get_scan(data_index, data_path, pid, study_yr, slice_idx, reversed)
            elif self.config.dimension == 2.5: # TODO refix the other dimensions
                data_path = '/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/nlst/preprocessed_data/protocol_5/25d'
                image = self._get_2_5(data_index, data_path, pid, study_yr)
            else:
                raise ValueError(f"[ERROR] Unknown dimension {self.config.dimension}")
             
        if image is None:
            raise ValueError(f"[ERROR] Image is None at index {data_index}. File info: {self.lung_metadataframe.loc[self.lung_metadataframe['path'] == self.file_names[data_index]]}")

        # TODO: Do the same for lung roi and 2.5D and resample

        image = image.astype(numpy.float32)
        # Apply augmentation only to duplicated/repeated images
        if (self.apply_data_augmentations and 
            data_index in self.augmented_indices and 
            self.subset_type == "train"):
            
            image = self.data_augmenter(image)
            # Squeeze the image to remove single-dimensional entries
            if image.ndim == 3 and self.config.dimension != 3:
                image = numpy.squeeze(image)
            elif image.ndim == 4 and self.config.dimension == 3:
                print(f"Image shape before squeeze: {image.shape}")
                image = numpy.squeeze(image, axis=-1)
                print(f"Image shape after augmentation: {image.shape}")

        if self.visualization:
            self.visualization_uploader.upload_image(
                image=image,  # Assuming the last slice is the one to visualize
                file_name= f"slice_{pid}_{study_yr}.png",
                dataset_name="NLSTPreprocessed"
            )
        
        image = self.image_transformer(image)
        data = dict(image=image)

        return data
    
    def _get_slice(self, data_index, data_path, pid, study_yr):
        try:
            slice_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{study_yr}.npy"
                )
            )

            # if self.config.resize:
            #     slice_image = numpy.resize(slice_image, (224, 224))
            

            return slice_image
        except Exception as e:
            print(f"Error loading slice {data_index}: {e}")
            print(f"File path: {self.file_names[data_index]}")
            return None
    

    def get_slice_range(total_slices, slice_idx, n_slices):
        """
        Calculates a robust slice range around slice_idx, always returning n_slices if possible.
        If slice_idx is None, it defaults to the center of the volume.
        """
        if slice_idx is None:
            slice_idx = total_slices // 2

        half = n_slices // 2

        # Initial guess
        start = slice_idx - half
        end = slice_idx + half + (0 if n_slices % 2 == 0 else 1)

        # Clamp to volume bounds
        if start < 0:
            end += abs(start)
            start = 0
        if end > total_slices:
            excess = end - total_slices
            start = max(0, start - excess)
            end = total_slices

        # Final adjustment to ensure exactly n_slices
        current_len = end - start
        if current_len < n_slices:
            if start > 0:
                missing = n_slices - current_len
                shift = min(missing, start)
                start -= shift
                current_len = end - start
            if current_len < n_slices and end < total_slices:
                missing = n_slices - current_len
                shift = min(missing, total_slices - end)
                end += shift

        return start, end

    
    def _get_scan(self, data_index, data_path, pid, study_yr, slice_idx, reversed):
        if self.config.resample_z:
            dicom_image = numpy.load() #TODO: Insert path to the numpy file
        else:
            dicom_image = numpy.load(
                os.path.join(
                    data_path,
                    f"{pid}_{self.roi}.npy"
                )
            )

            n_slices = 9

            # Compute start and end slice indices from the center of the nodule
            # Based on the metadataframe
            start, end = self.get_slice_range_3d(
                total_slices=dicom_image.shape[0],
                slice_idx= slice_idx,
                n_slices=n_slices
            )

            # Extract the central volume
            dicom_image = dicom_image[start:end, :, :]

            # if reversed:
            #     dicom_image = numpy.flip(dicom_image, axis=0)

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
    
