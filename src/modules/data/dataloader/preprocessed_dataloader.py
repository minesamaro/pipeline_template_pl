from src.modules.data.dataloader.preprocessed_dataloaders \
    .lidc_idri_preprocessed_dataloader \
    import LIDCIDRIPreprocessedKFoldDataLoader
from src.modules.data.dataloader.preprocessed_dataloaders \
    .luna25_preprocessed_dataloader \
    import LUNA25PreprocessedKFoldDataLoader
from src.modules.data.dataloader.preprocessed_dataloaders \
    .nlst_preprocessed_dataloader \
    import NLSTPreprocessedKFoldDataLoader



class PreprocessedDataLoader:
    def __new__(cls, config, lung_nodule_metadataframe):
        if config.dataset_name == "LIDC-IDRI":
            return LIDCIDRIPreprocessedKFoldDataLoader(
                config,
                lung_nodule_metadataframe
            )
        elif config.dataset_name == "LUNA25":
            return LUNA25PreprocessedKFoldDataLoader(
                config,
                lung_nodule_metadataframe
            )
        elif config.dataset_name == "NLST" or config.dataset_name == "NLSTLocal":
            return NLSTPreprocessedKFoldDataLoader(
                config,
                lung_nodule_metadataframe
            )
        else:
            raise ValueError(
                f"Invalid dataset name: {config.dataset_name}. "
                f"Supported dataset names are 'LIDC-IDRI' or 'LUNA25'."
            )