from src.modules.data.metadataframe.preprocessed_metadataframes \
    .lidc_idri_metadataframe import LIDCIDRIPreprocessedMetaDataFrame
from src.modules.data.metadataframe.preprocessed_metadataframes \
    .luna25_metadataframe import LUNA25PreprocessedMetaDataFrame
from src.modules.data.metadataframe.preprocessed_metadataframes \
    .nlst_metadataframe import NLSTPreprocessedMetaDataFrame
from src.modules.data.metadataframe.preprocessed_metadataframes \
    .nlstlocal_metadataframe import NLSTLocalPreprocessedMetaDataFrame

class MetadataFrame:
    def __new__(cls, config, experiment_execution_paths):
        if config.dataset_name == "LIDC-IDRI":
            return LIDCIDRIPreprocessedMetaDataFrame(
                config,
                experiment_execution_paths
            )
        elif config.dataset_name == "LUNA25":
            return LUNA25PreprocessedMetaDataFrame(
                config,
                experiment_execution_paths
            )
        elif config.dataset_name == "NLST":
            return NLSTPreprocessedMetaDataFrame(
                config,
                experiment_execution_paths
            )
        elif config.dataset_name == "NLSTLocal":
            return NLSTLocalPreprocessedMetaDataFrame(
                config,
                experiment_execution_paths
            )
        else:
            raise ValueError(
                f"Invalid dataset name: {config.dataset_name}. "
                f"Supported datasets are 'LUNA25', 'NLST', 'NLSTLocal."
            )