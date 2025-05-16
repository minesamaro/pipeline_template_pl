import pandas


class NLSTLocalPreprocessedMetaDataFrame:
    def __init__(self, config, experiment_execution_paths):
        self.config = config

        self.lung_metadataframe = pandas.read_csv(
            'C:\\Users\\HP\\OneDrive - Universidade do Porto\\Documentos\\UNIVERSIDADE\\Tese\\PhantomDataset\\cancer_info.csv'
        )
        self.dimension = int(self.config.dimension)
        # Apply transformations to the metadataframe
        self._apply_label_metadataframe_transformations() # Changes according to the wanted label
        self._apply_truncation_metadataframe_transformations() # Truncation of the metadataframe
        self._save_metadataframe_as_csv(
            experiment_execution_paths=experiment_execution_paths
        )

    def get_lung_metadataframe(self):
        return self.lung_metadataframe
        
    def _apply_label_metadataframe_transformations(self):
        if self.config.label not in self.lung_metadataframe.columns:
            raise ValueError(
                f"Label column '{self.config.label}' not found in metadata. "
                f"Available columns: {list(self.lung_metadataframe.columns)}"
            )
        print(f"Label column '{self.config.label}' found in metadata. "
              f"Available columns: {list(self.lung_metadataframe.columns)}")
        # Cleanly create or override the 'label' column used downstream
        self.lung_metadataframe = self.lung_metadataframe.copy()
        self.lung_metadataframe['label'] = self.lung_metadataframe[self.config.label]

    def _apply_truncation_metadataframe_transformations(self):
        # If config.resample is True, sct_slice_num = sct_slice_num_rs
        # If config.resample is False, sct_slice_num = sct_slice_num_og

        if self.config.dimension == 2 or self.config.dimension == 2.5:
            self.lung_metadataframe = self.lung_metadataframe[self.lung_metadataframe['sct_nod_err'] != 1 ]

        if self.config.resample:
            self.lung_metadataframe['sct_slice_num'] = \
                self.lung_metadataframe['sct_slice_num_rs']
        else:
            self.lung_metadataframe['sct_slice_num'] = \
                self.lung_metadataframe['sct_slice_num_og']
            
        self.lung_metadataframe['key'] = \
            self.lung_metadataframe['pid'].astype(str) + "_" + \
            self.lung_metadataframe['study_yr'].astype(str)
        
        self.lung_metadataframe = \
            self.lung_metadataframe[['pid', 'study_yr', 'path', 'key', 'label', 'sct_slice_num']]
    
    def _save_metadataframe_as_csv(self, experiment_execution_paths):
        self.lung_metadataframe.to_csv(
            "C:\\Users\\HP\\OneDrive - Universidade do Porto\\Documentos\\UNIVERSIDADE\\Tese\\PhantomDataset\\experiment_metadataframe.csv",
            index=False
        )