import os
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import ConcatDataset, AishellKWSDataset, MLSKWSDataset, AishellHotwordDataset, ACL6060KeywordDataset
from .data_collator import KWSDataCollator, HotwordDataCollator
from .sampler import AishellKWSSampler, MLSKWSSampler
from transformers import WhisperFeatureExtractor
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DatasetInfo:
    name: str
    root: str
    kw_type: str


class KWSDataMod(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        sampling: str,
        num_workers: int,
        train_info: List[DatasetInfo] ,
        val_info: List[DatasetInfo],
        test_info: DatasetInfo,
        hotwords_per_group: int,
        features_size: Tuple[int, int] = None,
        test_split: str = 'test',
        whisper_ckpt: str = 'openai/whisper-large-v2',
        max_duration: Optional[float] = None,
        resample_every_epoch: bool = True,
        **kwargs
    ):
        super().__init__()

        # features preprocessing parameters
        self.features_size = features_size
        
        # training data parameters    
        self.batch_size = batch_size
        self.whisper_ckpt = whisper_ckpt
        self.sampling = sampling
        self.num_workers = num_workers
        self.resample_every_epoch = resample_every_epoch

        # evaluation parameters
        self.hotwords_per_group = hotwords_per_group
        self.max_duration = max_duration

        # dataset information
        self.train_info = train_info  
        self.val_info = val_info
        self.test_info = test_info
        self.test_split = test_split        

        # adapt batch size given the data loading method
        if self.sampling == 'utterance-examples':            
            assert self.batch_size % 4 == 0, f'when loading all the positive and negative examples in the same batch, the batch size must be a multiple of 4, got {self.batch:size}'
            if self.train_info[0].name == 'aishell':
                self.batch_size = int(self.batch_size / 4)
        elif self.sampling != 'random':
            raise NotImplementedError(f'the required sampling method is not implemented, got {self.sampling}')

        # check parameters and file structure of training data
        assert not set([ds.name for ds in self.train_info]) - set(['aishell', 'mls']), f'at least one of the training datasets you asked for is not supported' 
        assert all(os.path.isdir(ds.root) for ds in self.train_info), f'at least one of the training dataset directories could not be found'
        if len(self.train_info) > 1:
            raise NotImplementedError(f'training with more than one dataset is not supported yet')
        # and validation data
        assert not set([ds.name for ds in self.val_info]) - set(['aishell', 'acl']), f'at least one of the validation datasets you asked for is not supported' 
        assert all(os.path.isdir(ds.root) for ds in self.val_info), f'at least one of the validation dataset directories could not be found'
        # and testing data
        assert self.test_info.name in ['aishell', 'acl'], f'at least one of the test datasets you asked for is not supported' 
        assert os.path.isdir(self.test_info.root), f'at least one of the test dataset directories could not be found'

        # instantiate a KWSDataCollator object
        self.collate_fn1 = KWSDataCollator(size=features_size)
        # instantiate a HotwordDataCollator object
        self.collate_fn2 = HotwordDataCollator()

    def setup(self, stage=None):

        # Assign val dataset for use in dataloader
        if stage == "validate" or stage is None:
            self.val_dataset = {}
            for ds in self.val_info:
                if ds.name == 'aishell':
                    self.val_dataset[ds.name + '/' + ds.kw_type] = AishellHotwordDataset(
                        root = os.path.join(ds.root, 'hotword'),
                        split = 'dev',
                        size = self.features_size,
                        r1_only = False,
                        hotwords_per_group = self.hotwords_per_group,
                        kw_type = ds.kw_type
                    )   
                elif ds.name == 'acl':
                    self.val_dataset[ds.name + '/' + ds.kw_type] = ACL6060KeywordDataset(
                        root = ds.root,
                        split = 'dev',
                        size = self.features_size,
                        keywords_per_group = self.hotwords_per_group,
                        kw_type = ds.kw_type
                    )

        # Assign train and val datasets for use in dataloader
        if stage == "fit" or stage is None:
            if self.train_info[0].name == 'aishell':
                if self.train_info[0].kw_type != 'all':
                    self.fit_dataset = AishellKWSDataset(
                        root = self.train_info[0].root,
                        kw_type = self.train_info[0].kw_type
                    )                    
                    self.sampler = AishellKWSSampler(
                        data_source = self.fit_dataset,
                        sampling = self.sampling,
                        negative_examples = {'random': 1, 'lexicographic': 2},
                        resample_every_epoch = self.resample_every_epoch
                    )
                else:
                    self.fit_dataset = ConcatDataset([
                        AishellKWSDataset(
                            root = self.train_info[0].root,
                            kw_type = kw_type
                        )
                    for kw_type in ['tts', 'natural']])
                    self.sampler = AishellKWSSampler(
                        data_source = self.fit_dataset.datasets[0],
                        sampling = self.sampling,
                        negative_examples = {'random': 1, 'lexicographic': 2},
                        resample_every_epoch = self.resample_every_epoch
                    )
            elif self.train_info[0].name == 'mls':
                if self.train_info[0].kw_type != 'all':
                    self.fit_dataset = MLSKWSDataset(
                        root = self.train_info[0].root,
                        languages = ['English', 'German', 'French', 'Spanish', 'Polish', 'Portuguese'],
                        kw_type = self.train_info[0].kw_type
                    )
                    self.sampler = MLSKWSSampler(
                        data_source = self.fit_dataset,
                        sampling = self.sampling,
                        negative_examples = {'random': 1, 'lexicographic': 2},
                        resample_every_epoch = self.resample_every_epoch
                    )
                else:
                    self.fit_dataset = ConcatDataset([
                        MLSKWSDataset(
                            root = self.train_info[0].root,
                            languages = ['English', 'German', 'French', 'Spanish', 'Polish', 'Portuguese'],
                            kw_type = kw_type
                        )
                    for kw_type in ['tts', 'natural']])
                    self.sampler = MLSKWSSampler(
                        data_source = self.fit_dataset.datasets[0],
                        sampling = self.sampling,
                        negative_examples = {'random': 1, 'lexicographic': 2},
                        resample_every_epoch = self.resample_every_epoch
                    )
            self.val_dataset = {}
            for ds in self.val_info:
                if ds.name == 'aishell':
                    self.val_dataset[ds.name + '/' + ds.kw_type] = AishellHotwordDataset(
                        root = os.path.join(ds.root, 'hotword'),
                        split = 'dev',
                        size = self.features_size,
                        r1_only = False,
                        hotwords_per_group = self.hotwords_per_group,
                        kw_type = ds.kw_type
                    )   
                elif ds.name == 'acl':
                    self.val_dataset[ds.name + '/' + ds.kw_type] = ACL6060KeywordDataset(
                        root = ds.root,
                        split = 'dev',
                        size = self.features_size,
                        keywords_per_group = self.hotwords_per_group,
                        kw_type = ds.kw_type
                    )
        
        # Assign test dataset for use in dataloader
        if stage == "test" or stage is None:
            if self.test_info.name == 'aishell':
                self.test_dataset = AishellHotwordDataset(
                    root = os.path.join(self.test_info.root, 'hotword'),
                    split = self.test_split,
                    size = self.features_size,
                    r1_only = False,
                    hotwords_per_group = self.hotwords_per_group,
                    kw_type = self.test_info.kw_type,
                    load_audio = True,
                    wav_folder = os.path.join(self.test_info.root, 'wav'),
                    feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisper_ckpt)
                )
            elif self.test_info.name == 'acl':
                self.test_dataset = ACL6060KeywordDataset(
                    root = self.test_info.root,
                    split = self.test_split,
                    size = self.features_size,
                    keywords_per_group = self.hotwords_per_group,
                    kw_type = self.test_info.kw_type,
                    load_audio = True,
                    feature_extractor = WhisperFeatureExtractor.from_pretrained(self.whisper_ckpt)
                )

    def train_dataloader(self): 
        if self.train_info[0].name == 'aishell':
            return DataLoader(self.fit_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn1, shuffle=False, sampler = self.sampler, num_workers = self.num_workers, persistent_workers = True if self.num_workers != 0 else False)
        elif self.train_info[0].name == 'mls':
            return DataLoader(self.fit_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn1, shuffle=False, sampler = self.sampler, num_workers = self.num_workers, persistent_workers = True if self.num_workers != 0 else False)
        
    def val_dataloader(self):
        # create one dataloader for each validation dataset
        dataloaders = [ 
            DataLoader(dataset, batch_size=1, collate_fn=self.collate_fn2, num_workers = self.num_workers, persistent_workers = True if self.num_workers != 0 else False)
        for ds, dataset in self.val_dataset.items()]
        
        return dataloaders

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, collate_fn=self.collate_fn2, num_workers = self.num_workers, persistent_workers = True if self.num_workers != 0 else False)