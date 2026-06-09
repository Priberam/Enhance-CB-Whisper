import os
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import (
    ConcatDataset,
    MLSKWSDataset,
    MLSEvaluationDataset,
    ACL6060KeywordDataset,
    AishellHotwordDataset,
)
from .data_collator import KWSDataCollator, HotwordDataCollator
from .sampler import MLSKWSSampler
from transformers import WhisperFeatureExtractor
from typing import List, Tuple, Optional
from dataclasses import dataclass


# TODO
# - create MLSEvaluationDataset test splits


@dataclass
class DatasetInfo:
    name: str
    root: str
    kw_type: str
    language: Optional[str] = None
    root_audios_transcripts: Optional[str] = ""


class KWSDataMod(LightningDataModule):

    def __init__(
        self,
        batch_size: int,
        sampling: str,
        num_workers: int,
        train_info: List[DatasetInfo],
        val_info: List[DatasetInfo],
        test_info: DatasetInfo,
        hotwords_per_group: int,
        features_size: Tuple[int, int] = None,
        test_split: str = "test",
        whisper_ckpt: str = "openai/whisper-large-v2",
        max_duration: Optional[float] = None,
        resample_every_epoch: bool = True,
        learn_features: bool = False,
        load_embeddings: bool = True,
        pad_long_before_resize: bool = False,
        kws_whisper_ckpt: Optional[str] = None,
        sru_ctx_window_kwd: int = 50,
        sru_ctx_window_utt: int = 250,
        n_layers: int = 1,
        relative_pos_embs: bool = False,
        condensed_dimension: str = "time",
        sru_chunk_size_utt: int = 0,
        sru_chunk_size_kwd: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.ctx_window_kwd = sru_ctx_window_kwd
        self.ctx_window_utt = sru_ctx_window_utt
        self.n_layers = n_layers
        self.relative_pos_embs = relative_pos_embs
        self.condensed_dimension = condensed_dimension
        self.sru_chunk_size_kwd = sru_chunk_size_kwd
        self.sru_chunk_size_utt = sru_chunk_size_utt

        # features preprocessing parameters
        self.features_size = features_size
        assert load_embeddings or (
            not load_embeddings and learn_features
        ), "when not loading pre-computed utterance embeddings, `learn_features` must be set to `True`"
        assert load_embeddings or (
            not load_embeddings and kws_whisper_ckpt != None
        ), "when not loading pre-computed utterance embeddings, `kws_whisper_ckpt` must be assigned"
        self.learn_features = learn_features
        self.load_embeddings = load_embeddings
        self.pad_long_before_resize = pad_long_before_resize
        self.kws_feature_extractor = (
            WhisperFeatureExtractor.from_pretrained(kws_whisper_ckpt)
            if not self.load_embeddings and kws_whisper_ckpt != None
            else None
        )
        # training data parameters
        self.batch_size = batch_size
        self.sampling = sampling
        self.num_workers = num_workers
        self.resample_every_epoch = resample_every_epoch

        # evaluation parameters
        self.whisper_ckpt = whisper_ckpt
        self.hotwords_per_group = hotwords_per_group
        self.max_duration = max_duration

        # dataset information
        self.train_info = train_info
        self.val_info = val_info
        self.test_info = test_info
        self.test_split = test_split

        # adapt batch size given the data loading method
        if self.sampling == "utterance-examples":
            assert (
                self.batch_size % 4 == 0
            ), f"when loading all the positive and negative examples in the same batch, the batch size must be a multiple of 4, got {self.batch:size}"
            if self.train_info[0].name == "aishell":
                self.batch_size = int(self.batch_size / 4)
        elif self.sampling != "random":
            raise NotImplementedError(
                f"the required sampling method is not implemented, got {self.sampling}"
            )

        # check parameters and file structure of training data
        assert not set([ds.name for ds in self.train_info]) - set(
            ["aishell", "mls"]
        ), "at least one of the training datasets you asked for is not supported"
        assert all(
            os.path.isdir(ds.root) for ds in self.train_info
        ), "at least one of the training dataset directories could not be found"
        if len(self.train_info) > 1:
            raise NotImplementedError(
                "training with more than one dataset is not supported yet"
            )
        # and validation data
        assert not set([ds.name for ds in self.val_info]) - set(
            ["mls", "mls-expanded", "aishell", "acl"]
        ), "at least one of the validation datasets you asked for is not supported"
        assert all(
            os.path.isdir(ds.root) for ds in self.val_info
        ), "at least one of the validation dataset directories could not be found"
        assert [
            ds.name != "mls"
            or (
                ds.language != None
                and ds.language
                in [
                    "english",
                    "french",
                    "german",
                    "polish",
                    "portuguese",
                    "spanish",
                ]
            )
            for ds in self.val_info
        ], "the requested MLS evaluation datasets miss a valid language"

        # instantiate a KWSDataCollator object
        self.collate_fn1 = KWSDataCollator(
            size=features_size,
            learn_features=self.learn_features,
            load_embeddings=self.load_embeddings,
        )
        # instantiate a HotwordDataCollator object
        self.collate_fn2 = HotwordDataCollator()

    def setup(self, stage=None):

        # Assign val dataset for use in dataloader
        if stage == "validate" or stage is None:
            self.val_dataset = {}
            for ds in self.val_info:
                if ds.name in ("mls", "mls-expanded"):
                    self.val_dataset[
                        ds.name + "/" + ds.language.lower() + "/" + ds.kw_type
                    ] = MLSEvaluationDataset(
                        root=os.path.join(ds.root),
                        language=ds.language.lower(),
                        split="dev",
                        size=self.features_size,
                        keywords_per_group=self.hotwords_per_group,
                        kw_type=ds.kw_type,
                        learn_features=self.learn_features,
                        load_embeddings=self.load_embeddings,
                        pad_long_before_resize=self.pad_long_before_resize,
                        kws_feature_extractor=(
                            self.kws_feature_extractor
                            if not self.load_embeddings
                            else None
                        ),
                        ctx_window_kwd=self.ctx_window_kwd,
                        ctx_window_utt=self.ctx_window_utt,
                        n_layers=self.n_layers,
                        relative_pos_embs=self.relative_pos_embs,
                        condensed_dimension=self.condensed_dimension,
                        sru_chunk_size_kwd=self.sru_chunk_size_kwd,
                        sru_chunk_size_utt=self.sru_chunk_size_utt,
                        root_audios_transcripts=ds.root_audios_transcripts,
                    )
                elif ds.name == "aishell":
                    self.val_dataset[ds.name + "/" + ds.kw_type] = (
                        AishellHotwordDataset(
                            root=os.path.join(ds.root, "hotword"),
                            split="dev",
                            size=self.features_size,
                            r1_only=False,
                            hotwords_per_group=self.hotwords_per_group,
                            kw_type=ds.kw_type,
                            learn_features=self.learn_features,
                        )
                    )
                elif ds.name == "acl":
                    self.val_dataset[ds.name + "/" + ds.kw_type] = (
                        ACL6060KeywordDataset(
                            root=ds.root,
                            split="dev",
                            size=self.features_size,
                            keywords_per_group=self.hotwords_per_group,
                            kw_type=ds.kw_type,
                            learn_features=self.learn_features,
                        )
                    )

        # Assign train and val datasets for use in dataloader
        if stage == "fit" or stage is None:
            if self.train_info[0].name == "mls":
                if self.train_info[0].kw_type != "all":
                    self.fit_dataset = MLSKWSDataset(
                        root=self.train_info[0].root,
                        languages=[
                            "English",
                            "German",
                            "French",
                            "Spanish",
                            "Polish",
                            "Portuguese",
                        ],
                        kw_type=self.train_info[0].kw_type,
                        size=self.features_size,
                        learn_features=self.learn_features,
                        load_embeddings=self.load_embeddings,
                        pad_long_before_resize=self.pad_long_before_resize,
                        feature_extractor=(
                            self.kws_feature_extractor
                            if not self.load_embeddings
                            else None
                        ),
                        ctx_window_kwd=self.ctx_window_kwd,
                        ctx_window_utt=self.ctx_window_utt,
                        n_layers=self.n_layers,
                        relative_pos_embs=self.relative_pos_embs,
                        condensed_dimension=self.condensed_dimension,
                        sru_chunk_size_kwd=self.sru_chunk_size_kwd,
                        sru_chunk_size_utt=self.sru_chunk_size_utt,
                    )
                    self.sampler = MLSKWSSampler(
                        data_source=self.fit_dataset,
                        sampling=self.sampling,
                        negative_examples={"random": 1, "lexicographic": 2},
                        resample_every_epoch=self.resample_every_epoch,
                    )
                else:
                    self.fit_dataset = ConcatDataset(
                        [
                            MLSKWSDataset(
                                root=self.train_info[0].root,
                                languages=[
                                    "English",
                                    "German",
                                    "French",
                                    "Spanish",
                                    "Polish",
                                    "Portuguese",
                                ],
                                kw_type=kw_type,
                                size=self.features_size,
                                learn_features=self.learn_features,
                                load_embeddings=self.load_embeddings,
                                pad_long_before_resize=self.pad_long_before_resize,
                                feature_extractor=(
                                    self.kws_feature_extractor
                                    if not self.load_embeddings
                                    else None
                                ),
                                ctx_window_kwd=self.ctx_window_kwd,
                                ctx_window_utt=self.ctx_window_utt,
                                n_layers=self.n_layers,
                                relative_pos_embs=self.relative_pos_embs,
                                condensed_dimension=self.condensed_dimension,
                                sru_chunk_size_kwd=self.sru_chunk_size_kwd,
                                sru_chunk_size_utt=self.sru_chunk_size_utt,
                            )
                            for kw_type in ["tts", "natural"]
                        ]
                    )
                    self.sampler = MLSKWSSampler(
                        data_source=self.fit_dataset.datasets[0],
                        sampling=self.sampling,
                        negative_examples={"random": 1, "lexicographic": 2},
                        resample_every_epoch=self.resample_every_epoch,
                    )
            self.val_dataset = {}
            for ds in self.val_info:
                if ds.name in ("mls", "mls-expanded"):
                    self.val_dataset[
                        ds.name + "/" + ds.language.lower() + "/" + ds.kw_type
                    ] = MLSEvaluationDataset(
                        root=os.path.join(ds.root),
                        language=ds.language.lower(),
                        split="dev",
                        size=self.features_size,
                        keywords_per_group=self.hotwords_per_group,
                        kw_type=ds.kw_type,
                        learn_features=self.learn_features,
                        load_embeddings=self.load_embeddings,
                        pad_long_before_resize=self.pad_long_before_resize,
                        kws_feature_extractor=(
                            self.kws_feature_extractor
                            if not self.load_embeddings
                            else None
                        ),
                        ctx_window_kwd=self.ctx_window_kwd,
                        ctx_window_utt=self.ctx_window_utt,
                        n_layers=self.n_layers,
                        relative_pos_embs=self.relative_pos_embs,
                        condensed_dimension=self.condensed_dimension,
                        sru_chunk_size_kwd=self.sru_chunk_size_kwd,
                        sru_chunk_size_utt=self.sru_chunk_size_utt,
                        root_audios_transcripts=ds.root_audios_transcripts,
                    )

        if stage == "test" or stage is None:
            if self.test_info.name == "aishell":
                self.test_dataset = AishellHotwordDataset(
                    root=os.path.join(self.test_info.root, "hotword"),
                    split=self.test_split,
                    size=self.features_size,
                    r1_only=False,
                    hotwords_per_group=self.hotwords_per_group,
                    kw_type=self.test_info.kw_type,
                    load_audio=True,
                    wav_folder=os.path.join(self.test_info.root, "wav"),
                    feature_extractor=WhisperFeatureExtractor.from_pretrained(
                        self.whisper_ckpt
                    ),
                    learn_features=self.learn_features,
                )
            elif self.test_info.name == "acl":
                self.test_dataset = ACL6060KeywordDataset(
                    root=self.test_info.root,
                    split=self.test_split,
                    size=self.features_size,
                    keywords_per_group=self.hotwords_per_group,
                    kw_type=self.test_info.kw_type,
                    load_audio=True,
                    feature_extractor=WhisperFeatureExtractor.from_pretrained(
                        self.whisper_ckpt
                    ),
                    learn_features=self.learn_features,
                )

    def train_dataloader(self):
        return DataLoader(
            self.fit_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn1,
            shuffle=False,
            sampler=self.sampler,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 0 else False,
        )

    def val_dataloader(self):
        # create one dataloader for each validation dataset
        dataloaders = [
            DataLoader(
                dataset,
                batch_size=1,
                collate_fn=self.collate_fn2,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers != 0 else False,
            )
            for ds, dataset in self.val_dataset.items()
        ]

        return dataloaders

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=self.collate_fn2,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers != 0 else False,
        )
