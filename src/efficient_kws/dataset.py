import os
import re
import torch
import torchvision
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm
import math
import torchaudio
from transformers import WhisperFeatureExtractor
from typing import Optional, List, Tuple, Union
import xml.etree.ElementTree as ET
from itertools import groupby, accumulate
from whisper.audio import SAMPLE_RATE, N_SAMPLES


###############################################################################################################################
# Disclaimer: instantiating a HotwordDataset object assumes the hidden states from the utterances and hotwords are normalized #
###############################################################################################################################


# TODO
# - develop a build_dataset function
# - adapt Aishell-KWS, AishellHotword and UrduSELMA to deal with learn_features
# - adapt Aishell-KWS, AishellHotword and UrduSELMA to deal with load_embeddings
# - add pad_utterance option to Aishell-KWS, AishellHotword, and UrduSELMA


LONG_MAX_LENGTH = 1500


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d.__getitem__(i) for d in self.datasets)

    def __len__(self):
        return min(d.__len__() for d in self.datasets)


class KWSDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def process_keyword(self, kwd):
        kwd = kwd[-self.n_layers :, :, :]
        # mask and pad
        if self.ctx_window_kwd - kwd.size(dim=1) >= 0:
            kwd_mask = torch.cat(
                (
                    torch.ones(kwd.size(dim=0), kwd.size(dim=1)),
                    torch.zeros(
                        kwd.size(dim=0), self.ctx_window_kwd - kwd.size(dim=1)
                    ),
                ),
                dim=1,
            )
            kwd = torch.cat(
                (
                    kwd,
                    torch.zeros(
                        kwd.size(dim=0),
                        self.ctx_window_kwd - kwd.size(dim=1),
                        kwd.size(dim=2),
                    ).type_as(kwd),
                ),
                dim=1,
            )
        else:
            kwd = kwd[:, : self.ctx_window_kwd, :]
            kwd_mask = torch.ones(kwd.size(dim=0), kwd.size(dim=1))

        n_channels, n_frames_kwd, emb_dim = kwd.shape

        # n_channels , self.ctx_window_kwd , n_chunks , sru_chunk_size_kwd
        kwd_strided = kwd.unfold(
            dimension=1 if self.condensed_dimension == "time" else 2,
            size=self.sru_chunk_size_kwd,
            step=self.sru_chunk_size_kwd,
        )

        # n_channels , n_chunks , sru_chunk_size_kwd , emb_dim
        if self.condensed_dimension == "time":
            kwd_strided = kwd_strided.permute(0, 1, 3, 2)
        # n_channels, n_chunks, sru_chunk_size_kwd, self.ctx_window_kwd
        else:
            kwd_strided = kwd_strided.permute(0, 2, 3, 1)

        # kwd_strided = kwd.unsqueeze(1).permute(0, 1, 3, 2)

        n_channels, n_chunks, condensed_dim, preserved_dim = kwd_strided.shape

        if self.condensed_dimension == "time":
            kwd_mask_strided = kwd_mask.unfold(
                dimension=1,
                size=self.sru_chunk_size_kwd,
                step=self.sru_chunk_size_kwd,
            ).permute(0, 1, 2)
        else:
            kwd_mask_strided = kwd_mask.unsqueeze(1)  # .repeat(1, n_chunks, 1)

        # positional embeddings
        # [n_chunks, condensed_dim]
        kwd_position_strided = (
            torch.arange(
                start=0,
                end=condensed_dim,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        kwd_position_strided = kwd_position_strided.repeat(
            n_channels, n_chunks, 1
        )

        item = {
            "kwd_strided": kwd_strided,
            "kwd_mask_strided": kwd_mask_strided,
            "kwd_position_strided": kwd_position_strided,
            # "kwd": kwd,
        }

        return item

    def process_utterance(self, utt):
        utt = utt[-self.n_layers :, :, :]

        # mask and pad
        if self.ctx_window_utt - utt.size(dim=1) >= 0:
            utt_mask = torch.cat(
                (
                    torch.ones(utt.size(dim=0), utt.size(dim=1)),
                    torch.zeros(
                        utt.size(dim=0), self.ctx_window_utt - utt.size(dim=1)
                    ),
                ),
                dim=1,
            )
            utt = torch.cat(
                (
                    utt,
                    torch.zeros(
                        utt.size(dim=0),
                        self.ctx_window_utt - utt.size(dim=1),
                        utt.size(dim=2),
                    ).type_as(utt),
                ),
                dim=1,
            )
        else:
            utt = utt[:, : self.ctx_window_utt, :]
            utt_mask = torch.ones(utt.size(dim=0), utt.size(dim=1))

        n_channels, n_frames_utt, emb_dim = utt.shape

        # n_channels , n_chunks , self.ctx_window_kwd , emb_dim
        utt_strided = utt.unfold(
            dimension=1 if self.condensed_dimension == "time" else 2,
            size=self.sru_chunk_size_utt,
            step=self.sru_chunk_size_utt,
        )
        if self.condensed_dimension == "time":
            utt_strided = utt_strided.permute(0, 1, 3, 2)
        else:
            utt_strided = utt_strided.permute(0, 2, 3, 1)

        # utt_strided = utt.unsqueeze(1).permute(0, 1, 3, 2)

        n_channels, n_chunks, condensed_dim, preserved_dim = utt_strided.shape

        if self.condensed_dimension == "time":
            utt_mask_strided = utt_mask.unfold(
                dimension=1,
                size=self.sru_chunk_size_utt,
                step=self.sru_chunk_size_utt,
            ).permute(0, 1, 2)
        else:
            utt_mask_strided = utt_mask.unsqueeze(1)  # .repeat(1, n_chunks, 1)

        # positional embeddings
        # [n_chunks, n_frames_utt]
        utt_position_strided = (
            torch.arange(
                start=0,
                end=(condensed_dim),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(n_channels, n_chunks, 1)

        item = {
            "utt_strided": utt_strided,
            "utt_mask_strided": utt_mask_strided,
            "utt_position_strided": utt_position_strided,
            # "utt": utt,
        }

        return item


class MLSKWSDataset(KWSDataset):

    def __init__(
        self,
        root: str,
        languages: List[str] = [
            "English",
            "German",
            "French",
            "Spanish",
            "Polish",
            "Portuguese",
        ],
        kw_type: str = "natural",
        size: Optional[Tuple[int, int]] = None,
        learn_features: bool = False,
        load_embeddings: bool = True,
        pad_long_before_resize: bool = False,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        ctx_window_kwd: int = 50,
        ctx_window_utt: int = 250,
        n_layers: int = 1,
        relative_pos_embs: bool = False,
        condensed_dimension: str = "time",
        sru_chunk_size_utt: int = 0,
        sru_chunk_size_kwd: int = 0,
    ):

        self.ctx_window_kwd = ctx_window_kwd
        self.ctx_window_utt = ctx_window_utt
        self.relative_pos_embs = relative_pos_embs
        self.n_layers = n_layers
        self.condensed_dimension = condensed_dimension
        self.sru_chunk_size_kwd = sru_chunk_size_kwd
        self.sru_chunk_size_utt = sru_chunk_size_utt
        self.features_size = size

        # check if the provided directories exist
        assert os.path.isdir(
            root
        ), "the directory you indicated with the dataset could not be found"
        assert all(
            [
                os.path.isdir(
                    os.path.join(root, "mls_" + language.lower() + "_opus")
                )
                for language in languages
            ]
        ), "at least one of the MLS subdatasets you asked could not be found"
        self.roots = {
            language: os.path.join(
                root, "mls_" + language.lower() + "_opus", "train"
            )
            for language in languages
        }
        self.languages = sorted(languages)

        # check if the keyword type is valid
        assert kw_type in [
            "tts",
            "natural",
        ], f"the provided keyword type is not valid, got {kw_type}"
        self.kw_type = kw_type

        assert load_embeddings or (
            not load_embeddings and learn_features
        ), "when not loading the utterance embeddings, feature must be necessarily learnt"
        assert load_embeddings or (
            not load_embeddings and feature_extractor != None
        ), "when not loading the utterance embeddings, a feature extractor must be provided"
        # whether or not to load the utterance embeddings
        self.load_embeddings = load_embeddings
        self.feature_extractor = feature_extractor
        # whether or not to pad the long edge of similarity matrices before resizing
        self.pad_long_before_resize = pad_long_before_resize

        # check if the keywords files exist
        assert all(
            [
                os.path.exists(os.path.join(root, "keywords.txt"))
                for _, root in self.roots.items()
            ]
        ), "there is no keywords file in at least one of the MLS subdatasets"
        # and get keywords list in lexicographic order
        self.keywords = {}
        self.kw_zfill = {}
        self.ghost_keyword_indices = {}
        for language, root in self.roots.items():
            with open(os.path.join(root, "keywords.txt"), "r") as f:
                self.keywords[language] = {
                    line.split()[0].strip(): idx
                    for idx, line in enumerate(f.readlines())
                }
            self.kw_zfill[language] = len(
                str(len(self.keywords[language]) - 1)
            )
            self.ghost_keyword_indices[language] = [
                idx
                for idx, _ in enumerate(self.keywords[language])
                if not os.path.exists(
                    os.path.join(
                        self.roots[language],
                        "keywords-hs",
                        self.kw_type,
                        str(idx).zfill(self.kw_zfill[language]) + ".bin",
                    )
                )
            ]
        # and also in reverse lexicographic order
        self.keywords_reverse = {
            language: sorted(keywords.keys(), key=lambda x: x[::-1])
            for language, keywords in self.keywords.items()
        }
        # accumulated number of keywords
        self.n_keywords = list(
            accumulate([len(k_) for k_ in self.keywords.values()])
        )
        # and get relevant sizes from a particular example
        rel_language = self.languages[0]
        rel_idx = list(
            set(range(len(self.keywords[rel_language])))
            - set(self.ghost_keyword_indices[self.languages[0]])
        )[0]
        with open(
            os.path.join(
                self.roots[rel_language],
                "keywords-hs",
                self.kw_type,
                str(rel_idx).zfill(self.kw_zfill[rel_language]) + ".bin",
            ),
            "rb",
        ) as f:
            rel_kwd = torch.load(f, map_location=torch.device("cpu")).detach()
        self.n_channels = rel_kwd.shape[0]
        self.hidden_dim = rel_kwd.shape[2]

        # check if the positive examples files exist
        assert all(
            [
                os.path.exists(os.path.join(root, "positives.tsv"))
                for _, root in self.roots.items()
            ]
        ), "there is no positive examples file in at least one of the MLS subdatasets"
        # and read TSV files with metadata
        self.metadata = []
        offset_idx = 0
        for language in self.languages:
            with open(
                os.path.join(self.roots[language], "positives.tsv"), "r"
            ) as f:
                data = [
                    [i_.strip() for i_ in line.split("\t")]
                    for line in f.readlines()
                ]
            self.metadata.append(
                {
                    "language": language,
                    "offset_idx": offset_idx,
                    "data": [
                        {
                            "code": item[0],
                            "f_utterance": os.path.join(
                                self.roots[language],
                                "audio",
                                (
                                    m_ := re.match(
                                        "(?P<f1>\d+)_(?P<f2>\d+)_\d+", item[0]
                                    )
                                ).group("f1"),
                                m_.group("f2"),
                                item[0] + ".opus",
                            ),
                            "positives": [
                                (
                                    item[i_],
                                    int(item[i_ + 1]),
                                    int(item[i_ + 2]),
                                )
                                for i_ in range(1, len(item), 3)
                            ],
                        }
                        for item in data
                    ],
                }
            )
            offset_idx += len(data) * self.n_keywords[-1]
        # dataset size
        self.size = offset_idx

        # whether or not to learn the features
        self.learn_features = learn_features

        self.frames = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # find subdataset in metadata that corresponds to the given idx
        submetadata = self.metadata[
            (
                [idx >= d_["offset_idx"] for d_ in self.metadata].index(False)
                - 1
                if not all([idx >= d_["offset_idx"] for d_ in self.metadata])
                else -1
            )
        ]
        # find the utterance data that corresponds to the given idx
        data = submetadata["data"][
            (idx - submetadata["offset_idx"]) // self.n_keywords[-1]
        ]
        # the selected keyword index
        keyword_idx = (idx - submetadata["offset_idx"]) % self.n_keywords[-1]
        keyword_language_idx = [
            keyword_idx < n_ for n_ in self.n_keywords
        ].index(True)
        keyword_idx = (
            keyword_idx - self.n_keywords[keyword_language_idx - 1]
            if keyword_language_idx != 0
            else keyword_idx
        )

        # item to be returned
        # add label, mask and domain attributes
        item = {
            "label": (
                1
                if any(
                    [
                        keyword_idx == positive_idx
                        for _, positive_idx, _ in data["positives"]
                    ]
                )
                and submetadata["language"]
                == self.languages[keyword_language_idx]
                else 0
            ),
            "mask": (
                1
                if keyword_idx
                not in self.ghost_keyword_indices[
                    self.languages[keyword_language_idx]
                ]
                else 0
            ),
            "domain": (0 if self.kw_type == "tts" else len(self.languages))
            + self.languages.index(submetadata["language"]),
        }
        # load utterance and keyword hidden states
        if self.load_embeddings:
            with open(
                os.path.join(
                    self.roots[submetadata["language"]],
                    "hs",
                    data["code"] + ".bin",
                ),
                "rb",
            ) as f:
                utt = torch.load(f, map_location=torch.device("cpu")).detach()
        else:
            # load utterance audio and preprocess it
            waveform, _ = torchaudio.load(data["f_utterance"])
            sample_rate = torchaudio.info(data["f_utterance"]).sample_rate
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(
                    torchaudio.functional.resample(
                        waveform, sample_rate, SAMPLE_RATE
                    ),
                    dim=0,
                    keepdim=True,
                )
            else:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, SAMPLE_RATE
                )
            # extract features
            utt = self.feature_extractor(
                waveform[0],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
            ).input_features

        if item["mask"] == 1:
            with open(
                os.path.join(
                    self.roots[self.languages[keyword_language_idx]],
                    "keywords-hs",
                    self.kw_type,
                    str(keyword_idx).zfill(
                        self.kw_zfill[self.languages[keyword_language_idx]]
                    )
                    + ".bin",
                ),
                "rb",
            ) as f:
                kwd = torch.load(f, map_location=torch.device("cpu")).detach()
        else:
            kwd = torch.zeros(self.n_channels, 1, self.hidden_dim).to("cpu")

        if (
            self.features_size[0] - kwd.size(dim=1) >= 0
        ) and self.pad_long_before_resize:
            kwd_mask = torch.cat(
                (
                    torch.ones(kwd.size(dim=0), kwd.size(dim=1)),
                    torch.zeros(
                        kwd.size(dim=0),
                        self.features_size[0] - kwd.size(dim=1),
                    ),
                ),
                dim=1,
            )

            self.frames += kwd.size(dim=1)
            kwd = torch.cat(
                (
                    kwd,
                    torch.zeros(
                        kwd.size(dim=0),
                        self.features_size[0] - kwd.size(dim=1),
                        kwd.size(dim=2),
                    ).type_as(kwd),
                ),
                dim=1,
            )
        else:
            kwd = kwd[:, : self.features_size[0], :]
            kwd_mask = torch.ones(kwd.size(dim=0), kwd.size(dim=1))
            self.frames += kwd.size(dim=1)

        if (
            self.features_size[1] - utt.size(dim=1) >= 0
        ) and self.pad_long_before_resize:
            utt_mask = torch.cat(
                (
                    torch.ones(utt.size(dim=0), utt.size(dim=1)),
                    torch.zeros(
                        utt.size(dim=0),
                        self.features_size[1] - utt.size(dim=1),
                    ),
                ),
                dim=1,
            )
            utt = torch.cat(
                (
                    utt,
                    torch.zeros(
                        utt.size(dim=0),
                        self.features_size[1] - utt.size(dim=1),
                        utt.size(dim=2),
                    ).type_as(utt),
                ),
                dim=1,
            )
        else:
            utt = utt[:, : self.features_size[1], :]
            utt_mask = torch.ones(utt.size(dim=0), utt.size(dim=1))

        utt = utt[-self.n_layers :, :, :]  # .unsqueeze(0)
        utt_mask = utt_mask[-self.n_layers :, :]  # .unsqueeze(0)
        kwd = kwd[-self.n_layers :, :, :]  # .unsqueeze(0)
        kwd_mask = kwd_mask[-self.n_layers :, :]  # .unsqueeze(0)

        utt_item = {"utt": utt, "utt_mask": utt_mask}
        kwd_item = {"kwd": kwd, "kwd_mask": kwd_mask}

        if False:
            # compute similarity matrices
            # simple inner product because vectors are normalized
            item.update(
                [
                    ("idx", idx),
                    ("utt_features", utt_item["utt"]),
                    ("kwd_features", kwd_item["kwd"]),
                    (
                        "features",
                        torch.matmul(
                            kwd_item["kwd"],
                            utt_item["utt"].transpose(1, 2),
                        ),
                    ),
                ]
            )
        else:
            item.update(
                [
                    ("idx", idx),
                    ("utt_features", utt_item["utt"]),
                    ("kwd_features", kwd_item["kwd"]),
                    ("utt_mask", utt_item["utt_mask"]),
                    ("kwd_mask", kwd_item["kwd_mask"]),
                ]
            )

        return item


class MLSEvaluationDataset(KWSDataset):

    def __init__(
        self,
        root: str,
        language: str,
        split: str,
        kw_type: str = "natural",
        size: Optional[Tuple[int, int]] = None,
        keywords_per_group: int = -1,
        load_audio: bool = False,
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        learn_features: bool = False,
        load_embeddings: bool = True,
        pad_long_before_resize: bool = False,
        kws_feature_extractor: Optional[WhisperFeatureExtractor] = None,
        ctx_window_kwd: int = 50,
        ctx_window_utt: int = 250,
        n_layers: int = 1,
        relative_pos_embs: bool = False,
        condensed_dimension: str = "time",
        sru_chunk_size_utt: int = 0,
        sru_chunk_size_kwd: int = 0,
        root_audios_transcripts: str = "",
    ):
        self.ctx_window_kwd = ctx_window_kwd
        self.ctx_window_utt = ctx_window_utt
        self.relative_pos_embs = relative_pos_embs
        self.n_layers = n_layers
        self.sru_chunk_size_kwd = sru_chunk_size_kwd
        self.sru_chunk_size_utt = sru_chunk_size_utt
        self.condensed_dimension = condensed_dimension
        self.root_audios_transcripts = root_audios_transcripts

        # size of features
        assert size is None or (
            len(size) == 2 and all([i_ >= 32 for i_ in size])
        ), "provide a valid size for the input features of the KWS model"
        self.size = size

        # load audio
        self.load_audio = load_audio
        if self.load_audio:
            raise NotImplementedError(
                "loading audio is not implemented in MLSEvaluationDataset"
            )

        # save instance of WhisperFeatureExtractor for Whisper
        self.feature_extractor = feature_extractor

        assert load_embeddings or (
            not load_embeddings and learn_features
        ), "when not loading the utterance embeddings, feature must be necessarily learnt"
        assert load_embeddings or (
            not load_embeddings and kws_feature_extractor != None
        ), "when not loading the utterance embeddings, a feature extractor must be provided"
        # whether or not to load the utterance embeddings
        self.load_embeddings = load_embeddings
        # whether or not to pad the long edge of similarity matrices before resizing
        self.pad_long_before_resize = pad_long_before_resize
        # save instance of WhisperFeatureExtractor for keyword-spotting
        self.kws_feature_extractor = kws_feature_extractor

        # check if the provided directories exist
        assert os.path.isdir(
            root
        ), "the directory you indicated with the dataset could not be found"
        assert os.path.isdir(
            os.path.join(root, "mls_" + language.lower() + "_opus")
        ), "the evaluation MLS subdataset you asked could not be found"
        assert split in ["dev"], f"the split is not supported, got: {split}"
        self.split_folder = os.path.join(
            root, "mls_" + language.lower() + "_opus", split
        )
        self.split_audios_folder = os.path.join(
            root_audios_transcripts, "mls_" + language.lower() + "_opus", split
        )
        self.languages = language
        self.split = split

        # check if the keyword type is valid
        assert kw_type in [
            "tts",
            "natural",
        ], f"the provided keyword type is not valid, got {kw_type}"
        self.kw_type = kw_type

        # get keywords and corresponding whisper features in groups
        with open(os.path.join(self.split_folder, "keywords.txt"), "r") as f:
            self.keywords = [line.strip() for line in f.readlines()]
        self.database = []
        kw_zfill = len(str(len(self.keywords) - 1))
        ghost_keyword_indices = []
        for idx, _ in enumerate(self.keywords):
            if os.path.exists(
                os.path.join(
                    self.split_folder,
                    "keywords-hs",
                    self.kw_type,
                    str(idx).zfill(kw_zfill) + ".bin",
                )
            ):
                with open(
                    os.path.join(
                        self.split_folder,
                        "keywords-hs",
                        self.kw_type,
                        str(idx).zfill(kw_zfill) + ".bin",
                    ),
                    "rb",
                ) as f:
                    self.database.append(
                        torch.load(
                            f, map_location=torch.device("cpu")
                        ).detach()
                    )
            else:
                self.database.append(None)
                ghost_keyword_indices.append(idx)
        # set ghost keywords features to zeros
        smaller_idx = min(
            [
                (idx, hs.shape)
                for idx, hs in enumerate(self.database)
                if hs != None
            ],
            key=lambda x: x[1][1],
        )[0]
        for idx in ghost_keyword_indices:
            self.database[idx] = torch.zeros_like(self.database[smaller_idx])
        # separate into groups
        # also adding a mask for the ghost keyword positions
        if keywords_per_group == -1:
            self.keywords_per_group = len(self.keywords)
        else:
            self.keywords_per_group = keywords_per_group
        self.database = [
            {
                "keywords": self.keywords[i : i + self.keywords_per_group],
                "hidden_states": self.database[
                    i : i + self.keywords_per_group
                ],
                "kwd": self.database[i : i + self.keywords_per_group],
                "kwd_mask": [
                    0
                    for _ in range(
                        i, min(i + self.keywords_per_group, len(self.keywords))
                    )
                ],
                "max_length": max(
                    max(
                        [
                            t_.size(dim=1)
                            for t_ in self.database[
                                i : i + self.keywords_per_group
                            ]
                        ]
                    ),
                    32,
                ),
                "mask": torch.tensor(
                    [
                        0 if idx in ghost_keyword_indices else 1
                        for idx in range(
                            i,
                            min(
                                i + self.keywords_per_group, len(self.keywords)
                            ),
                        )
                    ]
                ),
            }
            for i in range(0, len(self.keywords), self.keywords_per_group)
        ]

        for i, group in enumerate(self.database):
            for j, kwd in enumerate(group["kwd"]):
                if (
                    self.size[0] - kwd.size(dim=1) >= 0
                ) and self.pad_long_before_resize:
                    kwd_mask = torch.cat(
                        (
                            torch.ones(kwd.size(dim=0), kwd.size(dim=1)),
                            torch.zeros(
                                kwd.size(dim=0), self.size[0] - kwd.size(dim=1)
                            ),
                        ),
                        dim=1,
                    )

                    if self.pad_long_before_resize:
                        kwd = torch.cat(
                            (
                                kwd,
                                torch.zeros(
                                    kwd.size(dim=0),
                                    self.size[0] - kwd.size(dim=1),
                                    kwd.size(dim=2),
                                ).type_as(kwd),
                            ),
                            dim=1,
                        )
                else:
                    kwd = kwd[:, : self.size[0], :]
                    kwd_mask = torch.ones(kwd.size(dim=0), kwd.size(dim=1))

                kwd = kwd[-self.n_layers :, :, :]  # .unsqueeze(0)
                kwd_mask = kwd_mask[-self.n_layers :, :]  # .unsqueeze(0)

                self.database[i]["kwd"][j] = kwd
                self.database[i]["kwd_mask"][j] = kwd_mask

        # load transcripts
        if self.is_expanded():
            path = self.split_audios_folder
        else:
            path = self.split_folder

        with open(os.path.join(path, "uttid"), "r") as f:
            uttid = [line.strip() for line in f.readlines()]

        transcripts_path = os.path.join(path, "transcripts.txt")

        with open(transcripts_path, "r") as f:
            transcripts = {
                code: line.split("\t")[1].strip()
                for line in f.readlines()
                if (code := line.split("\t")[0].strip()) in uttid
            }
        # load keywords for each transcript
        with open(os.path.join(path, "positives.tsv"), "r") as f:
            keywords = {
                (l_ := line.split("\t"))[0].strip(): [
                    {
                        "mention": l_[i_].strip(),
                        "total_offset": int(l_[i_ + 1].strip()),
                        "end_offset": int(l_[i_ + 2].strip()),
                    }
                    for i_ in range(1, len(l_), 3)
                ]
                for line in f.readlines()
            }

        # # get speaker information
        # with open(os.path.join(self.split_folder, 'text/xml/ACL.6060.dev.en-xx.en.xml') if split == 'dev' else os.path.join(self.split_folder, 'text/xml/ACL.6060.eval.en-xx.en.xml'), 'r') as f:
        #     root = ET.fromstring(re.sub('&', '', f.read()))
        # idx2speaker = {int(child.attrib['id']) : speaker_id for speaker_id, doc in enumerate(root[0]) for child in doc if child.tag == 'seg'}

        # create dataset
        self.dataset = [
            {
                "transcript": transcript,
                "utterance": {
                    "audio": (
                        os.path.join(
                            path,
                            "audio",
                            (
                                m_ := re.match(
                                    "(?P<f1>\d+)_(?P<f2>\d+)_\d+", code
                                )
                            ).group("f1"),
                            m_.group("f2"),
                            code + ".opus",
                        )
                        if self.load_audio or not self.load_embeddings
                        else None
                    ),
                    "hidden_states": (
                        os.path.join(path, "hs", code + ".bin")
                        if self.load_embeddings
                        else None
                    ),
                },
                "hotword_labels": [
                    torch.tensor(
                        [
                            (
                                1
                                if keyword
                                in [kw_["mention"] for kw_ in keywords[code]]
                                else 0
                            )
                            for keyword in self.keywords[
                                i : i + self.keywords_per_group
                            ]
                        ]
                    )
                    for i in range(
                        0, len(self.keywords), self.keywords_per_group
                    )
                ],
                "keywords": keywords[code],
                # 'speaker': idx2speaker[idx + 1]
            }
            for code, transcript in transcripts.items()
        ]

        # whether or not to learn the features
        self.learn_features = learn_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # print("idx: ", idx)

        item = {"hotword_labels": self.dataset[idx]["hotword_labels"]}
        # add ghost keywords mask
        item.update([("hotword_mask", [g["mask"] for g in self.database])])

        if self.load_embeddings:
            # Load hidden_states from utterance
            with open(
                self.dataset[idx]["utterance"]["hidden_states"], "rb"
            ) as f:
                hidden_states = torch.load(
                    f, map_location=torch.device("cpu")
                ).detach()
        else:
            # load utterance audio and preprocess it
            waveform, _ = torchaudio.load(
                self.dataset[idx]["utterance"]["audio"]
            )
            sample_rate = torchaudio.info(
                self.dataset[idx]["utterance"]["audio"]
            ).sample_rate
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(
                    torchaudio.functional.resample(
                        waveform, sample_rate, SAMPLE_RATE
                    ),
                    dim=0,
                    keepdim=True,
                )
            else:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, SAMPLE_RATE
                )
            # extract features
            features = self.kws_feature_extractor(
                waveform[0],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
            ).input_features

        if False:
            if self.pad_long_before_resize:
                hidden_states = torch.cat(
                    (
                        hidden_states,
                        torch.zeros(
                            hidden_states.size(dim=0),
                            LONG_MAX_LENGTH - hidden_states.size(dim=1),
                            hidden_states.size(dim=2),
                        ).type_as(hidden_states),
                    ),
                    dim=1,
                )
            # compute similarity matrices
            # simple inner product because vectors are normalized
            item.update(
                [
                    (
                        "features",
                        [
                            [
                                torch.matmul(hs, hidden_states.transpose(1, 2))
                                for hs in group["hidden_states"]
                            ]
                            for group in self.database
                        ],
                    ),
                    (
                        "utt_features",
                        hidden_states if self.load_embeddings else features,
                    ),
                    (
                        "kwd_features",
                        [
                            [hs for hs in group["hidden_states"]]
                            for group in self.database
                        ],
                    ),
                ]
            )
            if self.size is not None:
                # resize both edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (self.size[0], self.size[1]),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
            else:
                # resize only the short edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (
                                                group["max_length"],
                                                hidden_states.size(dim=1),
                                            ),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
        else:
            item.update(
                [
                    (
                        "kwd",
                        [
                            [hs for hs in group["kwd"]]
                            for group in self.database
                        ],
                    ),
                ]
            )

        # load utterance audio
        if self.load_audio:
            # load utterance audio and preprocess it
            waveform, sample_rate = torchaudio.load(item["utterance"]["audio"])
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(
                    torchaudio.functional.resample(
                        waveform, sample_rate, SAMPLE_RATE
                    ),
                    dim=0,
                    keepdim=True,
                )
            else:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, SAMPLE_RATE
                )
            # whether or not the audio has duration smaller than 30 seconds
            is_shortform = waveform.shape[0] <= N_SAMPLES
            # extract features
            if is_shortform:
                output_features = self.feature_extractor(
                    waveform[0],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    truncation=True if is_shortform else False,
                    padding="max_length" if is_shortform else "longest",
                    return_attention_mask=True,
                )
            features = output_features.input_features
            attention_mask = output_features.attention_mask
            item["utterance"].update(
                [("features", features), ("attention_mask", attention_mask)]
            )

        if self.load_embeddings:
            if (
                self.size[1] - hidden_states.size(dim=1) >= 0
            ) and self.pad_long_before_resize:
                hidden_states_mask = torch.cat(
                    (
                        torch.ones(
                            hidden_states.size(dim=0),
                            hidden_states.size(dim=1),
                        ),
                        torch.zeros(
                            hidden_states.size(dim=0),
                            self.size[1] - hidden_states.size(dim=1),
                        ),
                    ),
                    dim=1,
                )
                if self.pad_long_before_resize:
                    hidden_states = torch.cat(
                        (
                            hidden_states,
                            torch.zeros(
                                hidden_states.size(dim=0),
                                self.size[1] - hidden_states.size(dim=1),
                                hidden_states.size(dim=2),
                            ).type_as(hidden_states),
                        ),
                        dim=1,
                    )
            else:
                hidden_states = hidden_states[:, : self.size[1], :]
                hidden_states_mask = torch.ones(
                    hidden_states.size(dim=0), hidden_states.size(dim=1)
                )

            hidden_states = hidden_states[
                -self.n_layers :, :, :
            ]  # .unsqueeze(0)
            hidden_states_mask = hidden_states_mask[
                -self.n_layers :, :
            ]  # .unsqueeze(0)
            item.update(
                [("utt", hidden_states), ("utt_mask", hidden_states_mask)]
            )

            item.update(
                [
                    (
                        "kwd_mask",
                        [
                            [hs for hs in group["kwd_mask"]]
                            for group in self.database
                        ],
                    ),
                ]
            )

        return item

    def is_expanded(self):
        return self.root_audios_transcripts != ""


class AishellHotwordDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "dev",
        r1_only: bool = False,
        size: Optional[Tuple[int, int]] = None,
        hotwords_per_group: int = -1,
        kw_type: str = "natural",
        load_audio: bool = False,
        wav_folder: str = None,
        feature_extractor: WhisperFeatureExtractor = None,
        learn_features: bool = False,
    ):

        # size of features
        assert size is None or (
            len(size) == 2 and all([i_ >= 32 for i_ in size])
        ), "provide a valid size for the input features of the KWS model"
        self.size = size

        # load audio
        self.load_audio = load_audio

        # save instance of WhisperFeatureExtractor
        self.feature_extractor = feature_extractor

        # check if the provided directories exist
        assert os.path.isdir(
            root
        ), "the directory you indicated with the dataset could not be found"
        self.root = root
        if self.load_audio:
            assert os.path.isdir(
                wav_folder
            ), "the directory you indicated with the audio files could not be found"

        # check if the indicated split is valid and whether the corresponding files exist
        self.valid_splits = ["dev", "test"]
        assert (
            split in self.valid_splits
        ), f"the indicated split name is not valid, got {split}"
        self.split_folder = os.path.join(root, split)
        assert os.path.isdir(
            self.split_folder
        ), "missing indicated split folder in the provided directory"

        # check if the hotwords file exists
        assert (
            os.path.exists(os.path.join(self.split_folder, "hotword.txt"))
            if not r1_only
            else os.path.exists(
                os.path.join(self.split_folder, "r1-hotword.txt")
            )
        ), "there is no keywords file in the dataset directory"
        # check if the transcript file exists
        assert os.path.exists(
            os.path.join(self.split_folder, "text")
        ), "the file with transcripts does not exist"

        # check if the keyword type is valid
        assert kw_type in [
            "tts",
            "natural",
            "all",
        ], "the provided keyword type is not valid"
        self.kw_type = kw_type
        if self.kw_type == "all":
            raise NotImplementedError(
                "the 'all' keyword type is not supported yet"
            )

        # get hotwords and corresponding whisper features in groups
        with open(
            (
                os.path.join(self.split_folder, "hotword.txt")
                if not r1_only
                else os.path.join(self.split_folder, "r1-hotword.txt")
            ),
            "r",
        ) as f:
            self.hotwords = [line.strip() for line in f.readlines()]
        self.database = []
        hw_zfill = len(str(len(self.hotwords) - 1))
        ghost_hotword_indices = []
        for idx, _ in enumerate(self.hotwords):
            if os.path.exists(
                os.path.join(
                    self.root,
                    split,
                    "keywords-hs",
                    self.kw_type,
                    str(idx).zfill(hw_zfill) + ".bin",
                )
            ):
                with open(
                    os.path.join(
                        self.root,
                        split,
                        "keywords-hs",
                        self.kw_type,
                        str(idx).zfill(hw_zfill) + ".bin",
                    ),
                    "rb",
                ) as f:
                    self.database.append(
                        torch.load(
                            f, map_location=torch.device("cpu")
                        ).detach()
                    )
            else:
                self.database.append(None)
                ghost_hotword_indices.append(idx)
        # set ghost hotwords features to zeros
        smaller_idx = min(
            [
                (idx, hs.shape)
                for idx, hs in enumerate(self.database)
                if hs != None
            ],
            key=lambda x: x[1][1],
        )[0]
        for idx in ghost_hotword_indices:
            self.database[idx] = torch.zeros_like(self.database[smaller_idx])
        # separate into groups
        # also adding a mask for the ghost hotword positions
        if hotwords_per_group == -1:
            self.hotwords_per_group = len(self.hotwords)
        else:
            self.hotwords_per_group = hotwords_per_group
        self.database = [
            {
                "keywords": self.hotwords[i : i + self.hotwords_per_group],
                "hidden_states": self.database[
                    i : i + self.hotwords_per_group
                ],
                "kwd": self.database[i : i + self.hotwords_per_group],
                "kwd_mask": [
                    0
                    for _ in range(
                        i, min(i + self.hotwords_per_group, len(self.hotwords))
                    )
                ],
                "max_length": max(
                    max(
                        [
                            t_.size(dim=1)
                            for t_ in self.database[
                                i : i + self.hotwords_per_group
                            ]
                        ]
                    ),
                    32,
                ),
                "mask": torch.tensor(
                    [
                        0 if idx in ghost_hotword_indices else 1
                        for idx in range(
                            i,
                            min(
                                i + self.hotwords_per_group, len(self.hotwords)
                            ),
                        )
                    ]
                ),
            }
            for i in range(0, len(self.hotwords), self.hotwords_per_group)
        ]

        for i, group in enumerate(self.database):
            for j, kwd in enumerate(group["kwd"]):
                if self.size[0] - kwd.size(dim=1) >= 0:
                    kwd_mask = torch.cat(
                        (
                            torch.ones(kwd.size(dim=0), kwd.size(dim=1)),
                            torch.zeros(
                                kwd.size(dim=0), self.size[0] - kwd.size(dim=1)
                            ),
                        ),
                        dim=1,
                    )

                    kwd = torch.cat(
                        (
                            kwd,
                            torch.zeros(
                                kwd.size(dim=0),
                                self.size[0] - kwd.size(dim=1),
                                kwd.size(dim=2),
                            ).type_as(kwd),
                        ),
                        dim=1,
                    )
                else:
                    kwd = kwd[:, : self.size[0], :]
                    kwd_mask = torch.ones(kwd.size(dim=0), kwd.size(dim=1))

                self.database[i]["kwd"][j] = kwd
                self.database[i]["kwd_mask"][j] = kwd_mask

        # get transcripts
        with open(os.path.join(self.split_folder, "text"), "r") as f:
            self.metadata = [
                [i_.strip() for i_ in line.split()] for line in f.readlines()
            ]
        regex = re.compile("BAC\d+(?P<subfolder>.+)W\d+")

        # create dataset
        self.dataset = [
            {
                "transcript": item[1],
                "utterance": {
                    "audio": (
                        os.path.join(
                            wav_folder,
                            split,
                            regex.match(item[0]).group("subfolder"),
                            item[0] + ".wav",
                        )
                        if self.load_audio
                        else None
                    ),
                    "hidden_states": os.path.join(
                        self.split_folder, "hs", item[0] + ".bin"
                    ),
                },
                "hotword_labels": [
                    torch.tensor(
                        [
                            1 if hotword in item[1] else 0
                            for hotword in self.hotwords[
                                i : i + self.hotwords_per_group
                            ]
                        ]
                    )
                    for i in range(
                        0, len(self.hotwords), self.hotwords_per_group
                    )
                ],
                "speaker": re.match(
                    "BAC\d{3}S(?P<speaker>\d{4}).+", item[0]
                ).groups("speaker"),
            }
            for item in self.metadata
        ]

        # whether or not to learn the features
        self.learn_features = learn_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = deepcopy(self.dataset[idx])
        # add ghost keywords mask
        item.update([("hotword_mask", [g["mask"] for g in self.database])])

        # Load hidden_states from utterance
        with open(item["utterance"]["hidden_states"], "rb") as f:
            hidden_states = torch.load(
                f, map_location=torch.device("cpu")
            ).detach()

        if False:
            # compute similarity matrices
            # simple inner product because vectors are normalized
            item.update(
                [
                    (
                        "features",
                        [
                            [
                                torch.matmul(hs, hidden_states.transpose(1, 2))
                                for hs in group["hidden_states"]
                            ]
                            for group in self.database
                        ],
                    )
                ]
            )
            if self.size is not None:
                # resize both edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (self.size[0], self.size[1]),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
            else:
                # resize only the short edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (
                                                group["max_length"],
                                                hidden_states.size(dim=1),
                                            ),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
        else:
            if self.size[1] - hidden_states.size(dim=1) >= 0:
                hidden_states_mask = torch.cat(
                    (
                        torch.ones(
                            hidden_states.size(dim=0),
                            hidden_states.size(dim=1),
                        ),
                        torch.zeros(
                            hidden_states.size(dim=0),
                            self.size[1] - hidden_states.size(dim=1),
                        ),
                    ),
                    dim=1,
                )

                hidden_states = torch.cat(
                    (
                        hidden_states,
                        torch.zeros(
                            hidden_states.size(dim=0),
                            self.size[1] - hidden_states.size(dim=1),
                            hidden_states.size(dim=2),
                        ).type_as(hidden_states),
                    ),
                    dim=1,
                )
            else:
                hidden_states = hidden_states[:, : self.size[1], :]
                hidden_states_mask = torch.ones(
                    hidden_states.size(dim=0), hidden_states.size(dim=1)
                )

            item.update(
                [
                    ("utt_features", hidden_states),
                    (
                        "kwd",
                        [
                            [hs for hs in group["kwd"]]
                            for group in self.database
                        ],
                    ),
                    ("utt", hidden_states),
                    ("utt_mask", hidden_states_mask),
                    (
                        "kwd_mask",
                        [
                            [hs for hs in group["kwd_mask"]]
                            for group in self.database
                        ],
                    ),
                ]
            )

        # load utterance audio
        if False:
            # load utterance audio and preprocess it
            waveform, sample_rate = torchaudio.load(item["utterance"]["audio"])
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(
                    torchaudio.functional.resample(
                        waveform, sample_rate, SAMPLE_RATE
                    ),
                    dim=0,
                    keepdim=True,
                )
            else:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, SAMPLE_RATE
                )
            # whether or not the audio has duration smaller than 30 seconds
            is_shortform = waveform.shape[0] <= N_SAMPLES
            # extract features
            if is_shortform:
                output_features = self.feature_extractor(
                    waveform[0],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    truncation=True if is_shortform else False,
                    padding="max_length" if is_shortform else "longest",
                    return_attention_mask=True,
                )
            features = output_features.input_features
            attention_mask = output_features.attention_mask
            item["utterance"].update(
                [("features", features), ("attention_mask", attention_mask)]
            )

        return item

    def is_expanded(self):
        return False


class ACL6060KeywordDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "dev",
        size: Optional[Tuple[int, int]] = None,
        keywords_per_group: int = -1,
        kw_type: str = "natural",
        load_audio: bool = False,
        feature_extractor: WhisperFeatureExtractor = None,
        learn_features: bool = False,
    ):

        # size of features
        assert size is None or (
            len(size) == 2 and all([i_ >= 32 for i_ in size])
        ), "provide a valid size for the input features of the KWS model"
        self.size = size

        # load audio
        self.load_audio = load_audio

        # save instance of WhisperFeatureExtractor
        self.feature_extractor = feature_extractor

        # check if the provided directories exist
        assert os.path.isdir(
            root
        ), "the directory you indicated with the dataset could not be found"
        self.root = root

        # check if the indicated split is valid and whether the corresponding files exist
        self.valid_splits = ["dev", "test"]
        assert (
            split in self.valid_splits
        ), f"the indicated split name is not valid, got {split}"
        self.split_folder = (
            os.path.join(root, "2", "acl_6060", split)
            if split == "dev"
            else os.path.join(root, "2", "acl_6060", "eval")
        )
        assert os.path.isdir(
            self.split_folder
        ), "missing indicated split folder in the provided directory"

        # check if the keyword type is valid
        assert kw_type in [
            "tts",
            "natural",
            "all",
        ], "the provided keyword type is not valid"
        self.kw_type = kw_type
        if self.kw_type == "all":
            raise NotImplementedError(
                "the 'all' keyword type is not supported yet"
            )

        # check if important files exist
        assert (
            os.path.exists(
                os.path.join(
                    self.split_folder,
                    "text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt",
                )
            )
            if split == "dev"
            else os.path.exists(
                os.path.join(
                    self.split_folder,
                    "text/tagged_terminology/ACL.6060.eval.tagged.en-xx.en.txt",
                )
            )
        ), "there is no file with tagged transcripts in the dataset directory"
        assert (
            os.path.exists(
                os.path.join(
                    self.split_folder, "text/txt/ACL.6060.dev.en-xx.en.txt"
                )
            )
            if split == "dev"
            else os.path.exists(
                os.path.join(
                    self.split_folder, "text/txt/ACL.6060.eval.en-xx.en.txt"
                )
            )
        ), "there is no file with transcripts in the dataset directory"
        assert os.path.exists(
            os.path.join(self.split_folder, "text/keywords.txt")
        ), "the file with keywords does not exist"

        # get keywords and corresponding whisper features in groups
        with open(
            os.path.join(self.split_folder, "text", "keywords.txt"), "r"
        ) as f:
            self.keywords = [line.strip() for line in f.readlines()]
        self.database = []
        kw_zfill = len(str(len(self.keywords) - 1))
        ghost_keyword_indices = []
        for idx, _ in enumerate(self.keywords):
            if os.path.exists(
                os.path.join(
                    self.split_folder,
                    "keywords-hs",
                    self.kw_type,
                    str(idx).zfill(kw_zfill) + ".bin",
                )
            ):
                with open(
                    os.path.join(
                        self.split_folder,
                        "keywords-hs",
                        self.kw_type,
                        str(idx).zfill(kw_zfill) + ".bin",
                    ),
                    "rb",
                ) as f:
                    self.database.append(
                        torch.load(
                            f, map_location=torch.device("cpu")
                        ).detach()
                    )
            else:
                self.database.append(None)
                ghost_keyword_indices.append(idx)
        # set ghost keywords features to zeros
        smaller_idx = min(
            [
                (idx, hs.shape)
                for idx, hs in enumerate(self.database)
                if hs != None
            ],
            key=lambda x: x[1][1],
        )[0]
        for idx in ghost_keyword_indices:
            self.database[idx] = torch.zeros_like(self.database[smaller_idx])
        # separate into groups
        # also adding a mask for the ghost keyword positions
        if keywords_per_group == -1:
            self.keywords_per_group = len(self.keywords)
        else:
            self.keywords_per_group = keywords_per_group
        self.database = [
            {
                "keywords": self.keywords[i : i + self.keywords_per_group],
                "hidden_states": self.database[
                    i : i + self.keywords_per_group
                ],
                "kwd": self.database[i : i + self.keywords_per_group],
                "kwd_mask": [
                    0
                    for _ in range(
                        i, min(i + self.keywords_per_group, len(self.keywords))
                    )
                ],
                "max_length": max(
                    max(
                        [
                            t_.size(dim=1)
                            for t_ in self.database[
                                i : i + self.keywords_per_group
                            ]
                        ]
                    ),
                    32,
                ),
                "mask": torch.tensor(
                    [
                        0 if idx in ghost_keyword_indices else 1
                        for idx in range(
                            i,
                            min(
                                i + self.keywords_per_group, len(self.keywords)
                            ),
                        )
                    ]
                ),
            }
            for i in range(0, len(self.keywords), self.keywords_per_group)
        ]

        for i, group in enumerate(self.database):
            for j, kwd in enumerate(group["kwd"]):
                if self.size[0] - kwd.size(dim=1) >= 0:
                    kwd_mask = torch.cat(
                        (
                            torch.ones(kwd.size(dim=0), kwd.size(dim=1)),
                            torch.zeros(
                                kwd.size(dim=0), self.size[0] - kwd.size(dim=1)
                            ),
                        ),
                        dim=1,
                    )

                    kwd = torch.cat(
                        (
                            kwd,
                            torch.zeros(
                                kwd.size(dim=0),
                                self.size[0] - kwd.size(dim=1),
                                kwd.size(dim=2),
                            ).type_as(kwd),
                        ),
                        dim=1,
                    )
                else:
                    kwd = kwd[:, : self.size[0], :]
                    kwd_mask = torch.ones(kwd.size(dim=0), kwd.size(dim=1))

                self.database[i]["kwd"][j] = kwd
                self.database[i]["kwd_mask"][j] = kwd_mask

        # load transcripts in correct order
        with open(
            (
                os.path.join(
                    self.split_folder, "text/txt/ACL.6060.dev.en-xx.en.txt"
                )
                if split == "dev"
                else os.path.join(
                    self.split_folder, "text/txt/ACL.6060.eval.en-xx.en.txt"
                )
            ),
            "r",
        ) as f:
            transcripts = [line.strip() for line in f.readlines()]

        # load keywords for each transcript
        with open(
            (
                os.path.join(
                    self.split_folder,
                    "text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt",
                )
                if split == "dev"
                else os.path.join(
                    self.split_folder,
                    "text/tagged_terminology/ACL.6060.eval.tagged.en-xx.en.txt",
                )
            ),
            "r",
        ) as f:
            keywords = [
                [
                    {
                        "mention": (
                            match.group(1)
                            if (x := match.group(1)) in self.keywords
                            else x[0].lower() + x[1:]
                        ),
                        "total_offset": match.start() - m_idx * 2,
                        "end_offset": match.end() - m_idx * 2 - 2,
                    }
                    for m_idx, match in enumerate(
                        re.finditer("\[(\w+)\]", line)
                    )
                ]
                for line in f.readlines()
            ]

        # get speaker information
        with open(
            (
                os.path.join(
                    self.split_folder, "text/xml/ACL.6060.dev.en-xx.en.xml"
                )
                if split == "dev"
                else os.path.join(
                    self.split_folder, "text/xml/ACL.6060.eval.en-xx.en.xml"
                )
            ),
            "r",
        ) as f:
            root = ET.fromstring(re.sub("&", "", f.read()))
        idx2speaker = {
            int(child.attrib["id"]): speaker_id
            for speaker_id, doc in enumerate(root[0])
            for child in doc
            if child.tag == "seg"
        }

        # create dataset
        self.dataset = [
            {
                "transcript": transcript,
                "utterance": {
                    "audio": (
                        os.path.join(
                            self.split_folder,
                            "segmented_wavs/gold",
                            "sent_" + str(idx + 1) + ".wav",
                        )
                        if self.load_audio
                        else None
                    ),
                    "hidden_states": os.path.join(
                        self.split_folder,
                        "hs",
                        "sent_" + str(idx + 1) + ".bin",
                    ),
                },
                "hotword_labels": (
                    [
                        torch.tensor(
                            [
                                1 if keyword in transcript else 0
                                for keyword in self.keywords[
                                    i : i + self.keywords_per_group
                                ]
                            ]
                        )
                        for i in range(
                            0, len(self.keywords), self.keywords_per_group
                        )
                    ]
                    if split == "dev"
                    else [
                        torch.tensor(
                            [
                                (
                                    1
                                    if keyword
                                    in [kw_["mention"] for kw_ in kw]
                                    else 0
                                )
                                for keyword in self.keywords[
                                    i : i + self.keywords_per_group
                                ]
                            ]
                        )
                        for i in range(
                            0, len(self.keywords), self.keywords_per_group
                        )
                    ]
                ),
                "keywords": kw,
                "speaker": idx2speaker[idx + 1],
            }
            for idx, (transcript, kw) in enumerate(zip(transcripts, keywords))
        ]

        # whether or not to learn the features
        self.learn_features = learn_features

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = deepcopy(self.dataset[idx])
        # add ghost keywords mask
        item.update([("hotword_mask", [g["mask"] for g in self.database])])

        # Load hidden_states from utterance
        with open(item["utterance"]["hidden_states"], "rb") as f:
            hidden_states = torch.load(
                f, map_location=torch.device("cpu")
            ).detach()

        if False:
            # compute similarity matrices
            # simple inner product because vectors are normalized
            item.update(
                [
                    (
                        "features",
                        [
                            [
                                torch.matmul(hs, hidden_states.transpose(1, 2))
                                for hs in group["hidden_states"]
                            ]
                            for group in self.database
                        ],
                    )
                ]
            )
            if self.size is not None:
                # resize both edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (self.size[0], self.size[1]),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
            else:
                # resize only the short edges
                item.update(
                    [
                        (
                            "features",
                            [
                                torch.stack(
                                    [
                                        torchvision.transforms.functional.resize(
                                            matrices,
                                            (
                                                group["max_length"],
                                                hidden_states.size(dim=1),
                                            ),
                                            antialias=False,
                                        )
                                        for matrices in features
                                    ],
                                    dim=0,
                                )
                                for group, features in zip(
                                    self.database, item["features"]
                                )
                            ],
                        )
                    ]
                )
        else:
            if self.size[1] - hidden_states.size(dim=1) >= 0:
                hidden_states_mask = torch.cat(
                    (
                        torch.ones(
                            hidden_states.size(dim=0),
                            hidden_states.size(dim=1),
                        ),
                        torch.zeros(
                            hidden_states.size(dim=0),
                            self.size[1] - hidden_states.size(dim=1),
                        ),
                    ),
                    dim=1,
                )
                hidden_states = torch.cat(
                    (
                        hidden_states,
                        torch.zeros(
                            hidden_states.size(dim=0),
                            self.size[1] - hidden_states.size(dim=1),
                            hidden_states.size(dim=2),
                        ).type_as(hidden_states),
                    ),
                    dim=1,
                )
            else:
                hidden_states = hidden_states[:, : self.size[1], :]
                hidden_states_mask = torch.ones(
                    hidden_states.size(dim=0), hidden_states.size(dim=1)
                )
            item.update(
                [
                    (
                        "kwd",
                        [
                            [hs for hs in group["kwd"]]
                            for group in self.database
                        ],
                    ),
                    ("utt", hidden_states),
                    ("utt_mask", hidden_states_mask),
                    (
                        "kwd_mask",
                        [
                            [hs for hs in group["kwd_mask"]]
                            for group in self.database
                        ],
                    ),
                ]
            )
            # item.update(
            #     [
            #         ("utt_features", hidden_states),
            #         (
            #             "kwd_features",
            #             [
            #                 [hs for hs in group["hidden_states"]]
            #                 for group in self.database
            #             ],
            #         ),
            #     ]
            # )

        # load utterance audio
        if False:
            # load utterance audio and preprocess it
            waveform, sample_rate = torchaudio.load(item["utterance"]["audio"])
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(
                    torchaudio.functional.resample(
                        waveform, sample_rate, SAMPLE_RATE
                    ),
                    dim=0,
                    keepdim=True,
                )
            else:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, SAMPLE_RATE
                )
            # whether or not the audio has duration smaller than 30 seconds
            is_shortform = waveform.shape[0] <= N_SAMPLES
            # extract features
            if is_shortform:
                output_features = self.feature_extractor(
                    waveform[0],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    truncation=True if is_shortform else False,
                    padding="max_length" if is_shortform else "longest",
                    return_attention_mask=True,
                )
            features = output_features.input_features
            attention_mask = output_features.attention_mask
            item["utterance"].update(
                [("features", features), ("attention_mask", attention_mask)]
            )

        return item

    def is_expanded(self):
        return False
