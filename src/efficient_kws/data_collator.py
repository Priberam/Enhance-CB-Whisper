import torch
from typing import Tuple, Optional


class KWSDataCollator:

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        learn_features: bool = False,
        load_embeddings: bool = True,
    ):
        # check size
        assert size is None or (
            len(size) == 2 and all([i_ >= 32 for i_ in size])
        ), "provide a valid size for the input features of the KWS model"
        self.size = size
        # whether to learn features
        self.learn_features = learn_features
        # whether the utterance embeddings are pre-computed
        self.load_embeddings = load_embeddings

    def __call__(self, features):
        # case each feature is a tuple
        # means the first uses keyword audios from tts and the second from natural speech
        # flatten the tuple
        # the remaining logic is valid
        if isinstance(features[0], tuple):
            features = [j_ for i_ in features for j_ in i_]

        # container for features and labels
        batch = {}

        for k in features[0]:
            if k not in ("idx", "label", "domain", "mask"):
                batch[k] = torch.stack(
                    [feature[k] for feature in features], dim=0
                )

        batch["labels"] = (
            torch.tensor([feature["label"] for feature in features])
            .type_as(batch["utt_features"])
            .long()
        )

        # join domain labels
        if features[0].get("domain", None) != None:
            batch["domain"] = (
                torch.tensor([feature["domain"] for feature in features])
                .type_as(batch["utt_features"])
                .long()
            )

        return batch


class HotwordDataCollator:

    def __call__(self, features):

        return features[0]
