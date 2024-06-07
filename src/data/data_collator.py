import torch
import torchvision
from typing import Tuple, Optional


class KWSDataCollator:

    def __init__(self, size: Optional[Tuple[int, int]] = None):
        # check size
        assert size is None or (len(size) == 2 and all([i_ >= 32 for i_ in size])), f'provide a valid size for the input features of the KWS model'
        self.size = size
  
    def __call__(self, features):

        # case each feature is a tuple
        # means the first uses keyword audios from tts and the second from natural speech
        # flatten the tuple
        # the remaining logic is valid
        if isinstance(features[0], tuple):
            features = [j_ for i_ in features for j_ in i_]

        # flatten features and labels
        if isinstance(features[0]['features'], list):
            features = [{
                'features': t_,
                'label': l_ if m_ == 1 else -100
            } for i_ in features for t_, l_, m_ in zip(i_['features'], i_['label'], i_['mask'])]

        batch_size = len(features)
        # container for features and labels
        batch = {}

        # get batch maximum lengths for resizing and padding
        #short_max_length = max(feature['features'].size(dim=1) for feature in features)
        #long_max_length = max(feature['features'].size(dim=2) for feature in features)
        if self.size is None:
            short_length = max(max(feature['features'].size(dim=1) for feature in features), 32)
            long_length = max(max(feature['features'].size(dim=2) for feature in features), 32)
        else:
            short_length = self.size[0]
            long_length = self.size[1]

        # resize short edges
        batch['features'] = [torchvision.transforms.functional.resize(feature['features'], (short_length, feature['features'].size(dim=2)), antialias=True) for feature in features]
        if self.size is None:
            # and pad long edges
            batch['features'] = torch.stack([torch.cat((t_, torch.zeros(t_.size(dim=0), t_.size(dim=1), long_length - t_.size(dim=2)).type_as(t_)), dim=2) for t_ in batch['features']], dim=0)
        else:
            # resize long edges
            batch['features'] = torch.stack([torchvision.transforms.functional.resize(feature, (feature.size(dim=1), long_length), antialias=True) for feature in batch['features']], dim=0)

        # join labels
        batch['labels'] = torch.tensor([feature['label'] for feature in features]).type_as(batch['features']).long()  

        # join domain labels
        if features[0].get('domain', None) != None:
            batch['domain'] = torch.tensor([feature['domain'] for feature in features]).type_as(batch['features']).long()  
        
        return batch


class HotwordDataCollator:
  
    def __call__(self, features):

        return features[0]