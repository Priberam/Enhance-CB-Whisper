import torch
from torch.utils.data.sampler import Sampler
from typing import Iterator, Sized


class AishellKWSSampler(Sampler[int]):
    data_source: Sized

    def __init__(
        self, 
        data_source: Sized,
        sampling: str = 'random',
        negative_examples: dict = {'random': 1, 'lexicographic': 2},
        negative_diversity: float = 5.,
        resample_every_epoch: bool = True,
        seed: int = 123
    ) -> None:
        
        # dataset instance
        self.data_source = data_source

        # check if the sampling method is valid
        assert sampling in ['random', 'utterance-examples'], f'the provided sampling method for the MLSKWSDataset does not exist'
        self.sampling = sampling

        # check if the number of negative examples
        assert all([n_type in ['random', 'lexicographic'] for n_type in negative_examples.keys()]), f'the number of negative examples parameter got an invalid key'
        assert negative_examples.get('lexicographic', None) is None or negative_examples['lexicographic'] % 2 == 0, f'the number of negative examples of type `lexicographic` must be a multiple of 2'
        self.negative_examples = negative_examples

        # get number of samples
        self.num_samples = len(self.data_source.metadata) * (1 + sum(self.negative_examples.values()))

        # standard deviation of zero-mean Gaussian distribution to sample lexicographic negative examples
        self.negative_diversity = negative_diversity

        # total number of keywords
        self.n_keywords = len(self.data_source.keywords)

        # for resampling every epoch
        self.resample_every_epoch = resample_every_epoch
        self.seed = seed

    def __iter__(self) -> Iterator[int]:

        if not self.resample_every_epoch:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None
        
        # sample the utt/kwd pair indices
        indices = []
        
        for utt_idx, utterance in enumerate(self.data_source.metadata):
            # sample positive example
            positive = utterance['positives'][torch.randint(high=len(utterance['positives']), size=(1,), generator=generator).item()]
            positive_idx = utt_idx * self.n_keywords + positive[1]
            indices.append(positive_idx)
            indices_to_avoid = set([utt_idx * self.n_keywords + p_[1] for p_ in utterance['positives']])
            # sample random negative examples
            if self.negative_examples.get('random', None) != None and self.negative_examples['random'] > 0:
                while len(set(random_idx := (utt_idx * self.n_keywords + torch.randint(high=self.n_keywords, size=(self.negative_examples['random'],), generator=generator)).tolist()) - indices_to_avoid) != self.negative_examples['random']:
                    continue
                indices += random_idx
                indices_to_avoid.union(set(random_idx))
            # sample lexicographic negative examples
            if self.negative_examples.get('lexicographic', None) != None and self.negative_examples['random'] > 0:
                # from the left
                while len(set(lexicographic_idx := (positive_idx + torch.round(torch.randn((self.negative_examples['lexicographic'] // 2,), generator=generator) * self.negative_diversity).long()).tolist()) - indices_to_avoid) != self.negative_examples['lexicographic'] // 2 or not all([idx >= utt_idx * self.n_keywords and idx < utt_idx * self.n_keywords + self.n_keywords for idx in lexicographic_idx]):
                    continue
                indices += lexicographic_idx
                indices_to_avoid.union(set(lexicographic_idx))
                # and from the right
                while len(set(lexicographic_idx := [utt_idx * self.n_keywords + self.data_source.keywords[self.data_source.keywords_reverse[idx]] for idx in torch.round(positive[2] + torch.randn((self.negative_examples['lexicographic'] // 2,), generator=generator) * self.negative_diversity).long().tolist() if idx >= 0 and idx < self.n_keywords]) - indices_to_avoid) != self.negative_examples['lexicographic'] // 2:
                    continue
                indices += lexicographic_idx
        indices = torch.tensor(indices)   

        # shuffle the indices
        if self.sampling == 'random':
            yield from indices[torch.randperm(self.num_samples, generator=generator)].tolist()
        elif self.sampling == 'utterance-examples':
            n_examples = 1 + sum(self.negative_examples.values())
            yield from indices[((torch.randperm(self.num_samples // n_examples, generator=generator) * n_examples).unsqueeze(dim=1).repeat(1, n_examples) + torch.arange(n_examples)).flatten()].tolist()

    def __len__(self) -> int:
        return self.num_samples


class MLSKWSSampler(Sampler[int]):
    data_source: Sized

    def __init__(
        self, 
        data_source: Sized,
        sampling: str = 'random',
        negative_examples: dict = {'random': 1, 'lexicographic': 2},
        negative_diversity: float = 5.,
        resample_every_epoch: bool = True,
        seed: int = 123
    ) -> None:
        
        # dataset instance
        self.data_source = data_source

        # check if the sampling method is valid
        assert sampling in ['random', 'utterance-examples'], f'the provided sampling method for the MLSKWSDataset does not exist'
        self.sampling = sampling

        # check if the number of negative examples
        assert all([n_type in ['random', 'lexicographic'] for n_type in negative_examples.keys()]), f'the number of negative examples parameter got an invalid key'
        assert negative_examples.get('lexicographic', None) is None or negative_examples['lexicographic'] % 2 == 0, f'the number of negative examples of type `lexicographic` must be a multiple of 2'
        self.negative_examples = negative_examples

        # get number of samples
        self.num_samples = sum([len(submetadata['data']) * (1 + sum(self.negative_examples.values())) for submetadata in self.data_source.metadata])

        # standard deviation of zero-mean Gaussian distribution to sample lexicographic negative examples
        self.negative_diversity = negative_diversity

        # total number of keywords
        self.n_keywords = sum([len(k_) for k_ in self.data_source.keywords.values()])

        # for resampling every epoch
        self.resample_every_epoch = resample_every_epoch
        self.seed = seed

    def __iter__(self) -> Iterator[int]:

        if not self.resample_every_epoch:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            generator = None
        
        # sample the utt/kwd pair indices
        indices = []
        for submetadata in self.data_source.metadata:
            utt_language_keyword_offset = self.data_source.languages.index(submetadata['language'])
            utt_language_keyword_offset = self.data_source.n_keywords[utt_language_keyword_offset - 1] if utt_language_keyword_offset != 0 else 0
            for utt_idx, utterance in enumerate(submetadata['data']):
                # sample positive example
                positive = utterance['positives'][torch.randint(high=len(utterance['positives']), size=(1,), generator=generator).item()]
                positive_idx = submetadata['offset_idx'] + utt_idx * self.n_keywords + utt_language_keyword_offset + positive[1]
                indices.append(positive_idx)
                indices_to_avoid = set([submetadata['offset_idx'] + utt_idx * self.n_keywords + utt_language_keyword_offset + p_[1] for p_ in utterance['positives']])
                # sample random negative examples
                if self.negative_examples.get('random', None) != None and self.negative_examples['random'] > 0:
                    while len(set(random_idx := (submetadata['offset_idx'] + utt_idx * self.n_keywords + torch.randint(high=self.n_keywords, size=(self.negative_examples['random'],), generator=generator)).tolist()) - indices_to_avoid) != self.negative_examples['random']:
                        continue
                    indices += random_idx
                    indices_to_avoid.union(set(random_idx))
                # sample lexicographic negative examples
                if self.negative_examples.get('lexicographic', None) != None and self.negative_examples['random'] > 0:
                    # from the left
                    while len(set(lexicographic_idx := (positive_idx + torch.round(torch.randn((self.negative_examples['lexicographic'] // 2,), generator=generator) * self.negative_diversity).long()).tolist()) - indices_to_avoid) != self.negative_examples['lexicographic'] // 2 or not all([idx >= submetadata['offset_idx'] + utt_idx * self.n_keywords + utt_language_keyword_offset and idx < submetadata['offset_idx'] + utt_idx * self.n_keywords + utt_language_keyword_offset + len(self.data_source.keywords[submetadata['language']]) for idx in lexicographic_idx]):
                        continue
                    indices += lexicographic_idx
                    indices_to_avoid.union(set(lexicographic_idx))
                    # and from the right
                    while len(set(lexicographic_idx := [submetadata['offset_idx'] + utt_idx * self.n_keywords + utt_language_keyword_offset + self.data_source.keywords[submetadata['language']][self.data_source.keywords_reverse[submetadata['language']][idx]] for idx in torch.round(positive[2] + torch.randn((self.negative_examples['lexicographic'] // 2,), generator=generator) * self.negative_diversity).long().tolist() if idx >= 0 and idx < len(self.data_source.keywords[submetadata['language']])]) - indices_to_avoid) != self.negative_examples['lexicographic'] // 2:
                        continue
                    indices += lexicographic_idx
        indices = torch.tensor(indices) 

        # shuffle the indices
        if self.sampling == 'random':
            yield from indices[torch.randperm(self.num_samples, generator=generator)].tolist()
        elif self.sampling == 'utterance-examples':
            n_examples = 1 + sum(self.negative_examples.values())
            yield from indices[((torch.randperm(self.num_samples // n_examples, generator=generator) * n_examples).unsqueeze(dim=1).repeat(1, n_examples) + torch.arange(n_examples)).flatten()].tolist()

    def __len__(self) -> int:
        return self.num_samples