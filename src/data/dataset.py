import os
import re
import torch
import torchvision
from torch.utils.data import Dataset
from copy import deepcopy
import torchaudio
from transformers import WhisperFeatureExtractor
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET
from itertools import accumulate
from whisper.audio import SAMPLE_RATE, N_SAMPLES


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d.__getitem__(i) for d in self.datasets)

    def __len__(self):
        return min(d.__len__() for d in self.datasets)
    

class AishellKWSDataset(Dataset):
    def __init__(
        self,
        root: str,
        kw_type: str = 'natural'
    ):
        
        # check if the provided directories exist
        assert os.path.isdir(os.path.join(root, 'kws')), f'the directory you indicated with the dataset could not be found'
        self.root = os.path.join(root, 'kws')

        # check if the keywords file exists
        assert os.path.exists(os.path.join(self.root, 'keywords.txt')), f'there is no keywords file in the dataset directory' 

        # check if the keyword type is valid
        assert kw_type in ['tts', 'natural', 'all'], f'the provided keyword type is not valid'
        self.kw_type = kw_type
        if self.kw_type == 'all':
            raise NotImplementedError(f'the \'all\' keyword type is not supported yet')
        
        # and get keywords list in lexicographic order
        self.keywords = {}
        self.ghost_keyword_indices = []
        with open(os.path.join(self.root, 'keywords.txt'), 'r') as f: 
            self.keywords = {line.split()[0].strip(): idx for idx, line in enumerate(f.readlines())}
        self.n_keywords = len(self.keywords)
        self.kw_zfill = len(str(len(self.keywords) - 1))
        self.ghost_keyword_indices = [idx for idx, _ in enumerate(self.keywords) if not os.path.exists(os.path.join(self.root, 'keywords-hs', self.kw_type, str(idx).zfill(self.kw_zfill) + '.bin'))]
        # and also in reverse lexicographic order
        self.keywords_reverse = sorted(self.keywords.keys(), key=lambda x: x[::-1])

        # check if the positive examples files exist
        assert os.path.exists(os.path.join(self.root, 'positives.tsv')), f'there is no positive examples file'
        # and read TSV files with metadata
        with open(os.path.join(self.root, 'positives.tsv'), 'r') as f:
            data = [[i_.strip() for i_ in line.split('\t')] for line in f.readlines()]
        self.metadata = [{
            'code': item[0],
            'positives': [(item[i_], int(item[i_+1]), int(item[i_+2])) for i_ in range(1, len(item), 3)]
        } for item in data]

        # set size of the dataset
        self.size = len(self.metadata) * self.n_keywords
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # find the utterance data that corresponds to the given idx
        data = self.metadata[idx // self.n_keywords]
        # the selected keyword index
        keyword_idx = idx % self.n_keywords

        # item to be returned
        # add label, mask and domain attributes
        item = {
            'label': 1 if any([keyword_idx == positive_idx for _, positive_idx, _ in data['positives']]) else 0,
            'mask': 1 if keyword_idx not in self.ghost_keyword_indices else 0,
            'domain': 0 if self.kw_type == 'tts' else 1
        }
        # load utterance and keyword hidden states
        with open(os.path.join(self.root, 'hs', data['code'] + '.bin'), 'rb') as f:
            utt = torch.load(f, map_location=torch.device('cpu')).detach()
        if item['mask'] == 1:
            with open(os.path.join(self.root, 'keywords-hs', self.kw_type, str(keyword_idx).zfill(self.kw_zfill) + '.bin'), 'rb') as f:
                kwd = torch.load(f, map_location=torch.device('cpu')).detach()
        else:
            kwd = torch.zeros(utt.size(dim=0), 1, utt.size(dim=2)).type_as(utt)
        # compute similarity matrices
        # simple inner product because vectors are normalized
        item.update([('features', torch.matmul(kwd, utt.transpose(1, 2)))])   

        # for debugging
        item.update([('code', data['code']), ('keyword', [kwd for kwd in self.keywords.keys() if self.keywords[kwd] == keyword_idx][0])])

        return item
    

class MLSKWSDataset(Dataset):
    def __init__(
        self,
        root: str,
        languages: List[str] = ['English', 'German', 'French', 'Spanish', 'Polish', 'Portuguese'],
        kw_type: str = 'natural'
    ):

        # check if the provided directories exist
        assert os.path.isdir(root), f'the directory you indicated with the dataset could not be found'
        assert all([os.path.isdir(os.path.join(root, 'mls_' + language.lower() + '_opus')) for language in languages]), f'at least one of the MLS subdatasets you asked could not be found'
        self.roots = {
            language: os.path.join(root, 'mls_' + language.lower() + '_opus', 'train')
        for language in languages} 
        self.languages = sorted(languages)       

        # check if the keyword type is valid
        assert kw_type in ['tts', 'natural'], f'the provided keyword type is not valid, got {kw_type}'
        self.kw_type = kw_type

        # check if the keywords files exist
        assert all([os.path.exists(os.path.join(root, 'keywords.txt')) for _, root in self.roots.items()]), f'there is no keywords file in at least one of the MLS subdatasets'
        # and get keywords list in lexicographic order
        self.keywords = {}
        self.kw_zfill = {}
        self.ghost_keyword_indices = {}
        for language, root in self.roots.items():
            with open(os.path.join(root, 'keywords.txt'), 'r') as f: 
                self.keywords[language] = {line.split()[0].strip(): idx for idx, line in enumerate(f.readlines())}
            self.kw_zfill[language] = len(str(len(self.keywords[language]) - 1))
            self.ghost_keyword_indices[language] = [idx for idx, _ in enumerate(self.keywords[language]) if not os.path.exists(os.path.join(self.roots[language], 'keywords-hs', self.kw_type, str(idx).zfill(self.kw_zfill[language]) + '.bin'))]
        # and also in reverse lexicographic order
        self.keywords_reverse = {
            language : sorted(keywords.keys(), key=lambda x: x[::-1])
        for language, keywords in self.keywords.items()}
        # accumulated number of keywords
        self.n_keywords = list(accumulate([len(k_) for k_ in self.keywords.values()]))

        # check if the positive examples files exist
        assert all([os.path.exists(os.path.join(root, 'positives.tsv')) for _, root in self.roots.items()]), f'there is no positive examples file in at least one of the MLS subdatasets'
        # and read TSV files with metadata
        self.metadata = []
        offset_idx = 0
        for language in self.languages:
            with open(os.path.join(self.roots[language], 'positives.tsv'), 'r') as f:
                data = [[i_.strip() for i_ in line.split('\t')] for line in f.readlines()]
            self.metadata.append({
                'language': language,
                'offset_idx': offset_idx,
                'data': [{
                    'code': item[0],
                    'positives': [(item[i_], int(item[i_+1]), int(item[i_+2])) for i_ in range(1, len(item), 3)]
                } for item in data]
            })
            offset_idx += len(data) * self.n_keywords[-1]
        # dataset size
        self.size = offset_idx
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        # find subdataset in metadata that corresponds to the given idx
        submetadata = self.metadata[[idx >= d_['offset_idx'] for d_ in self.metadata].index(False) - 1 if not all([idx >= d_['offset_idx'] for d_ in self.metadata]) else -1]
        # find the utterance data that corresponds to the given idx
        data = submetadata['data'][(idx - submetadata['offset_idx']) // self.n_keywords[-1]]
        # the selected keyword index
        keyword_idx = (idx - submetadata['offset_idx']) % self.n_keywords[-1]
        keyword_language_idx = [keyword_idx < n_ for n_ in self.n_keywords].index(True)
        keyword_idx = keyword_idx - self.n_keywords[keyword_language_idx - 1] if keyword_language_idx != 0 else keyword_idx

        # item to be returned
        # add label, mask and domain attributes
        item = {
            'label': 1 if any([keyword_idx == positive_idx for _, positive_idx, _ in data['positives']]) and submetadata['language'] == self.languages[keyword_language_idx] else 0,
            'mask': 1 if keyword_idx not in self.ghost_keyword_indices[self.languages[keyword_language_idx]] else 0,
            #'domain': (0 if self.kw_type == 'tts' else len(self.languages)**2) + keyword_language_idx * len(self.languages) + self.languages.index(submetadata['language'])
            'domain': (0 if self.kw_type == 'tts' else len(self.languages)) + self.languages.index(submetadata['language'])
        }
        # load utterance and keyword hidden states
        with open(os.path.join(self.roots[submetadata['language']], 'hs', data['code'] + '.bin'), 'rb') as f:
            utt = torch.load(f, map_location=torch.device('cpu')).detach()
        if item['mask'] == 1:
            with open(os.path.join(self.roots[self.languages[keyword_language_idx]], 'keywords-hs', self.kw_type, str(keyword_idx).zfill(self.kw_zfill[self.languages[keyword_language_idx]]) + '.bin'), 'rb') as f:
                kwd = torch.load(f, map_location=torch.device('cpu')).detach()
        else:
            kwd = torch.zeros(utt.size(dim=0), 1, utt.size(dim=2)).type_as(utt)
        # compute similarity matrices
        # simple inner product because vectors are normalized
        item.update([('features', torch.matmul(kwd, utt.transpose(1, 2)))])   

        # for debugging
        #item.update([('code', data['code']), ('keyword', [kw for kw in self.keywords[self.languages[keyword_language_idx]] if self.keywords[self.languages[keyword_language_idx]][kw] == keyword_idx][0])])

        return item


class AishellHotwordDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'dev',
        r1_only: bool = False,
        size: Optional[Tuple[int, int]] = None,
        hotwords_per_group: int = -1,
        kw_type: str = 'natural',
        load_audio: bool = False,
        wav_folder: str = None,
        feature_extractor: WhisperFeatureExtractor = None
    ):
        
        # size of features
        assert size is None or (len(size) == 2 and all([i_ >= 32 for i_ in size])), f'provide a valid size for the input features of the KWS model'
        self.size = size

        # load audio
        self.load_audio = load_audio

        # save instance of WhisperFeatureExtractor
        self.feature_extractor = feature_extractor

        # check if the provided directories exist
        assert os.path.isdir(root), f'the directory you indicated with the dataset could not be found'
        self.root = root
        if self.load_audio:
            assert os.path.isdir(wav_folder), f'the directory you indicated with the audio files could not be found'

        # check if the indicated split is valid and whether the corresponding files exist
        self.valid_splits = ['dev', 'test']
        assert split in self.valid_splits, f'the indicated split name is not valid, got {split}'
        self.split_folder = os.path.join(root, split)
        assert os.path.isdir(self.split_folder), f'missing indicated split folder in the provided directory'

        # check if the hotwords file exists
        assert os.path.exists(os.path.join(self.split_folder, 'hotword.txt')) if not r1_only else os.path.exists(os.path.join(self.split_folder, 'r1-hotword.txt')), f'there is no keywords file in the dataset directory'
        # check if the transcript file exists
        assert os.path.exists(os.path.join(self.split_folder, 'text')), f'the file with transcripts does not exist'

        # check if the keyword type is valid
        assert kw_type in ['tts', 'natural', 'all'], f'the provided keyword type is not valid'
        self.kw_type = kw_type
        if self.kw_type == 'all':
            raise NotImplementedError(f'the \'all\' keyword type is not supported yet')

        # get hotwords and corresponding whisper features in groups
        with open(os.path.join(self.split_folder, 'hotword.txt') if not r1_only else os.path.join(self.split_folder, 'r1-hotword.txt'), 'r') as f:
            self.hotwords = [line.strip() for line in f.readlines()]
        self.database = []
        hw_zfill = len(str(len(self.hotwords) - 1))
        ghost_hotword_indices = []
        for idx, _ in enumerate(self.hotwords):
            if os.path.exists(os.path.join(self.root, split, 'keywords-hs', self.kw_type, str(idx).zfill(hw_zfill) + '.bin')):
                with open(os.path.join(self.root, split, 'keywords-hs', self.kw_type, str(idx).zfill(hw_zfill) + '.bin'), 'rb') as f:
                    self.database.append(torch.load(f, map_location=torch.device('cpu')).detach())
            else:
                self.database.append(None)
                ghost_hotword_indices.append(idx)
        # set ghost hotwords features to zeros
        smaller_idx = min([(idx, hs.shape) for idx, hs in enumerate(self.database) if hs != None], key=lambda x: x[1][1])[0]
        for idx in ghost_hotword_indices:
            self.database[idx] = torch.zeros_like(self.database[smaller_idx])
        # separate into groups
        # also adding a mask for the ghost hotword positions
        if hotwords_per_group == -1:
            self.hotwords_per_group = len(self.hotwords)
        else:
            self.hotwords_per_group = hotwords_per_group
        self.database = [{
            'keywords': self.hotwords[i:i+self.hotwords_per_group],
            'hidden_states': self.database[i:i+self.hotwords_per_group],
            #'max_length': max([t_.size(dim=1) for t_ in self.database[i:i+self.hotwords_per_group]]),
            'max_length': max(max([t_.size(dim=1) for t_ in self.database[i:i+self.hotwords_per_group]]), 32),
            'mask': torch.tensor([0 if idx in ghost_hotword_indices else 1 for idx in range(i, min(i+self.hotwords_per_group, len(self.hotwords)))])
        } for i in range(0, len(self.hotwords), self.hotwords_per_group)]
        
        # get transcripts
        with open(os.path.join(self.split_folder, 'text'), 'r') as f:
            self.metadata = [[i_.strip() for i_ in line.split()] for line in f.readlines()]
        regex = re.compile('BAC\d+(?P<subfolder>.+)W\d+')

        # create dataset
        self.dataset = [{
            'transcript': item[1],
            'utterance': {
                'audio': os.path.join(wav_folder, split, regex.match(item[0]).group('subfolder'), item[0] + '.wav') if self.load_audio else None,
                'hidden_states': os.path.join(self.split_folder, 'hs', item[0] + '.bin')
            },
            'hotword_labels': [torch.tensor([1 if hotword in item[1] else 0 for hotword in self.hotwords[i:i+self.hotwords_per_group]]) for i in range(0, len(self.hotwords), self.hotwords_per_group)],
            'speaker': re.match('BAC\d{3}S(?P<speaker>\d{4}).+', item[0]).groups('speaker')
        } for item in self.metadata]
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = deepcopy(self.dataset[idx])
        # add ghost hotwords mask
        item.update([('hotword_mask', [g['mask'] for g in self.database])])

        # Load hidden_states from utterance
        with open(item['utterance']['hidden_states'], 'rb') as f:
            hidden_states = torch.load(f, map_location=torch.device('cpu')).detach()

        # compute similarity matrices
        # simple inner product because vectors are normalized
        item.update([('features', [[torch.matmul(hs, hidden_states.transpose(1, 2)) for hs in group['hidden_states']] for group in self.database])])
        if not self.size is None:
            # resize both edges
            item.update([('features', [torch.stack([torchvision.transforms.functional.resize(matrices, (self.size[0], self.size[1]), antialias=False) for matrices in features], dim=0) for group, features in zip(self.database, item['features'])])])
        else:
            # resize only the short edges
            item.update([('features', [torch.stack([torchvision.transforms.functional.resize(matrices, (group['max_length'], hidden_states.size(dim=1)), antialias=False) for matrices in features], dim=0) for group, features in zip(self.database, item['features'])])])

        # load utterance audio
        if self.load_audio:
            # load utterance audio and preprocess it
            waveform, sample_rate = torchaudio.load(item['utterance']['audio'])
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE), dim=0, keepdim=True)
            else:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            # whether or not the audio has duration smaller than 30 seconds
            is_shortform = waveform.shape[0] <= N_SAMPLES
            
            # extract features
            if is_shortform:
                output_features = self.feature_extractor(
                    waveform[0],
                    sampling_rate = SAMPLE_RATE,
                    return_tensors = 'pt',
                    truncation = True if is_shortform else False,
                    padding = 'max_length' if is_shortform else 'longest',
                    return_attention_mask = True
                )
            features = output_features.input_features
            attention_mask = output_features.attention_mask
            item['utterance'].update([('features', features), ('attention_mask', attention_mask)])

        return item


class ACL6060KeywordDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'dev',
        size: Optional[Tuple[int, int]] = None,
        keywords_per_group: int = -1,
        kw_type: str = 'natural',
        load_audio: bool = False,
        feature_extractor: WhisperFeatureExtractor = None
    ):
        
        # size of features
        assert size is None or (len(size) == 2 and all([i_ >= 32 for i_ in size])), f'provide a valid size for the input features of the KWS model'
        self.size = size

        # load audio
        self.load_audio = load_audio

        # save instance of WhisperFeatureExtractor
        self.feature_extractor = feature_extractor

        # check if the provided directories exist
        assert os.path.isdir(root), f'the directory you indicated with the dataset could not be found'
        self.root = root

        # check if the indicated split is valid and whether the corresponding files exist
        self.valid_splits = ['dev', 'test']
        assert split in self.valid_splits, f'the indicated split name is not valid, got {split}'
        self.split_folder = os.path.join(root, '2', 'acl_6060', split) if split == 'dev' else os.path.join(root, '2', 'acl_6060', 'eval')
        assert os.path.isdir(self.split_folder), f'missing indicated split folder in the provided directory'

        # check if the keyword type is valid
        assert kw_type in ['tts', 'natural', 'all'], f'the provided keyword type is not valid'
        self.kw_type = kw_type
        if self.kw_type == 'all':
            raise NotImplementedError(f'the \'all\' keyword type is not supported yet')

        # check if important files exist
        assert os.path.exists(os.path.join(self.split_folder, 'text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt')) if split == 'dev' else os.path.exists(os.path.join(self.split_folder, 'text/tagged_terminology/ACL.6060.eval.tagged.en-xx.en.txt')), f'there is no file with tagged transcripts in the dataset directory'
        assert os.path.exists(os.path.join(self.split_folder, 'text/txt/ACL.6060.dev.en-xx.en.txt')) if split == 'dev' else os.path.exists(os.path.join(self.split_folder, 'text/txt/ACL.6060.eval.en-xx.en.txt')), f'there is no file with transcripts in the dataset directory'
        assert os.path.exists(os.path.join(self.split_folder, 'text/keywords.txt')), f'the file with keywords does not exist'

        # get keywords and corresponding whisper features in groups
        with open(os.path.join(self.split_folder, 'text', 'keywords.txt'), 'r') as f:
            self.keywords = [line.strip() for line in f.readlines()]
        self.database = []
        kw_zfill = len(str(len(self.keywords) - 1))
        ghost_keyword_indices = []
        for idx, _ in enumerate(self.keywords):
            if os.path.exists(os.path.join(self.split_folder, 'keywords-hs', self.kw_type, str(idx).zfill(kw_zfill) + '.bin')):
                with open(os.path.join(self.split_folder, 'keywords-hs', self.kw_type, str(idx).zfill(kw_zfill) + '.bin'), 'rb') as f:
                    self.database.append(torch.load(f, map_location=torch.device('cpu')).detach())
            else:
                self.database.append(None)
                ghost_keyword_indices.append(idx)
        # set ghost keywords features to zeros
        smaller_idx = min([(idx, hs.shape) for idx, hs in enumerate(self.database) if hs != None], key=lambda x: x[1][1])[0]
        for idx in ghost_keyword_indices:
            self.database[idx] = torch.zeros_like(self.database[smaller_idx])
        # separate into groups
        # also adding a mask for the ghost keyword positions
        if keywords_per_group == -1:
            self.keywords_per_group = len(self.keywords)
        else:
            self.keywords_per_group = keywords_per_group
        self.database = [{
            'keywords': self.keywords[i:i+self.keywords_per_group],
            'hidden_states': self.database[i:i+self.keywords_per_group],
            #'max_length': max([t_.size(dim=1) for t_ in self.database[i:i+self.keywords_per_group]]),
            'max_length': max(max([t_.size(dim=1) for t_ in self.database[i:i+self.keywords_per_group]]), 32),
            'mask': torch.tensor([0 if idx in ghost_keyword_indices else 1 for idx in range(i, min(i+self.keywords_per_group, len(self.keywords)))])
        } for i in range(0, len(self.keywords), self.keywords_per_group)]

        # load transcripts in correct order
        with open(os.path.join(self.split_folder, 'text/txt/ACL.6060.dev.en-xx.en.txt') if split == 'dev' else os.path.join(self.split_folder, 'text/txt/ACL.6060.eval.en-xx.en.txt'), 'r') as f:
            transcripts = [line.strip() for line in f.readlines()]
        # load keywords for each transcript
        with open(os.path.join(self.split_folder, 'text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt') if split == 'dev' else os.path.join(self.split_folder, 'text/tagged_terminology/ACL.6060.eval.tagged.en-xx.en.txt'), 'r') as f:
            keywords = [[{
                'mention': match.group(1) if (x := match.group(1)) in self.keywords else x[0].lower() + x[1:],
                'total_offset': match.start() - m_idx * 2,
                'end_offset': match.end() - m_idx * 2 - 2
            } for m_idx, match in enumerate(re.finditer('\[(\w+)\]', line))] for line in f.readlines()]

        # get speaker information
        with open(os.path.join(self.split_folder, 'text/xml/ACL.6060.dev.en-xx.en.xml') if split == 'dev' else os.path.join(self.split_folder, 'text/xml/ACL.6060.eval.en-xx.en.xml'), 'r') as f:
            root = ET.fromstring(re.sub('&', '', f.read()))
        idx2speaker = {int(child.attrib['id']) : speaker_id for speaker_id, doc in enumerate(root[0]) for child in doc if child.tag == 'seg'}

        # create dataset
        self.dataset = [{
            'transcript': transcript,
            'utterance': {
                'audio': os.path.join(self.split_folder, 'segmented_wavs/gold', 'sent_' + str(idx + 1) + '.wav') if self.load_audio else None,
                'hidden_states': os.path.join(self.split_folder, 'hs', 'sent_' + str(idx + 1) + '.bin')
            },
            'hotword_labels': [torch.tensor([1 if keyword in transcript else 0 for keyword in self.keywords[i:i+self.keywords_per_group]]) for i in range(0, len(self.keywords), self.keywords_per_group)] if split == 'dev' else [torch.tensor([1 if keyword in [kw_['mention'] for kw_ in kw] else 0 for keyword in self.keywords[i:i+self.keywords_per_group]]) for i in range(0, len(self.keywords), self.keywords_per_group)],
            'keywords': kw,
            'speaker': idx2speaker[idx + 1]
        } for idx, (transcript, kw) in enumerate(zip(transcripts, keywords))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = deepcopy(self.dataset[idx])
        # add ghost keywords mask
        item.update([('hotword_mask', [g['mask'] for g in self.database])])

        # Load hidden_states from utterance
        with open(item['utterance']['hidden_states'], 'rb') as f:
            hidden_states = torch.load(f, map_location=torch.device('cpu')).detach()

        # compute similarity matrices
        # simple inner product because vectors are normalized
        item.update([('features', [[torch.matmul(hs, hidden_states.transpose(1, 2)) for hs in group['hidden_states']] for group in self.database])])
        if not self.size is None:
            # resize both edges
            item.update([('features', [torch.stack([torchvision.transforms.functional.resize(matrices, (self.size[0], self.size[1]), antialias=False) for matrices in features], dim=0) for group, features in zip(self.database, item['features'])])])
        else:
            # resize only the short edges
            item.update([('features', [torch.stack([torchvision.transforms.functional.resize(matrices, (group['max_length'], hidden_states.size(dim=1)), antialias=False) for matrices in features], dim=0) for group, features in zip(self.database, item['features'])])])

        # load utterance audio
        if self.load_audio:
            # load utterance audio and preprocess it
            waveform, sample_rate = torchaudio.load(item['utterance']['audio'])
            if waveform.size(dim=0) > 1:
                waveform = torch.mean(torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE), dim=0, keepdim=True)
            else:
                waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
            # whether or not the audio has duration smaller than 30 seconds
            is_shortform = waveform.shape[0] <= N_SAMPLES
            # extract features
            if is_shortform:
                output_features = self.feature_extractor(
                    waveform[0],
                    sampling_rate = SAMPLE_RATE,
                    return_tensors = 'pt',
                    truncation = True if is_shortform else False,
                    padding = 'max_length' if is_shortform else 'longest',
                    return_attention_mask = True
                )
            features = output_features.input_features
            attention_mask = output_features.attention_mask
            item['utterance'].update([('features', features), ('attention_mask', attention_mask)])

        return item