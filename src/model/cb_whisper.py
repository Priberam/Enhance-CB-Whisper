import re
from typing import List, Union, Optional, Tuple
import torch
import torchvision
import pytorch_lightning as pl
from transformers import WhisperProcessor, WhisperModel
from .model import KWSModel
from .pba_whisper import PBAWhisper
import sys
sys.path.insert(1, '../data')
from dataset import AishellHotwordDataset, ACL6060KeywordDataset
from whisper.audio import N_FRAMES
from scorer import entity_recall
import random
import pandas as pd
from math import ceil
from confidence_intervals import evaluate_with_conf_int


class CBWhisper(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        split: str,
        root: str,
        kw_type: str,
        encoder_ckpt: str,
        whisper_ckpt: str,
        kws_ckpt: str,
        language: str,
        prompt: bool = True,
        oracle: Union[bool, str] = 'kws',
        kws_features_size: Optional[Tuple[int, int]] = (150, 750),
        keyword_prompt_prepend: str = '(',
        keyword_prompt_append: str = ')',
        keyword_separator: str = ' ',
        keywords_per_group: int = 100
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()  

        # create two instances of WhisperProcessor
        # one for PBAWhisper
        self.processor_whisper = WhisperProcessor.from_pretrained(
            self.hparams.whisper_ckpt,
            task = 'transcribe'
        )   
        # another for the Whisper encoder to extract features
        if self.hparams.encoder_ckpt == 'openai/whisper-large-v3':
            self.processor_encoder = WhisperProcessor.from_pretrained(self.hparams.whisper_ckpt)   
        else:
            self.processor_encoder = self.processor_whisper

        # create an instance of PBAWhisper
        self.whisper = PBAWhisper.from_pretrained(self.hparams.whisper_ckpt)

        # create an instance of a KWSModel
        self.kws = KWSModel.load_from_checkpoint(self.hparams.kws_ckpt)

        # create an instance of KeywordDatabase
        self.kw_database = DatabaseLite(
            dataset = self.hparams.dataset,
            split = self.hparams.split,
            root = self.hparams.root,
            kw_type = self.hparams.kw_type,
            keywords_per_group = self.hparams.keywords_per_group
        )

        # instantiate WhisperModel object and get encoder
        self.encoder = WhisperModel.from_pretrained(self.hparams.encoder_ckpt).encoder

        # check if oracle is valid
        if isinstance(self.hparams.oracle, bool):
            self.hparams.oracle = 'gold' if oracle else 'kws'
        assert self.hparams.oracle in ['gold', 'kws', 'random'], f'the provided oracle type is not supported, got f{oracle}'    

        # initialize oracle buffer
        self.oracle_buffer = []

    def keyword_spotting(
        self,
        input_features: torch.Tensor,
        start_of_prev: bool = False
    ):
        num_segments = input_features.size(dim=0)
        num_groups = self.kw_database.num_groups()
        keywords = [[]] * num_segments

        # case no prompt to provide
        if not self.hparams.prompt:
            return [[]] * num_segments 

        # case keywords are retrieved using the KWSModel
        if self.hparams.oracle == 'kws':
            if num_groups > 0:
                # extract hidden_states from the segment features
                try:
                    utt_hs = torch.stack(self.encoder(
                        input_features = input_features, 
                        output_hidden_states = True, 
                        return_dict = True
                    )['hidden_states'][10:22], dim=0).transpose(0, 1)
                    # normalize hidden states
                    utt_hs = utt_hs / torch.linalg.norm(utt_hs, dim=-1, keepdim=True)  
                except Exception as e:
                    utt_hs = None

            for idx in range(num_groups):
                kw_group = self.kw_database.group(idx, device=self.device)
                # calculate cosine similarity matrices
                cossim_matrices = self._calculate_cosine_similarity_matrices_(
                    utt_hs = utt_hs,
                    kwd_hs = kw_group['hidden_states']
                )
                # do keyword spotting
                # loop through each segment
                for seg_idx, matrices in enumerate(cossim_matrices):
                    # skip failed features
                    if matrices is None:
                        continue
                    # forward group of cossine similarity matrices through the KWS model
                    kws = self.kws.forward(
                        input_features = matrices
                    )
                    # and get the retrieved keywords
                    indices = torch.argwhere(torch.argmax(kws.logits, dim=1)).squeeze(dim=1)
                    keywords[seg_idx] += [kw_group['keywords'][idx] for idx in indices] 
            
            # remove duplicates
            keywords = [list(set(kwds)) for kwds in keywords]

        # case keywords are random or gold, copy from model buffer
        else:
            keywords = [self.oracle_buffer]

        #print([self.hparams.keyword_prompt_prepend + self.hparams.keyword_separator.join(kwds) + self.hparams.keyword_prompt_append if kwds != [] else [] for kwds in keywords])
        # and get prompt token ids
        if start_of_prev:
            kw_ids = [
                self.processor_whisper.get_prompt_ids(self.hparams.keyword_prompt_prepend + self.hparams.keyword_separator.join(kwds) + self.hparams.keyword_prompt_append) if kwds != [] else []
            for kwds in keywords]
        else:
            kw_ids = [
                self.processor_whisper.get_prompt_ids(self.hparams.keyword_prompt_prepend + self.hparams.keyword_separator.join(kwds) + self.hparams.keyword_prompt_append)[1:] if kwds != [] else []
            for kwds in keywords]

        return kw_ids    

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        oracle: List[str] = []
    ):
        # set buffer of the oracle
        self.oracle_buffer = oracle           

        # whether the audio is shorter than 30 seconds
        is_shortform = True
        if input_features.size(dim=0) > ceil(N_FRAMES / 2):
            is_shortform = False

        # generate transcript
        pred = self.whisper.generate(
            input_features = input_features,
            attention_mask = attention_mask,
            task = 'transcribe',
            language = self.hparams.language,
            return_timestamps = False if is_shortform else True,
            condition_on_prev_tokens = False if is_shortform else True,
            return_segments = False if is_shortform else True,
            num_beams = 5,
            do_sample = False,
            temperature = 0,
            keyword_spotting = self.keyword_spotting
        )
        # decode prediction
        if is_shortform:
            pred = self.processor_whisper.tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
        else:
            pred = self.processor_whisper.tokenizer.batch_decode(pred['sequences'], skip_special_tokens=True)[0]
        # and strip whitespaces
        pred = pred.strip()

        return pred
    
    def _calculate_cosine_similarity_matrices_(
        self,
        utt_hs: torch.Tensor,
        kwd_hs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        num_segments = utt_hs.size(dim=0)
        num_keywords = len(kwd_hs)
        # compute similarity matrices
        # simple inner product because vectors are normalized
        cossim_matrices = [matrices for matrices in [torch.matmul(kwd_hs_, utt_hs.transpose(2, 3)) if utt_hs != None else None for kwd_hs_ in kwd_hs]]
        cossim_matrices = [[cossim_matrices[kwd_idx][seg_idx] for kwd_idx in range(num_keywords)] for seg_idx in range(num_segments)]

        # resize edges
        if self.hparams.kws_features_size is None:
            short_edge = max([hs.size(dim=1) for hs in kwd_hs])
            long_edge = utt_hs.size(dim=2)
        else:
            short_edge = self.hparams.kws_features_size[0]
            long_edge = self.hparams.kws_features_size[1]
        cossim_matrices = [torch.stack([torchvision.transforms.functional.resize(matrices, (short_edge, long_edge), antialias=False) for matrices in cossim_matrices_], dim=0) if cossim_matrices_ != None else None for cossim_matrices_ in cossim_matrices]

        return cossim_matrices

    def on_test_epoch_start(self):
        # list for storing the outputs of each test step
        # in the past this was done automatically using the test_epoch_end hook
        # but https://github.com/Lightning-AI/lightning/pull/16520
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):  

        # parameters
        if self.hparams.oracle == 'gold':
            oracle = [self.kw_database[idx]['keyword'] for idx in torch.argwhere(torch.cat(batch['hotword_labels'], dim=0)).squeeze(dim=1)]
        elif self.hparams.oracle == 'random':
            oracle = [self.kw_database[idx]['keyword'] for idx in random.sample(list(set(range(len(self.kw_database))) - set(torch.argwhere(torch.cat(batch['hotword_labels'], dim=0)).squeeze(dim=1).tolist())), torch.sum(torch.cat(batch['hotword_labels'], dim=0)).item())]
        else:
            oracle = []
        
        # get predictions from the CB-Whisper for the given setting
        preds = self.forward(
            input_features = batch['utterance']['features'],
            attention_mask = batch['utterance']['attention_mask'],
            oracle = oracle
        )

        # add results for later evaluation
        self.test_step_outputs.append({
            'preds': preds,
            'target': batch['transcript'],
            'speaker': batch.get('speaker', None)
        })
        if batch.get('keywords', None) != None:
            self.test_step_outputs[-1].update([('keywords', batch['keywords'])])

    def on_test_epoch_end(self):

        # get list of predictions
        preds = [out['preds'] for out in self.test_step_outputs]
        # get list of reference transcripts
        refs = [out['target'] for out in self.test_step_outputs]    
        # and generate the list of keywords for each transcript
        if self.test_step_outputs[0].get('keywords', None) != None:
            keywords = [[{
                **kw, **{'ner_tag': 'UNK'}
            } for kw in step_output['keywords']] for step_output in self.test_step_outputs]
        else:
            keywords = [[{
                'mention': keyword,
                'total_offset': match.start(),
                'end_offset': match.end(),
                'ner_tag': 'UNK'
            } for group_idx in range(self.kw_database.num_groups()) for keyword in self.kw_database.group(group_idx, load_hs = False)['keywords'] for match in re.finditer(keyword, ref)] for ref in refs]

        def f_entity_recall(labels, samples, samples2=None):   
            preds_ = samples
            refs_, keywords_ = list(zip(*labels))
            # compute the entity recall for each setting
            recall = entity_recall(
                preds = preds_ ,
                refs = refs_,
                mentions = keywords_,
                ner_tags = 'ALL',
                char_split = True
            )['ALL']
            return recall

        # set conditions based on speaker
        speakers = [step_output['speaker'] for step_output in self.test_step_outputs]
        if speakers[0] == None:
            conditions = None
        else:
            speaker2id = {speaker: speaker_id for speaker_id, speaker in enumerate(set(speakers))}
            conditions = [speaker2id[speaker] for speaker in speakers]
        samples = Flexlist(preds)
        labels = Flexlist(zip(refs, keywords))
        recall = evaluate_with_conf_int(samples, f_entity_recall, labels, conditions, num_bootstraps=1000, alpha=5)

        # display results using pandas DataFrame
        results = pd.DataFrame([[recall[0], recall[1][0], recall[1][1]]], index=[('w/ prompt' if self.hparams.prompt else 'w/o prompt') + ' - ' + self.hparams.oracle], columns=["Entity Recall", "Entity Recall LB", "Entity Recall UB"])
        print(results)


class Flexlist(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return [self[int(k)] for k in keys]


class DatabaseLite:
    def __init__(
        self,
        dataset: str,
        split: str,
        root: str,
        kw_type: str,
        keywords_per_group: int = 100
    ):
        # check dataset
        assert dataset in ['aishell', 'acl'], f'DatabaseLite: the dataset is not supported, got {dataset}'
        # check split
        assert split in ['dev', 'test'], f'DatabaseLite: the split is not supported, got {split} for {dataset}'
        # check keyword type
        assert kw_type in ['tts', 'natural'], f'DatabaseLite: the keyword type is not supported, got {kw_type} for {dataset}'

        # get database
        if dataset == 'aishell':
            self.database = AishellHotwordDataset(
                root = root,
                split = split,
                r1_only = False,
                hotwords_per_group = keywords_per_group,
                kw_type = kw_type
            ).database
        elif dataset == 'acl':
            self.database = ACL6060KeywordDataset(
                root = root,
                split = split,
                keywords_per_group = keywords_per_group,
                kw_type = kw_type
            ).database

        # set group size
        self.keywords_per_group = keywords_per_group

        # set number of keywords
        self.num_keywords = sum([len(group['keywords']) for group in self.database])
        
    def __len__(
        self
    ) -> int:
        return self.num_keywords
    
    def __getitem__(
        self,
        idx: int
    ) -> dict:
        keyword = {
            'keyword': self.database[idx // self.keywords_per_group]['keywords'][idx % self.keywords_per_group],
            'hidden_states': self.database[idx // self.keywords_per_group]['hidden_states'][idx % self.keywords_per_group]
        }
        return keyword
    
    def num_groups(
        self
    ) -> int:
        return len(self.database)
    
    def group(
        self,
        idx: int,
        device: str = 'cpu',
        load_hs: bool = True
    ) -> dict:        
        kw_group = {
            'keywords' : self.database[idx]['keywords'],
            'hidden_states' : [hs.to(device) for hs in self.database[idx]['hidden_states']] if load_hs else None
        }
        return kw_group