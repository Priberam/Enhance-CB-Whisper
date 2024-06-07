import os
from glob import glob
import argparse
from typing import Iterable
from tqdm import tqdm
from pydub import AudioSegment
import torch
import torchaudio
from transformers import WhisperFeatureExtractor, WhisperModel
from math import ceil
import random
import asyncio
import edge_tts
import xml.etree.ElementTree as ET


async def _get_voices() -> Iterable:
    voices = await edge_tts.VoicesManager.create()
    return voices


async def amain(
    text: str,
    voice: str,
    output_file: str
) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


def keyword_tts(
    tts_folder: str,
    keyword_file: str,
    locale: str,
    voice: str = None
):
    # argument check
    assert os.path.isdir(tts_folder), f'the provided folder for storing the synthesized speech does not exist'
    assert os.path.exists(keyword_file), f'there is no file with keywords list'

    # get list of already produced synthesized speech
    tts_file_indices = [int(os.path.splitext(os.path.basename(f_name))[0]) for f_name in glob(os.path.join(tts_folder, '*.mp3'))]

    # get keywords
    with open(keyword_file, 'r') as f:
        keywords = [{
            'keyword': line.split('\t')[0].strip(),
            'voice': line.split('\t')[1].strip() if len(line.split('\t')) != 1 else None,
            'idx': idx
        } for idx, line in enumerate(f.readlines())]

    leading_zeros = len(str(len(keywords) - 1))
    # remove indices of the already produced speech
    keywords = [item for item in keywords if item['idx'] not in tts_file_indices]

    # generate audio for each keyword
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        voices = loop.run_until_complete(_get_voices())
        l_voices = voices.find(Locale=locale)
        for item in tqdm(keywords):  
            if item['voice'] == None:
                v_ = random.choice(l_voices) if voice == None else l_voices[[v_['ShortName'] == voice for v_ in l_voices].index(True)]
            else:
                v_ = l_voices[[v_['ShortName'] == item['voice'] for v_ in l_voices].index(True)]
            item['voice'] = v_['ShortName']    
            while True:
                try:
                    loop.run_until_complete(amain(item['keyword'], v_['Name'], os.path.join(tts_folder, str(item['idx']).zfill(leading_zeros) + '.mp3')))
                except Exception as e:
                    print(e)
                    continue
                finally:
                    break
    finally:
        loop.close()

    # dump keywords metadata with voice information
    with open(os.path.splitext(keyword_file)[0] + '_voice.txt' if 'voice' not in keyword_file else keyword_file, 'w') as f:
        f.write('\n'.join(['\t'.join([item['keyword'], item['voice']]) for item in keywords]))


def get_keywords_audios(
    wav: str,
    keywords: str,
    keywords_audios: str
):    
    # check if dataset folder exists
    assert os.path.isdir(wav), f'the directory for the audios could not be found, got {wav}'

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    # or of folders with folders of only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(wav, '*'))]):
        if all([os.path.isdir(f_name) for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*'))]):
            audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for subsubfolder in glob(os.path.join(subfolder, '*')) for f_name in glob(os.path.join(subsubfolder, '*.mp3'))+glob(os.path.join(subsubfolder, '*.wav'))+glob(os.path.join(subsubfolder, '*.opus'))]
        else:
            audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(wav, '*.mp3'))+glob(os.path.join(wav, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] : f_name
    for f_name in audio_files}

    # load keywords data
    with open(keywords, 'r') as f:
        metadata = [{
            'keyword': line.split('\t')[0].strip(),
            'source': line.split('\t')[1].strip(),
            'start': int(float(line.split('\t')[2].strip()) * 1000),
            'end': int(float(line.split('\t')[3].strip()) * 1000)
        } if len(line.split('\t')) == 4 else None for line in f.readlines()]

    leading_zeros = len(str(len(metadata) - 1))
    for idx, m_ in tqdm(enumerate(metadata)):
        if m_ == None:
            continue
        # skip keywords that were not aligned
        if m_['start'] == m_['end']:
            continue
        # load the audio file
        audio = AudioSegment.from_file(audio_files[m_['source']])
        # cut the audio
        cut_audio = audio[m_['start']:m_['end']]
        # save the cut audio to a new file
        cut_audio.export(os.path.join(keywords_audios, str(idx).zfill(leading_zeros) + '.mp3'), format="mp3")


def extract_hidden_states(
    audios: str,
    whisper_ckpt: str,
    target: str,
    codes: str = None
):
    # check if audio and target folders exist
    assert os.path.isdir(audios), f'the directory for the audios could not be found, got {audios}'
    assert os.path.isdir(target), f'the directory for the target could not be found, got {target}'

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # instantiate WhisperProcessor object
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_ckpt)

    # instantiate WhisperModel object and get encoder
    encoder = WhisperModel.from_pretrained(whisper_ckpt).encoder.to(device)

    # get codes if necessary
    if codes != None:
        with open(codes, 'r') as f:
            codes = [line.split('\t')[0].strip().split(' ')[0].strip() for line in f.readlines()]

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    # or of folders with folders of only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(audios, '*'))]):
        if all([os.path.isdir(f_name) for subfolder in glob(os.path.join(audios, '*')) for f_name in glob(os.path.join(subfolder, '*'))]):
            audio_files = [f_name for subfolder in glob(os.path.join(audios, '*')) for subsubfolder in glob(os.path.join(subfolder, '*')) for f_name in glob(os.path.join(subsubfolder, '*.mp3'))+glob(os.path.join(subsubfolder, '*.wav'))+glob(os.path.join(subsubfolder, '*.opus'))]
        else:
            audio_files = [f_name for subfolder in glob(os.path.join(audios, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(audios, '*.mp3'))+glob(os.path.join(audios, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] : f_name
    for f_name in audio_files}

    # and extract hidden states for each one
    # if code is present in list of codes, if it exists
    for code, audio_file in tqdm(audio_files.items()):
        if codes != None and not any([c_ in code for c_ in codes]):
            continue
        try:
            # load utterance audio and preprocess it
            t_waveform, _ = torchaudio.load(audio_file)
            t_sample_rate = torchaudio.info(audio_file).sample_rate
            if t_waveform.size(dim=0) > 1:
                t_waveform = torch.mean(torchaudio.functional.resample(t_waveform, t_sample_rate, 16000), dim=0, keepdim=True)
            else:
                t_waveform = torchaudio.functional.resample(t_waveform, t_sample_rate, 16000)
            # extract features and hidden states
            t_features = feature_extractor(t_waveform[0], sampling_rate=16000, return_tensors='pt').input_features
            t_len = ceil(feature_extractor(t_waveform[0], sampling_rate=16000, return_tensors='pt', padding=True).input_features.size(dim=2) / 2.)
            t_hidden_states = torch.cat(encoder(
                input_features = t_features.to(device), 
                output_hidden_states = True, 
                return_dict = True
            )['hidden_states'][10:22], dim=0)[:, :t_len, :]

            # normalize hidden states
            t_hidden_states = t_hidden_states / torch.linalg.norm(t_hidden_states, dim=-1, keepdim=True)

            # target file name
            f_name = os.path.join(target, os.path.splitext(os.path.basename(audio_file))[0] + '.bin') if 'audio-' not in os.path.splitext(os.path.basename(audio_file))[0] else os.path.join(target, os.path.splitext(os.path.basename(audio_file))[0][6:] + '.bin')
            # and dump hidden_states
            with open(f_name, 'wb') as f:
                torch.save(t_hidden_states.clone(), f)

        except Exception as e:
            print(e)
            continue


def cut_audios(
    wav: str,
    segments: str,
    segments_audios: str
):    
    # check if dataset folder exists
    assert os.path.isdir(wav), f'the directory for the audios could not be found, got {wav}'
    # check if xml segments file exists
    assert os.path.exists(segments), f'the file with segments does not exist'

    # get all audio files
    # assumes the wav folder is either composed of only files
    # or of folders with only files
    if all([os.path.isdir(f_name) for f_name in glob(os.path.join(wav, '*'))]):
        audio_files = [f_name for subfolder in glob(os.path.join(wav, '*')) for f_name in glob(os.path.join(subfolder, '*.mp3'))+glob(os.path.join(subfolder, '*.wav'))]
    else:
        audio_files = [f_name for f_name in glob(os.path.join(wav, '*.mp3'))+glob(os.path.join(wav, '*.wav'))]
    audio_files = {
        os.path.splitext(os.path.basename(f_name))[0] if 'audio-' not in os.path.splitext(os.path.basename(f_name))[0] else os.path.splitext(os.path.basename(f_name))[0][6:] : f_name
    for f_name in audio_files}

    # parse xml file
    tree = ET.parse(segments)
    root = tree.getroot()
    for doc in tqdm(root):
        code = doc.attrib['code']
        for segment in doc:
            seg_id = segment.attrib['id']
            start = float(segment.attrib['start']) * 1000
            end = float(segment.attrib['end']) * 1000
            transcript = segment.find('current').text

            if transcript.strip() == '':
                continue
            # skip segments that were not aligned
            if start == end:
                continue
            # load the audio file
            audio = AudioSegment.from_file(audio_files[code])
            # cut the audio
            cut_audio = audio[start:end]
            # save the cut audio to a new file
            cut_audio.export(os.path.join(segments_audios, code + '-seg' + str(seg_id) + '.wav'), format='wav')


def main():

    parser = argparse.ArgumentParser(description = 'Utilities for building datasets')

    # command options
    parser.add_argument('--tts', dest='tts', action='store_true', help='use edge-tts to generate the audios for the keywords')
    parser.add_argument('--cut_audios', dest='cut_audios', action='store_true', help='cut audios')
    parser.add_argument('--extract_hs', dest='extract_hs', action='store_true', help='extract the hidden states from the whisper encoder')

    # input options
    parser.add_argument('-a', '--audios', dest='audios', type=str, help='folder with the audios')
    parser.add_argument('-k', '--keywords', dest='keywords', type=str, help='file with keywords and other relevant information')
    parser.add_argument('-t', '--target', dest='target', type=str, help='folder to store results')
    parser.add_argument('-u', '--utterances', dest='utterances', type=str, default='', help='list of utterances ids and other relevant information')
    parser.add_argument('-s', '--segments', dest='segments', type=str, help='xml file with segments from whole audios')
    parser.add_argument('-l', '--locale', dest='locale', type=str, help='locale input for edge-tts')
    parser.add_argument('-v', '--voice', dest='voice', type=str, default='', help='optional voice input for edge-tts')
    parser.add_argument('-w', '--whisper', dest='whisper', type=str, help='whisper version')

    args = parser.parse_args()

    if args.tts:
        keyword_tts(
            tts_folder = args.target,
            keyword_file = args.keywords,
            locale = args.locale,
            voice = args.voice if args.voice != '' else None
        )
    elif args.cut_audios:
        if args.segments != None:
            cut_audios(
                wav = args.audios,
                segments = args.segments,
                segments_audios = args.target
            )
        else:
            get_keywords_audios(
                wav = args.audios,
                keywords = args.keywords,
                keywords_audios = args.target
            )
    elif args.extract_hs:
        extract_hidden_states(
            audios = args.audios,
            whisper_ckpt = args.whisper,
            target = args.target,
            codes = args.utterances if args.utterances != '' else None
        )


if __name__ == '__main__':

    main()