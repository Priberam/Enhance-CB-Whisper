#!/bin/bash

# function to test existence of a file
exists () {
    if test -f $1; then
        echo ""
    else
        echo "$1 does not exist." >> $2
        exit 1
    fi
}

# create log file
LOG="logfile.log"
> $LOG

# ask for the path to the zipped aishell dataset
read -p "Path to the zipped aishell dataset: " AISHELL
exists $AISHELL $LOG

# unzip the aishell dataset
AISHELLROOT="${AISHELL%.*}"
if !(test -d $AISHELLROOT); then
    tar -xzvf $AISHELL -C $(dirname $AISHELL)
    for entry in "$AISHELLROOT/wav"/*
    do
      tar -xvf "$entry" -C "$AISHELLROOT/wav"
      rm "$entry"
    done
fi

# create kws (train) dataset
mkdir -p -- "$AISHELLROOT/kws"
for entry in "$PWD/train"/*
do
    cp "$entry" "$AISHELLROOT/kws/$(basename -- "$entry")"
done
mkdir -p -- "$AISHELLROOT/kws/hs"
mkdir -p -- "$AISHELLROOT/kws/keywords-audios"
mkdir -p -- "$AISHELLROOT/kws/keywords-hs"
mkdir -p -- "$AISHELLROOT/kws/keywords-audios/tts"
mkdir -p -- "$AISHELLROOT/kws/keywords-audios/natural"
mkdir -p -- "$AISHELLROOT/kws/keywords-hs/tts"
mkdir -p -- "$AISHELLROOT/kws/keywords-hs/natural"

# create aishell hotword (dev/test) dataset
HOTWORD_REPOSITORY="https://github.com/R1ckShi/SeACo-Paraformer.git"
if !(test -d "$AISHELLROOT/hotword"); then
    git clone "$HOTWORD_REPOSITORY" "$AISHELLROOT/hotword"
fi
if test -d "$AISHELLROOT/hotword/data"; then
    mv "$AISHELLROOT/hotword/data/dev" "$AISHELLROOT/hotword/dev"
    mv "$AISHELLROOT/hotword/data/test" "$AISHELLROOT/hotword/test"
    for file in "$AISHELLROOT/hotword/"* ; do
        if !([ $(basename -- "$file") == "dev" ] || [ $(basename -- "$file") == "test" ]); then
            rm -rf "$file"
        fi
    done
fi
for split in "dev" "test" ; do
    for entry in "$PWD/$split"/*
    do
        cp "$entry" "$AISHELLROOT/hotword/$split/$(basename -- "$entry")"
    done
    mkdir -p -- "$AISHELLROOT/hotword/$split/hs"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-audios"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-hs"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-audios/tts"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-audios/natural"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-hs/tts"
    mkdir -p -- "$AISHELLROOT/hotword/$split/keywords-hs/natural"
done

# extract the hidden states from the utterances
echo "Extracting the hidden states of the utterances from the whisper encoder..."
python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/wav/train/" -t "$AISHELLROOT/kws/hs/" -w "openai/whisper-medium" -u "$AISHELLROOT/kws/positives.tsv"
for split in "dev" "test"; do
    python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/wav/$split/" -t "$AISHELLROOT/hotword/$split/hs/" -w "openai/whisper-medium" -u "$AISHELLROOT/hotword/$split/uttid"
done
echo "Extraction completed."

# extract the keywords audios from natural speech
echo "Cutting audios to generate natural-speech-based keywords..."
python3 ../../src/utils.py --cut_audios -a "$AISHELLROOT/wav/train/" -k "$AISHELLROOT/kws/aligned.txt" -t "$AISHELLROOT/kws/keywords-audios/natural/"
for split in "dev" "test"; do
    python3 ../../src/utils.py --cut_audios -a "$AISHELLROOT/wav/$split/" -k "$AISHELLROOT/hotword/$split/aligned.txt" -t "$AISHELLROOT/hotword/$split/keywords-audios/natural/"
done
echo "Cutting completed."

# extract the hidden states from the natural-speech-based keywords
echo "Extracting the hidden states of the natural-speech-based keywords from the whisper encoder..."
python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/kws/keywords-audios/natural/" -t "$AISHELLROOT/kws/keywords-hs/natural/" -w "openai/whisper-medium"
for split in "dev" "test"; do
    python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/hotword/$split/keywords-audios/natural/" -t "$AISHELLROOT/hotword/$split/keywords-hs/natural/" -w "openai/whisper-medium"
done
echo "Extraction completed."

# use edge-tts to generate synthetic audios for the keywords
echo "Generating synthetic audios for the keywords..."
python3 ../../src/utils.py --tts -k "$AISHELLROOT/kws/keywords_voice.txt" -t "$AISHELLROOT/kws/keywords-audios/tts/" -l "zh-CN"
for split in "dev" "test"; do
    python3 ../../src/utils.py --tts -k "$AISHELLROOT/hotword/$split/hotword_voice.txt" -t "$AISHELLROOT/hotword/$split/keywords-audios/tts/" -l "zh-CN"
done
echo "Generation completed."

# extract the hidden states from the tts-based keywords
echo "Extracting the hidden states of the tts-based keywords from the whisper encoder..."
python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/kws/keywords-audios/tts/" -t "$AISHELLROOT/kws/keywords-hs/tts/" -w "openai/whisper-medium"
for split in "dev" "test"; do
    python3 ../../src/utils.py --extract_hs -a "$AISHELLROOT/hotword/$split/keywords-audios/tts/" -t "$AISHELLROOT/hotword/$split/keywords-hs/tts/" -w "openai/whisper-medium"
done
echo "Extraction completed."