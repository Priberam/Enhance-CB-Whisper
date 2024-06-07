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

# ask for the path to the zipped ACL6060 dataset
read -p "Path to the zipped acl dataset: " ACL6060
exists $ACL $LOG

# unzip the ACL6060 dataset
ACL6060ROOT="$(dirname $ACL6060)/ACL6060"
if !(test -d $ACL6060ROOT); then
    unzip $ACL6060 -d "$(dirname $ACL6060)/ACL6060"
fi

# create several folders and copy some metadata from repo
for split in "dev" "eval" ; do
    for entry in "$PWD/$split"/*
    do
        cp "$entry" "$ACL6060ROOT/2/acl_6060/$split/text/$(basename -- "$entry")"
    done
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/hs"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-audios"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-hs"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/tts"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/natural"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-hs/tts"
    mkdir -p -- "$ACL6060ROOT/2/acl_6060/$split/keywords-hs/natural"
done

# extract the hidden states from the utterances
echo "Extracting the hidden states of the utterances from the whisper encoder..."
for split in "dev" "eval"; do
    python3 ../../src/utils.py --extract_hs -a "$ACL6060ROOT/2/acl_6060/$split/segmented_wavs/gold/" -t "$ACL6060ROOT/2/acl_6060/$split/hs/" -w "openai/whisper-medium"
done
echo "Extraction completed."

# extract the keywords audios from natural speech
echo "Cutting audios to generate natural-speech-based keywords..."
for split in "dev" "eval"; do
    python3 ../../src/utils.py --cut_audios -a "$ACL6060ROOT/2/acl_6060/$split/segmented_wavs/gold/" -k "$ACL6060ROOT/2/acl_6060/$split/text/aligned.tsv" -t "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/natural/"
done
echo "Cutting completed."

# extract the hidden states from the natural-speech-based keywords
echo "Extracting the hidden states of the natural-speech-based keywords from the whisper encoder..."
for split in "dev" "eval"; do
    python3 ../../src/utils.py --extract_hs -a "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/natural/" -t "$ACL6060ROOT/2/acl_6060/$split/keywords-hs/natural/" -w "openai/whisper-medium"
done
echo "Extraction completed."

# use edge-tts to generate synthetic audios for the keywords
echo "Generating synthetic audios for the keywords..."
for split in "dev" "eval"; do
    python3 ../../src/utils.py --tts -k "$ACL6060ROOT/2/acl_6060/$split/text/keywords_voice.txt" -t "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/tts/" -l "en-US"
done
echo "Generation completed."

# extract the hidden states from the tts-based keywords
echo "Extracting the hidden states of the tts-based keywords from the whisper encoder..."
for split in "dev" "eval"; do
    python3 ../../src/utils.py --extract_hs -a "$ACL6060ROOT/2/acl_6060/$split/keywords-audios/tts/" -t "$ACL6060ROOT/2/acl_6060/$split/keywords-hs/tts/" -w "openai/whisper-medium"
done
echo "Extraction completed."