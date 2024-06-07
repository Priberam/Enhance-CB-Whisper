#!/bin/bash

# function to test existence of a folder
exists () {
    if test -d $1; then
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
read -p "Path to the folder with the zipped MLS subdatasets: " MLSROOT
exists $MLSROOT $LOG

# list of MLS subdatasets
MLSSUBCORPORA=("mls_english_opus" "mls_french_opus" "mls_german_opus" "mls_polish_opus" "mls_portuguese_opus" "mls_spanish_opus")
# and locales for synthetic voices
LOCALES=("en-US" "fr-FR" "de-DE" "pl-PL" "pt-PT" "es-ES")

# unzip each of the MLS subdatasets, if not unzipped already
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    if !(test -d "$MLSROOT/$MLSSUBCORPUS"); then
        if !(test -f "$MLSROOT/$MLSSUBCORPUS.tar.gz"); then
            MLSSUBCORPORA=( "${MLSSUBCORPORA[@]/$MLSSUBCORPUS}" )
        else
            echo "Unzipping $MLSROOT/$MLSSUBCORPUS.tar.gz..."
            tar -xzvf "$MLSROOT/$MLSSUBCORPUS.tar.gz" -C $(dirname "$MLSROOT/$MLSSUBCORPUS.tar.gz")
        fi
    else
        echo "$MLSROOT/$MLSSUBCORPUS already exists"
    fi
done

# copy some metadata from repo and create some folders
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    for entry in "$PWD/train/$MLSSUBCORPUS"/*
    do
        cp "$entry" "$MLSROOT/$MLSSUBCORPUS/train/$(basename -- "$entry")"
    done
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/hs"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-hs"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios/tts"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios/natural"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-hs/tts"
    mkdir -p -- "$MLSROOT/$MLSSUBCORPUS/train/keywords-hs/natural"
done

# extract the hidden states from the utterances
echo "Extracting the hidden states of the utterances from the whisper encoder..."
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    python3 ../../src/utils.py --extract_hs -a "$MLSROOT/$MLSSUBCORPUS/train/audio/" -t "$MLSROOT/$MLSSUBCORPUS/train/hs/" -w "openai/whisper-medium" -u "$MLSROOT/$MLSSUBCORPUS/train/uttid"
done
echo "Extraction completed."

# extract the keywords audios from natural speech
echo "Cutting audios to generate natural-speech-based keywords..."
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    python3 ../../src/utils.py --cut_audios -a "$MLSROOT/$MLSSUBCORPUS/train/audio/" -k "$MLSROOT/$MLSSUBCORPUS/train/aligned.tsv" -t "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios/natural/"
done
echo "Cutting completed."

# extract the hidden states from the natural-speech-based keywords
echo "Extracting the hidden states of the natural-speech-based keywords from the whisper encoder..."
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    python3 ../../src/utils.py --extract_hs -a "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios/natural/" -t "$MLSROOT/$MLSSUBCORPUS/train/keywords-hs/natural/" -w "openai/whisper-medium"
done
echo "Extraction completed."

# use edge-tts to generate synthetic audios for the keywords
echo "Generating synthetic audios for the keywords..."
for (( i=0; i<${#MLSSUBCORPORA[*]}; ++i)); do
    python3 ../../src/utils.py --tts -k "$MLSROOT/${MLSSUBCORPORA[$i]}/train/keywords_voice.txt" -t "$MLSROOT/${MLSSUBCORPORA[$i]}/train/keywords-audios/tts/" -l "${LOCALES[$i]}"
done
echo "Generation completed."

# extract the hidden states from the tts-based keywords
echo "Extracting the hidden states of the tts-based keywords from the whisper encoder..."
for MLSSUBCORPUS in ${MLSSUBCORPORA[@]}; do
    python3 ../../src/utils.py --extract_hs -a "$MLSROOT/$MLSSUBCORPUS/train/keywords-audios/tts/" -t "$MLSROOT/$MLSSUBCORPUS/train/keywords-hs/tts/" -w "openai/whisper-medium"
done
echo "Extraction completed."