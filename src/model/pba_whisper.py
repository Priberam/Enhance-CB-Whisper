import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Optional, Union, Tuple, Callable
import warnings
from transformers.models.whisper.generation_whisper import _pad_to_max_length, _get_attr_from_logit_processors, WhisperGenerationMixin
import copy
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList


class PBAWhisper(WhisperForConditionalGeneration):  

    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        keyword_spotting: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Transcribes or translates log-mel input features to a sequence of auto-regressively generated token ids.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            return_timestamps (`bool`, *optional*):
                Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
            task (`str`, *optional*):
                Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                will be updated accordingly.
            language (`str`, *optional*):
                Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
            is_multilingual (`bool`, *optional*):
                Whether or not the model is multilingual.
            prompt_ids (`torch.Tensor`, *optional*):
                Rank-1 tensor of token IDs created by passing text to [`~WhisperProcessor.get_prompt_ids`] that is
                provided as a prompt to each chunk. This can be used to provide or "prompt-engineer" a context for
                transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those words
                correctly. It cannot be used in conjunction with `decoder_start_token_id` as it overwrites this value.
            condition_on_prev_tokens (`bool`, *optional*):
                Only relevant for long-form transcription. Whether to condition each segment on the previous segment.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            temperature (`float` or list of `float`, *optional*):
                The temperature to be used for generation. Passing a single `float` value and `do_sample=True` activates
                generation using sampling. For long-form transcription, temperature fallback can be activated by passing
                a list of float values such as (0.0, 0.2, 0.4, 0.6, 0.8, 1.0). As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            compression_ratio_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the zlib compression rate of each segment will be computed. If the compression rate of
                a segment is higher than `compression_ratio_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
                repeated using a higher temperature. The intuition behind this feature is that segments with very high compression rates
                suffer from a lot of repetition. The unwanted repetition can be reduced by injecting more randomness by increasing the temperature. If `compression_ratio_threshold` is defined
                make sure that `temperature` is a list of values. A common value for `compression_ratio_threshold` is 1.35.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            logprob_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the average log-probability of each segment will be computed. If the log-probability of
                a given segment is lower than `logprob_threshold`, temperature fallback is activated: the generated segment is discarded and the generation is
                repeated using a higher temperature. The intuition behind this feature is that segments of low log-probability
                can be improved by injecting more randomness by increasing the temperature. If `logprob_threshold` is defined
                make sure that `temperature` is a list of values. A common value for `logprob_threshold` is -1.0.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            no_speech_threshold (`float`, *optional*):
                Only relevant for long-form transcription. If defined, the "no-speech" token combined with the `logprob_threshold`
                is used to determine whether a segment contains only silence. In this case, the transcription for this segment
                is skipped.
                As shown in the [the Whisper paper](https://cdn.openai.com/papers/whisper.pdf), this can help to improve
                performance.
            num_segment_frames (`int`, *optional*):
                The number of frames a single segment is made of. If not defined, `num_segment_frames` defaults to the model's stride
                times the maximum input length.
            attention_mask (`torch.Tensor`, *optional*):
                `attention_mask` needs to be passed when doing long-form transcription using a batch size > 1.
            time_precision (`int`, *optional*, defaults to 0.02):
                The duration of output token in seconds. *E.g.* 0.02 means that a generated token on average accounts
                for 20 ms.
            return_token_timestamps (`bool`, *optional*):
                Whether to return token-level timestamps with the text. This can be used with or without the
                `return_timestamps` option. To get word-level timestamps, use the tokenizer to group the tokens into
                words.
            return_segments (`bool`, *optional*, defaults to `False`):
                Whether to additionally return a list of all segments. Note that this option can only be enabled
                when doing long-form transcription.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of just returning the generated tokens.
                Note that when doing long-form transcription, `return_dict_in_generate` can only be enabled when
                `return_segments` is set True. In this case the generation outputs of each segment is added to each
                segment.
            keyword_spotting (`Callable`, *optional*, defaults to `None`):
                A callable function that returns a list of token ids corresponding to the keywords present in a given 
                segment of audio.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor` or `Dict[str, Any]`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor` or a dict of segments when `return_segments=True`.

                If the passed input is > 30 seconds / > 3000 mel input features and `return_segments=True` then a dictionary of generated sequence ids, called `sequences` and a list of each generated segment is returned.

                else if the passed input is <= 30 seconds / >= 3000 mel input features, the possible [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]

                else only the generated output sequence ids are returned.

        Example:

        - *Longform transcription*: To transcribe or translate audios longer than 30 seconds, process the audio files without truncation and pass all mel features at once to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset, Audio

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> model.cuda()

        >>> # load audios > 30 seconds
        >>> ds = load_dataset("distil-whisper/meanwhile", "default")["test"]
        >>> # resample to 16kHz
        >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        >>> # take first 8 audios and retrieve array
        >>> audio = ds[:8]["audio"]
        >>> audio = [x["array"] for x in audio]

        >>> # make sure to NOT truncate the input audio, to return the `attention_mask` and to pad to the longest audio
        >>> inputs = processor(audio, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=16_000)
        >>> inputs = inputs.to("cuda", torch.float32)

        >>> # transcribe audio to ids
        >>> generated_ids = model.generate(**inputs)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        >>> transcription[0]
        ' Folks, if you watch the show, you know, I spent a lot of time right over there. Patiently and astutely scrutinizing the boxwood and mahogany chest set of the day's biggest stories developing the central headline pawns, definitely maneuvering an oso topical night to F6, fainting a classic Sicilian, nade door variation on the news, all the while seeing eight moves deep and patiently marshalling the latest press releases into a fisher's shows in Lip Nitsky attack that culminates in the elegant lethal slow-played, all-passant checkmate that is my nightly monologue. But sometimes, sometimes, folks, I. CHEERING AND APPLAUSE Sometimes I startle away, cubside down in the monkey bars of a condemned playground on a super fun site. Get all hept up on goofballs. Rummage that were discarded tag bag of defective toys. Yank out a fist bowl of disembodied doll limbs, toss them on a stained kid's place mat from a defunct dennies. set up a table inside a rusty cargo container down by the Wharf and challenged toothless drifters to the godless bughouse blitz of tournament that is my segment. Meanwhile!'
        ```

        - *Shortform transcription*: If passed mel input features are < 30 seconds, the whole audio will be transcribed with a single call to generate.

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```

        """
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )
        # 1. copy generation config
        if generation_config is None:
            generation_config = copy.deepcopy(self.generation_config)
        else:
            generation_config = copy.deepcopy(generation_config)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        if is_shortform:
            # warn user of ignored inputs
            self._maybe_warn_unused_inputs(
                condition_on_prev_tokens=condition_on_prev_tokens,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                total_input_frames=total_input_frames,
            )

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            is_shortform=is_shortform,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        # raise error if prompt_ids are provided
        # PBAWhisper already sets the prompt with detected keywords
        if prompt_ids != None:
            raise ValueError(f'PBAWhisper: you can not provide prompt_ids to the generate method.')
        if is_shortform:
            if input_features.size(dim=0) != 1:
                raise ValueError(f'PBAWhisper: you can not pass audios with duration of at most 30 seconds in-batch.')
            # obtain tokens corresponding to identified keywords
            prompt_ids = keyword_spotting(input_features=input_features, start_of_prev=True)[0]
        # pass self.config for backward compatibility
        self._set_forced_decoder_ids(
            task=task,
            language=language,
            prompt_ids= prompt_ids if (prompt_ids is not None and len(prompt_ids) != 0) else None,
            generation_config=generation_config,
            config=self.config,
            kwargs=kwargs,
        )
        self._set_token_ids(generation_config=generation_config, config=self.config, kwargs=kwargs)
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )

        # 4. Retrieve logits processors
        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            no_speech_threshold=no_speech_threshold,
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
        )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            if temperature is not None:
                kwargs["temperature"] = temperature
            
            outputs = super(WhisperGenerationMixin, self).generate(
                input_features,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            return outputs[:, len(prompt_ids):]

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames
        )
        init_tokens = self._retrieve_init_tokens_from_forced_decoder_ids(generation_config=generation_config)

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = [[] for _ in range(batch_size)]
        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # obtain tokens corresponding to identified keywords
            keywords_tokens = keyword_spotting(input_features=segment_input)

            # 6.5 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )
            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                keywords_tokens=keywords_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                generation_config=generation_config,
                config=self.config,
                device=segment_input.device,
                suppress_tokens=suppress_tokens,
                kwargs=kwargs,
            )

            # 6.6 set max new tokens or max length
            kwargs = self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
                kwargs=kwargs,
            )

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                kwargs=kwargs,
            )

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            for i, seek_sequence in enumerate(seek_sequences):
                prev_i = batch_idx_map[i]

                if should_skip[i]:
                    seek[prev_i] += seek_num_frames[prev_i]
                    continue

                segments, segment_offset = self._retrieve_segment(
                    seek_sequence=seek_sequence,
                    seek_outputs=seek_outputs,
                    time_offset=time_offset,
                    timestamp_begin=timestamp_begin,
                    seek_num_frames=seek_num_frames,
                    time_precision=time_precision,
                    input_stride=input_stride,
                    prev_idx=prev_i,
                    idx=i,
                )

                current_segments[prev_i] += segments
                seek[prev_i] += segment_offset

        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        sequences = _pad_to_max_length(current_segments, generation_config.pad_token_id, padding="right")

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        if return_segments:
            return {"sequences": sequences, "segments": current_segments}

        return sequences
    
    # adapted from _prepare_decoder_input_ids from https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/models/whisper/generation_whisper.py#L1307C4-L1307C36
    @staticmethod
    def _prepare_decoder_input_ids(
        cur_bsz,
        init_tokens,
        keywords_tokens,
        current_segments,
        batch_idx_map,
        do_condition_on_prev_tokens,
        generation_config,
        config,
        device,
        suppress_tokens,
        kwargs,
    ):
        cut_off_length = config.max_target_positions // 2 - 1

        one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
        decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)

        prev_start_of_text = getattr(generation_config, "prev_sot_token_id", None)
        if prev_start_of_text is None:
            prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None

        #keywords_tokens = [[]]
        # process keywords_tokens
        if any(do_condition_on_prev_tokens) and any([len(t_) > 0 for t_ in keywords_tokens]):
            cut_off_length_keywords = (cut_off_length * 3) // 4 - 1
            keywords_tokens = _pad_to_max_length(
                [[{'tokens': torch.tensor(t_).type_as(decoder_input_ids)}] for t_ in keywords_tokens],
                generation_config.pad_token_id,
                padding="left",
                bos_token_tensor=None,
                cut_off_length=cut_off_length_keywords,
            )
        elif any([len(t_) > 0 for t_ in keywords_tokens]):
            cut_off_length_keywords = cut_off_length - 1
            keywords_tokens = _pad_to_max_length(
                [[{'tokens': torch.tensor(t_).type_as(decoder_input_ids)}] for t_ in keywords_tokens],
                generation_config.pad_token_id,
                padding="left",
                bos_token_tensor=None,
                cut_off_length=cut_off_length_keywords,
            )
        else:
            cut_off_length_keywords = 0
            keywords_tokens = torch.tensor(keywords_tokens).type_as(decoder_input_ids)

        if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
            # according to https://github.com/openai/whisper/blob/e58f28804528831904c3b6f2c0e473f346223433/whisper/decoding.py#L609
            active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]

            prev_tokens = _pad_to_max_length(
                active_segments,
                generation_config.pad_token_id,
                padding="left",
                bos_token_tensor=None,
                cut_off_length=cut_off_length-keywords_tokens.size(dim=1)-1,
            )            
        else:
            prev_tokens = torch.tensor([[]] * cur_bsz).type_as(decoder_input_ids)            

        if keywords_tokens.size(dim=1) > 0 or prev_tokens.size(dim=1) > 0:
            prev_start_of_text = getattr(generation_config, "prev_bos_token_id", None) or prev_start_of_text
            bos_token_tensor = (prev_start_of_text * one_tensor)
            decoder_input_ids = torch.cat([bos_token_tensor, keywords_tokens, prev_tokens, decoder_input_ids], dim=-1)
            kwargs["decoder_attention_mask"] = decoder_input_ids != generation_config.pad_token_id
        else:
            # make sure `"decoder_attention_mask"` is not passed to forward
            kwargs.pop("decoder_attention_mask", None)

        return decoder_input_ids, kwargs