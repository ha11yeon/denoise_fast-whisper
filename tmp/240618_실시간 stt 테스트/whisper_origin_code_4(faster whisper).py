import os
import time
from faster_whisper import WhisperModel

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
model_size = "tiny"


# Run on GPU with FP16
# model = WhisperModel(model_size, device="cpu", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")


segments, info = model.transcribe("RealTime_STT_with_Whisper/test_audio/0.wav", beam_size=5, word_timestamps=True, no_speech_threshold=0.4)
# for segment in segments: print(segment, end='\n\n')





# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))


dic_list = []
for segment in segments:
    no_speech_prob = segment.no_speech_prob
    words_with_timestamps = []
    for word in segment.words:
        _word = word.word
        _start = word.start
        _end = word.end
        dic_list.append([_word, _start, _end, no_speech_prob])

print(dic_list)
