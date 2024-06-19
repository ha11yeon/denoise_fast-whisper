import whisper
model = whisper.load_model("base")
transcript = model.transcribe(
    word_timestamps=True,
    audio="RealTime_STT_with_Whisper/test_audio/0.wav"
)
for segment in transcript['segments']:
    words_with_timestamps = []
    for word in segment['words']:
        word_with_timestamps = f"{word['word']}[{word['start']}/{word['end']}]"
        words_with_timestamps.append(word_with_timestamps)
    print(''.join(words_with_timestamps))

    
