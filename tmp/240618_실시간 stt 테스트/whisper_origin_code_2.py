import whisper


model_size = ['tiny', 'base', 'small', 'medium', 'large']
model = whisper.load_model(model_size[1])

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("RealTime_STT_with_Whisper/test_audio/test3.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(without_timestamps=False)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

print(f'mel shape: {mel.shape}')
print(f'mel dtype: {mel.dtype}')