import whisper
import time

model_size = ['tiny', 'tiny', 'base', 'small', 'medium', 'large']
for model_name in model_size:
    model = whisper.load_model(model_name)
    start = time.time()
    result = model.transcribe("RealTime_STT_with_Whisper/test_audio/test3.wav")
    print(f'{model_name} 소요 시간: {time.time()-start}')
    print(result["text"])
    print('')
