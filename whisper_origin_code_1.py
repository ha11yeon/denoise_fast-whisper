import whisper
import time

model_size = ['tiny', 'tiny', 'base', 'small', 'medium', 'large']
for model_name in model_size:
    model = whisper.load_model(model_name)
    start = time.time()
    result = model.transcribe("test2.wav")
    print(f'{model_name} 소요 시간: {time.time()-start}')
    print(result["text"])
    print('')
