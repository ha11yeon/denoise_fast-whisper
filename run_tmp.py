# tmp.wav파일을 인퍼런스하여 터미널에 출력하는 간단한 코드

from main import Cumtom_whisper

stt_model = Cumtom_whisper()
stt_model.set_model('tiny') # 기본 설정
_, result_txt = stt_model.run('tmp.wav')
print(result_txt)


