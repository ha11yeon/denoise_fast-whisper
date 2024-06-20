# RealTime_STT_with_Whisper

OpenAI의 Whisper 모델과 VAD(Voice Activity Detector)를 사용해 구현한 실시간 STT 모델

## 240618_v1
240618 실시간 STT 테스트

[문제점]
- tmp.wav를 저장해서 불러오는 방식이라 그 과정에서 계속 알 수 없는 에러가 발생(읽는 타이밍이 꼬이는듯)
- 마지막 단어를 배출하지 못하는 특성 있음. 그리고 10초가 지나면 잊혀짐
- wav파일을 넣었을 때랑, tmp.wav를 실시간으로 불러왔을 때랑 추론 결과가 다른 것 같음. 전자가 성능이 더 좋은 것 같음

## 240619_v2
GUI로 STT 테스트
- faster_whisper로 cpu통해서 tiny, base 모델 테스트 가능하게 gui 구성
- exe파일 생성

## 240619_v3
노이즈 제거 기능 테스트 중
- main.py기능은 안만짐
- '잡음 제거 테스트.ipynb' 하나 만들어서 잡음 제거 기능 테스트 중임