# RealTime_STT_with_Whisper

OpenAI의 Whisper 모델과 VAD(Voice Activity Detector)를 사용해 구현한 실시간 STT 모델

## 프로젝트 개요
이 프로젝트는 OpenAI의 Whisper 모델을 사용하여 음성을 텍스트로 변환하는 실시간 STT(Speech-to-Text) 시스템을 구현한 것입니다. VAD(Voice Activity Detector)를 통해 자동으로 녹음을 중지하는 기능도 포함되어 있습니다.

## 주요 기능
- **녹음 시작**: 음성 활동 감지 모델을 통해 자동으로 녹음을 시작하고, 1초 동안 음성이 없으면 자동으로 녹음을 중지합니다.
- **음성 로드**: 사전에 저장된 음성 파일을 로드하여 STT 기능을 수행할 수 있습니다.
- **모델 적용**: 다양한 Whisper STT 모델을 적용할 수 있으며, 실시간에 유리한 tiny, base 모델을 추천합니다.
- **STT 모델 추론**: 녹음되거나 로드된 음성을 STT하여 텍스트로 변환합니다.
- **출력창 리셋**: 텍스트 출력 창을 초기화합니다.
- **주변 소음 조절 초기화**: 주변의 마이크 소음을 조절하여 마이크를 최적화합니다.
- **노이즈 감도 조절**: 0에서 1 사이로 디노이즈 감도를 조절할 수 있습니다.

## 설치 및 실행 방법
이 프로젝트를 로컬에서 실행하려면 아래 단계를 따르세요.

### 1. 환경 설정
- Windows11 Anaconda python 3.10 환경 기준으로 설명합니다. 다른 환경에 설치할 경우 참고 바랍니다.
- 먼저 다음 링크에서 "Microsoft Build Tools"를 설치하십시오: https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
- "C++를 사용한 데스크톱 개발"만 설치하면 됩니다.
- 이후 아나콘더 터미널을 열고 아래와 같이 설치를 진행합니다.

```bash
conda create -n whisper python=3.10
y
conda activate whisper
pip install -r requirements.txt
```

## 모듈 구동 방법
```bash
python main.py
```

## 라이센스  
아래 라이브러리들의 라이센스를 포함합니다.  
pydub, pyaudio, faster-whisper, SpeechRecognition, noisereduce, webrtcvad  

## 참고 git-hub
- https://github.com/timsainb/noisereduce
- https://github.com/wiseman/py-webrtcvad
- https://pypi.org/project/SpeechRecognition/
- https://github.com/kafa46/acin_academy/tree/master/202_fine_tunning/whisper (fine-tuning)

  

  
