'''
240618 실시간 STT 테스트

[문제점]
- tmp.wav를 저장해서 불러오는 방식이라 그 과정에서 계속 알 수 없는 에러가 발생(읽는 타이밍이 꼬이는듯)
- 마지막 단어를 배출하지 못하는 특성 있음. 그리고 10초가 지나면 잊혀짐
- wav파일을 넣었을 때랑, tmp.wav를 실시간으로 불러왔을 때랑 추론 결과가 다른 것 같음. 전자가 성능이 더 좋은 것 같음
'''


import pyaudio
import wave
import os
import threading
import shutil

class Audio_streaming:
    def __init__(self, sr=32000, save_sec=10):
        '''
        마이크를 실시간으로 입력받아 wav파이롤 저장해주는 기능

        sr : 샘플레이트
        save_sec : 순차적으로 저장될 wav파일의 길이(초)
        overlap_sec : 오버랩될 길이(초)
        save_path : 저장 경로
        '''
        self.save_sec = save_sec
        self.chunk = sr # chunk 1개는 sr레이트랑 동일하다. 1초라는 의미
        self.buffer = [] # chunk를 쌓아두는 리스트
        self.lost_secs = 0 # buffer에서 실시간으로 버리는 chunk 개수 기록
        self.streaming = True # False가 되면 multi-thread들이 종료된다

        # 오디오 관련 선언
        self.sr = sr
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.sr, input=True, frames_per_buffer=self.chunk)

    def run(self):
        print('오디오 스트리밍, tmp.wav 저장 시작')
        threading.Thread(target=self._run).start()

    def stop(self):
        '''
        스트리밍 중지
        '''
        self.streaming = False


    def save_buffer(self):
        '''
        buffer에 있는 최근의 n초를 저장한다. 계산을 쉽게 하기 위하여 1개의 chunk는 무조건 1초로 한다. 그래서 n초는 buffer에서 n개의 원소를 뜻한다.
        '''
        # buffer가 n초 이상을 넘어가지 않게 관리한다.
        if len(self.buffer) > self.save_sec:
            self.lost_secs += len(self.buffer) - self.save_sec
            self.buffer = self.buffer[-self.save_sec:]
        # 최근의 n초를 저장한다
        self._frames_to_wav(self.buffer)

        # 버린 누적 chunk와 방금 저장한 buffer의 길이를 반환
        return self.lost_secs, len(self.buffer)


    def _run(self):
        '''
        스트리밍하여 buffer에 지속적으로 음성 chunk를 추가만 하는 쓰레드
        '''
        while self.streaming:
            one_chunk = self.stream.read(self.chunk)
            self.buffer.append(one_chunk)
        print('\n스트리밍 종료')

    def _frames_to_wav(self, frames):
        '''
        입력된 buffer안의 원소들을 join하여 wav로 저장
        '''
        # 기존 파일 삭제
        wav_name = 'tmp.wav'
        if os.path.exists(wav_name):
            # 다른 프로세스에서 파일을 사용중이라는 에러가 발생하여 try문으로 무한 시도하게 개조
            while True:
                try: os.remove(wav_name); break
                except: continue
        # 새로운 wav 저장
        with wave.open(wav_name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sr)
            wf.writeframes(b''.join(frames))


import os
from faster_whisper import WhisperModel
import time

class STT_faster_whisper:
    def __init__(self, model_size):
        '''
        최대 4배 빠른 faster whisper를 사용하여 cpu로 저장된 wav파일에 STT 수행
        
        model_size : tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
        read_path : wav가 저장되어있는 폴더 경로
        '''
        # 환경 설정(Window 아나콘다 환경에서 아래 코드 실행 안하면 에러남)
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except Exception as e: print(f'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except Exception as e: print(f'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        # 모델 선언
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    def run(self, wav_path, last_del=False):
        '''
        wav 경로를 입력하면 txt로 변경해주고, 각 단어에 대한 time stamp를 반환함

        wav_path : STT를 수행할 wav파일 full 경로
        last_del : STT된 마지막 word를 지울지. 마지막 word는 끊겼을 수 있다고 가정하기 때문
        '''
        start = time.time()
        # 인퍼런스
        segments, info = self.model.transcribe(wav_path, beam_size=5, word_timestamps=True, language='ko')

        # 결과 후처리
        dic_list = []
        for segment in segments:
            if segment.no_speech_prob > 0.6: continue # 말을 안했을 확률이 크다고 감지되면 무시
            for word in segment.words:
                _word = word.word
                _start = round(word.start, 2)
                _end = round(word.end, 2)
                dic_list.append([_word, _start, _end])
        self.time = round(time.time()-start, 2)
        # 마지막 word 삭제 옵션 적용
        if last_del == True and len(dic_list) > 0:
            del dic_list[-1]
        return dic_list
    
import time

class Realtime_stt:
    def __init__(self, model_size):
        '''
        실시간으로 마이크에서 음성을 저장하는 동시에, 꺼내와서 STT해주는 모듈
        핵심 기술:
        - 실시간으로 overlap 하여 저장
        - 선입선출로 음성을 가져와 STT 추론
        - 추론된 결과를 바탕으로 time stamp 기준으로 통합
        - 추론된 결과 실시간 제공
        '''
        self.model_size = model_size
        self.streaming = True

        self.total_dic_list = [] # 모든 텍스트 히스토리를 저장함
        self.stt_model = STT_faster_whisper(model_size)
        self.audio_streaming = Audio_streaming()
        self.txt_log = ''
        
    def run(self):
        self.audio_streaming.run()
        threading.Thread(target=self._run).start()
        
    def _run(self):
        while self.streaming:
            # 최근 오디오(최대10초) 저장
            lost_secs, buffer_len = self.audio_streaming.save_buffer()
            # 저장된 오디오 STT(마지막 word 제외)
            new_dic_list = self.stt_model.run('tmp.wav', last_del=True)
            updated_dic_list = self._time_update(new_dic_list, lost_secs)
            # total_dic_list에 새로운 텍스트 중복 제거 병합
            self.total_dic_list = self._murge_dic_list(self.total_dic_list, updated_dic_list)
            # 결과 출력
            result_txt = self._get_txt_from_dic_list(self.total_dic_list)
            self._txt_out(result_txt)

        
    def _time_update(self, new_dic_list, lost_secs):
        '''
        new_dic_list의 start, end 값들을 실제 처럼 업데이트
        '''
        updated_dic_list = []
        for dic in new_dic_list:
            dic[1] += lost_secs
            dic[2] += lost_secs
            updated_dic_list.append(dic)
        return updated_dic_list
            
    def _murge_dic_list(self, total_dic_list, new_dic_list):
        '''
        time stamp를 확인하여 두 stt결과를 중복 제거하여 병합
        '''
        # 마지막 단어가 끝나는 시점 가져오기
        if len(total_dic_list) > 0:
            last_end = total_dic_list[-1][2]
        else:
            last_end = 0
        # 새로운 리스트 병합하기
        for dic in new_dic_list:
            # 추가 조건 확인
            if dic[1] >= last_end: # dic 데이터 예시: [word, start, end]
                total_dic_list.append(dic)
            else:
                continue
        return total_dic_list
            
    def _get_txt_from_dic_list(self, dic_list):
        '''
        dic_list에서 txt만 뽑아서 반환
        '''
        txt = ''
        for dic in dic_list:
            new_txt = dic[0]
            txt = f'{txt}{new_txt}'
        return txt
    
    def _txt_out(self, txt):
        '''
        전체 텍스트를 출력 요청하면, 지금까지 출력된 텍스트를 제외하고 출력
        '''
        new_txt = txt[len(self.txt_log):]
        if len(new_txt) > 0:
            print(new_txt, end='')
        self.txt_log = txt

    def stop(self):
        '''
        멀티쓰레드로 구동되는 스트리밍 로직을 중지
        '''
        print('프로세스 중지 중...')
        self.audio_streaming.stop()
        self.streaming = False

realtime_stt = Realtime_stt(model_size='base')
realtime_stt.run()