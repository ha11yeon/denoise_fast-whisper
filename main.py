# 기본 
import os
import threading
import time
import wave
import tkinter as tk
from tkinter import filedialog

# 추가
from faster_whisper import WhisperModel
import speech_recognition as sr
import noisereduce as nr
import numpy as np

class Audio_record:
    def __init__(self):
        '''
        요청 받았을 때 오디오를 스트리밍 하여 원하는 만큼 녹음하여 디노이즈
        '''
        # 기본 선언
        self.chunk_size = 300
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=16000, chunk_size=self.chunk_size)
        self.buffer = []
        self.recording = False
        
        # 주변 소음 조정
        self.adjust_noise()

        print('Audio_record 초기화 성공')
        

    def adjust_noise(self):
        '''
        주변 소음 조정
        '''
        with self.microphone as source:
            print('주변 소음에 맞게 조정 중...')
            self.recognizer.adjust_for_ambient_noise(source)
            self.recognizer.energy_threshold += 100

    def record_start(self):
        '''녹음이 시작되는 함수'''
        if self.recording == False:
            self.record_thread = threading.Thread(target=self._record_start)
            self.record_thread.start()
    
    def _record_start(self):
        self.recording = True
        self.buffer = []
        with self.microphone as source:
            while self.recording:
                self.buffer.append(source.stream.read(self.chunk_size))
        
    def record_stop(self, denoise_value):
        '''녹음이 종료되고 디노이징 과정을 거치는 함수'''
        # thread 종료하고, 끝날 때 까지 join으로 대기
        self.recording = False
        self.record_thread.join()
        # 버퍼를 하나의 오디오 데이터로 결합
        audio_data = np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        sample_rate = self.microphone.SAMPLE_RATE
        return self._denoise_process(audio_data, sample_rate, denoise_value)

    
    def load_wav(self, path, denoise_value):
        '''wav파일을 불러와 디노이징 과정을 거치는 함수'''
        buffer = []
        with wave.open(path, 'rb') as wf:
            chunk_size = self.chunk_size
            data = wf.readframes(chunk_size)
            while data:
                buffer.append(data)
                data = wf.readframes(chunk_size)

        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        sample_rate = wf.getframerate()
        return self._denoise_process(audio_data, sample_rate, denoise_value)

    
    def _denoise_process(self, audio_data, sample_rate, denoise_value):
        '''
        오디오를 받아 디노이징을 적용하고, 원본과 디노이즈값둘 둘 다 저장하고 반환한다.
        
        audio_data : int16 np 형식 오디오 데이터. chunk를 append하여 만들어진 buffer를 다음과 같이 처리한 예시) np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        sample_rate : 샘플 레이트 입력
        denoise_value : 디노이즈 적용값 설정
        
        return: {'audio_denoise': audio_denoise, 'audio_noise': audio_noise, 'sample_rate': sample_rate}
        '''
        # 1. 노이즈 감소 처리
        denoise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=denoise_value)
        buffer_denoise = [denoise.tobytes()] # 데이터를 다시 버퍼로 변환
        # 2. 노이즈 감소 없이
        noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.0)
        buffer_noise = [noise.tobytes()] # 데이터를 다시 버퍼로 변환
        
        # 1. 노이즈 감소 파일 저장
        self._save_buffer_to_wav(buffer_denoise, self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH, 'input_denoise.wav')
        # 2. 노이즈 감소 없는 파일 저장
        self._save_buffer_to_wav(buffer_noise, self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH, 'input_noise.wav')
        
        # 오디오 소스 파일로 return
        audio_denoise = self._buffer_to_numpy(buffer_denoise, self.microphone.SAMPLE_RATE)
        audio_noise = self._buffer_to_numpy(buffer_noise, self.microphone.SAMPLE_RATE)

        return {'audio_denoise':audio_denoise, 'audio_noise':audio_noise, 'sample_rate':self.microphone.SAMPLE_RATE}


    def _buffer_to_numpy(self, buffer, sample_rate):
        '''buffer를 입력하면 whisper에서 추론 가능한 입력 형태의 오디오로 반환'''
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Convert to float32        
        return audio_data
        

    def _save_buffer_to_wav(self, buffer, sample_rate, sample_width, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(buffer))
        
class Cumtom_faster_whisper:
    def __init__(self):
        '''
        최대 4배 빠른 faster whisper를 사용하여 cpu로 저장된 wav파일에 STT 수행
        '''
        # 환경 설정(Window 아나콘다 환경에서 아래 코드 실행 안하면 에러남)
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except Exception as e: print(f'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except Exception as e: print(f'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')
        print('Cumtom_faster_whisper 초기화 성공')

    def set_model(self, model_name):
        '''
        모델 설정
        '''
        model_list = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large']
        if not model_name in model_list:
            model_name = 'base'
            print('모델 이름 잘못됨. base로 설정. 아래 모델 중 한가지 선택')
            print(model_list)
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        return model_name

    def run(self, audio, language=None):
        '''
        저장된 tmp.wav를 불러와서 STT 추론 수행

        audio : wav파일의 경로 or numpy로 변환된 오디오 파일 소스
        language : ko, en 등 언어 선택 가능. 선택하지 않으면 언어 분류 모델 내부적으로 수행함
        '''
        start = time.time()
        # 추론

        segments, info = self.model.transcribe(audio, beam_size=5, word_timestamps=True, language=language)
        # 결과 후처리
        dic_list = []
        for segment in segments:
            if segment.no_speech_prob > 0.6: continue # 말을 안했을 확률이 크다고 감지되면 무시
            for word in segment.words:
                _word = word.word
                _start = round(word.start, 2)
                _end = round(word.end, 2)
                dic_list.append([_word, _start, _end])
        # 시간 계산
        self.spent_time = round(time.time()-start, 2)
        
        # 텍스트 추출
        result_txt = self._make_txt(dic_list)
        print(result_txt)
        return dic_list, result_txt, self.spent_time

    def _make_txt(self, dic_list):
        '''
        [word, start, end]에서 word만 추출하여 txt로 반환
        '''
        result_txt = ''
        for dic in dic_list:
            txt = dic[0]
            result_txt = f'{result_txt}{txt}'
        return result_txt

class Gui:
    def __init__(self):
        '''
        GUI로 다양한 버튼 인터페이스 구성
        '''
        self.running = True # 모든 multi thread를 정지시키기 위한 플래그
        self.model_name = 'base'

        # model, 음성 녹음 관련 초기화
        self.audio_record = Audio_record()
        self.model = Cumtom_faster_whisper()
        self.model.set_model(self.model_name)

        # gui 관련 초기화
        self._init_gui()
        

    def _start_record(self):
        '''녹음 시작'''
        self.audio_record.record_start()

    def _stop_record(self):
        '''녹음 중지'''
        self.audio_dic = self.audio_record.record_stop(self.denoise_slider.get())

    def _load_wav(self):
        '''음성파일 불러오기'''
        filename = filedialog.askopenfilename()
        self.audio_dic = self.audio_record.load_wav(filename, self.denoise_slider.get())

    def _model_set(self):
        '''모델 적용'''
        # 텍스트 박스에서 사용자가 입력한 모델 이름 취득
        model_name = self.text_box_model_name.get('1.0', tk.END).strip()
        # 모델 적용 후 필터링된 모델 이름 취득
        model_name = self.model.set_model(model_name)
        # 필터링된 모델 이름 텍스트 박스에 다시 적용
        self.text_box_model_name.delete('1.0', tk.END)
        self.text_box_model_name.insert(tk.END, model_name)

    def _model_run(self):
        '''stt 모델 추론'''
        # 모델 추론
        _, result_denoise, time_denoise = self.model.run(self.audio_dic['audio_denoise'])
        _, result_noise, time_noise = self.model.run(self.audio_dic['audio_noise'])
        # 텍스트 구성
        txt_denoise = f'{result_denoise} ({time_denoise}s)'
        txt_noise = f'{result_noise} ({time_noise}s)'

        # 텍스트 박스 출력
        self._txtbox_insert(self.text_box_denoise, txt_denoise)
        self._txtbox_insert(self.text_box_normal, txt_noise)

    def _txtbox_insert(self, txt_box, txt):
        txt_box.insert(tk.END, f'>> {txt}\n')

    def _reset_output(self):
        self.text_box_normal.delete('1.0', tk.END)
        self.text_box_denoise.delete('1.0', tk.END)

    def _reset_ambient_noise(self):
        '''주변 소음 감소'''
        self.audio_record.adjust_noise()

    def _on_closing(self):
        self.running = False
        print('프로그램이 종료됩니다')
        self.root.destroy()

    def _init_gui(self):
        '''
        GUI로 다양한 버튼 인터페이스 구성
        '''
        # 메인 윈도우 설정
        self.root = tk.Tk()
        self.root.title('STT Model v2.0.0')
        self.root.protocol('WM_DELETE_WINDOW', self._on_closing) # 창 종료 관련

        # 버튼 생성
        # 녹음 시작 & 녹음 중지 & 음성 로드 & 모델 적용 버튼
        frame_small = tk.Frame(self.root, pady=20)
        frame_small.pack(fill=tk.X)
        btn_start = tk.Button(frame_small, text="녹음 시작", width=20, command=self._start_record)
        btn_start.pack(side=tk.LEFT, padx=10)
        btn_stop = tk.Button(frame_small, text="녹음 중지", width=20, command=self._stop_record)
        btn_stop.pack(side=tk.LEFT, padx=10)
        btn_load = tk.Button(frame_small, text="음성 로드", width=20, command=self._load_wav)
        btn_load.pack(side=tk.LEFT, padx=10)
        btn_set_model = tk.Button(frame_small, text="모델 적용", width=20, command=self._model_set)
        btn_set_model.pack(side=tk.LEFT, padx=10)

        # 모델 이름 텍스트창
        self.label_model_name = tk.Label(frame_small, text='모델 이름')
        self.label_model_name.pack(side=tk.LEFT, padx=10)
        self.text_box_model_name = tk.Text(frame_small, height=1, width=20)
        self.text_box_model_name.pack(side=tk.LEFT, padx=10)
        self.text_box_model_name.insert(tk.END, self.model_name) # 기본 모델 설정

        # 디노이즈 조절바
        frame_denoise = tk.Frame(self.root, pady=10)
        frame_denoise.pack(fill=tk.X)
        self.label_slide_bar = tk.Label(frame_denoise, text='노이즈 감소 정도 조절 (0 ~ 1)')
        self.label_slide_bar.pack(side=tk.TOP)  # 레이블을 슬라이더 위에 배치
        self.denoise_slider = tk.Scale(frame_denoise, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=300)
        self.denoise_slider.set(1.0)
        self.denoise_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 추론 버튼 & 출력창 리셋 & 주변 소음 조절 초기화 버튼
        frame_middle = tk.Frame(self.root, pady=20)
        frame_middle.pack(fill=tk.X)
        btn_process = tk.Button(frame_middle, text="STT 모델 추론", width=40, command=self._model_run)
        btn_process.pack(side=tk.LEFT, padx=10)
        btn_output_reset = tk.Button(frame_middle, text="출력창 리셋", width=40, command=self._reset_output)
        btn_output_reset.pack(side=tk.LEFT, padx=10)
        btn_ambient = tk.Button(frame_middle, text="주변 소음 조절 초기화", width=40, command=self._reset_ambient_noise)
        btn_ambient.pack(side=tk.LEFT, padx=10)
        
        # 아래 커다란 출력 창 2개
        self.label_normal = tk.Label(self.root, text='원본 음성 추론 결과')
        self.label_normal.pack()
        self.text_box_normal = tk.Text(self.root, height=15, width=120)
        self.text_box_normal.pack(pady=10)
        self.label_denoise = tk.Label(self.root, text='디노이즈 음성 추론 결과')
        self.label_denoise.pack()
        self.text_box_denoise = tk.Text(self.root, height=15, width=120)
        self.text_box_denoise.pack(pady=10)

        # gui 루프 시작
        self.root.mainloop()

gui = Gui()