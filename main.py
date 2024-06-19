# 기본 
import os
import threading
import time
import wave
import tkinter as tk
from tkinter import messagebox

# 추가
import pyaudio # PyAudio
from faster_whisper import WhisperModel # faster_whisper


class Audio_save:
    def __init__(self, path):
        '''
        요청 받았을 때 오디오를 스트리밍 하여 원하는 만큼 저장
        '''
        self.path = path
        self.sr = 16000
        self.chunk = int(self.sr/10)
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.sr, input=True, frames_per_buffer=self.chunk)

    def run(self):
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.start()

    def _run(self):
        '''
        run() thread
        '''
        self.buffer = []
        self.streaming_status = True
        while self.streaming_status:
            one_chunk = self.stream.read(self.chunk)
            self.buffer.append(one_chunk)
        self._save_buffer(self.buffer, self.path)
        
    def stop(self):
        '''
        녹음 중지
        '''
        self.streaming_status = False
        self.run_thread.join()

    def _save_buffer(self, buffer, path):
        # 저장
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sr)
            wf.writeframes(b''.join(buffer))


class Cumtom_whisper:
    def __init__(self):
        '''
        최대 4배 빠른 faster whisper를 사용하여 cpu로 저장된 wav파일에 STT 수행
        
        model_size : tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
        '''
        # 환경 설정(Window 아나콘다 환경에서 아래 코드 실행 안하면 에러남)
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except Exception as e: print(f'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except Exception as e: print(f'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

    def set_model(self, model_name):
        '''
        model_size : tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, or large
        '''
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print(f'STT 모델 변경: {model_name}')

    def run(self, wav_path):
        '''
        저장된 tmp.wav를 불러와서 STT 추론 수행
        '''
        start = time.time()
        # 추론
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
        # 시간 계산
        self.spent_time = round(time.time()-start, 2)
        
        # 텍스트 추출
        result_txt = self._make_txt(dic_list)
        return dic_list, result_txt

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
        # 버튼 관련
        # 메인 윈도우 설정
        self.root = tk.Tk()
        self.root.title('STT Model v1.0.0')
        self.root.geometry('500x500')
        self.root.protocol('WM_DELETE_WINDOW', self._on_closing) # 창 종료 관련

        # 버튼 생성
        w, h = 20, 2
        self.button1 = tk.Button(self.root, text='말하기', command=self._wake_up, width=w, height=h)
        self.button1.pack(pady=10)
        self.button2 = tk.Button(self.root, text='종료', command=self._wake_up_done, width=w, height=h)
        self.button2.pack(pady=10)
        self.button3 = tk.Button(self.root, text='tiny', command=self._set_model_tiny, width=w, height=h)
        self.button3.pack(pady=10)
        self.button4 = tk.Button(self.root, text='base', command=self._set_model_base, width=w, height=h)
        self.button4.pack(pady=10)
        self.button5 = tk.Button(self.root, text='출력 초기화', command=self._initial_txt_box, width=w, height=h)
        self.button5.pack(pady=10)

        # 텍스트 창 생성
        self.text_box = tk.Text(self.root, height=15, width=60)
        self.text_box.pack(pady=10)

        # Audio 관련
        self.audio_save = Audio_save('tmp.wav')

        # STT 관련
        self.stt_model = Cumtom_whisper()
        self.stt_model.set_model('tiny') # 기본 설정
        self._txtbox_insert('STT 모델: tiny')
        
    def run(self):
        '''
        GUI 실행
        '''
        # 메인 루프 실행
        self.root.mainloop()

    def _on_closing(self):
        print('프로그램이 종료됩니다')
        self.root.destroy()
        os.remove('tmp.wav')

    def _wake_up(self):
        self._txtbox_insert('듣는 중...')
        self.audio_save.run()

    def _wake_up_done(self):
        self.audio_save.stop()
        _, result_txt = self.stt_model.run('tmp.wav')
        output_txt = f'{result_txt} <{self.stt_model.spent_time}s>'
        self._txtbox_insert(output_txt)
        print(f'{output_txt}')

    
    def _set_model_tiny(self):
        self.stt_model.set_model('tiny')
        self._txtbox_insert('STT 모델: tiny')

    def _set_model_base(self):
        self.stt_model.set_model('base')
        self._txtbox_insert('STT 모델: base')

    def _initial_txt_box(self):
        self.text_box.delete('1.0', tk.END)

    def _txtbox_insert(self, txt):
        self.text_box.insert(tk.END, f'>> {txt}\n')

if __name__ == '__main__':
    gui = Gui()
    gui.run()