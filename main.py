# 기본 
import tkinter as tk
from tkinter import filedialog

# 추가
from utils.audio import Audio_record, Cumtom_faster_whisper
# from utils.LLM import Llama3_ko
from tkinter import Text  # 명시적으로 Text를 가져오기

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
        #self.text_box_denoise = tk.Text(self.root, height=15, width=120)
        


        # gui 관련 초기화
        self._init_gui()

    def _start_record(self):
        '''녹음 시작'''
        self.canvas.itemconfig(self.led, fill='red')
        self.audio_record.record_start()
        self._check_recording()

    def _stop_record(self):
        '''녹음 중지'''
        if self.audio_record.recording:  # 녹음 중일 때만 실행
            print("녹음을 중지합니다...")
            self.audio_dic = self.audio_record.record_stop(self.denoise_slider.get())
            self.canvas.itemconfig(self.led, fill='grey')  # LED 색상 초기화
            print("녹음이 중지되었습니다.")
        else:
            print("녹음이 이미 중지된 상태입니다.")


    def _check_recording(self):
        '''녹음 상태를 체크하고 업데이트'''
        if self.audio_record.recording:
            self.root.after(100, self._check_recording)  # 100ms 후에 다시 _check_recording 호출
        else:
            self.canvas.itemconfig(self.led, fill='green')
            self.audio_dic = self.audio_record.record_stop(self.denoise_slider.get())
            self.canvas.itemconfig(self.led, fill='grey')

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
        #_, result_noise, time_noise = self.model.run(self.audio_dic['audio_noise'])
        # 텍스트 구성
        txt_denoise = f'{result_denoise} ({time_denoise}s)'
        #txt_noise = f'{result_noise} ({time_noise}s)'

        # 텍스트 박스 출력
        self._txtbox_insert(self.text_box_denoise, txt_denoise)
        #self._txtbox_insert(self.text_box_normal, txt_noise)

    def _txtbox_insert(self, txt_box: Text, txt: str):
        '''텍스트 박스에 텍스트 삽입'''
        txt_box.insert(tk.END, f'>> {txt}\n')

    def _reset_output(self):
        #self.text_box_normal.delete('1.0', tk.END)
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

        # 버튼 생성 : 녹음 시작 & LED 캔버스 & 음성 로드 & 모델 적용 버튼
        # 녹음 시작
        frame_small = tk.Frame(self.root, pady=20)
        frame_small.pack(fill=tk.X)
        btn_start = tk.Button(frame_small, text="녹음 시작", width=20, command=self._start_record)
        btn_start.pack(side=tk.LEFT, padx=10)

        # 녹음 종료 버튼
        btn_stop = tk.Button(frame_small, text="녹음 종료", width=20, command=self._stop_record)
        btn_stop.pack(side=tk.LEFT, padx=10)
        # LED 캔버스
        self.canvas = tk.Canvas(frame_small, width=20, height=20)
        self.canvas.pack(side=tk.LEFT, padx=10)
        self.led = self.canvas.create_oval(5, 5, 20, 20) # 바운딩 박스 좌표. 박스 좌표로 타원이 생성된다.
        self.canvas.itemconfig(self.led, fill='grey') # 초기 색깔 grey(red, green, blue, yellow, orange, purple, pink, brown, black, white 사용 가능)
        # 음성 로드
        btn_load = tk.Button(frame_small, text="음성 로드", width=20, command=self._load_wav)
        btn_load.pack(side=tk.LEFT, padx=10)
        # 모델 적용 버튼
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
        self.denoise_slider.set(0.7)
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
        #self.label_normal = tk.Label(self.root, text='원본 음성 추론 결과')
        #self.label_normal.pack()
        #self.text_box_normal = tk.Text(self.root, height=15, width=120)
        #self.text_box_normal.pack(pady=10)
        self.label_denoise = tk.Label(self.root, text='디노이즈 음성 추론 결과')
        self.label_denoise.pack()
        self.text_box_denoise = tk.Text(self.root, height=15, width=120)
        self.text_box_denoise.pack(pady=10)

        # gui 루프 시작
        self.root.mainloop()

gui = Gui()