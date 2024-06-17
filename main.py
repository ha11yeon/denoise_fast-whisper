import pyaudio
import wave
import webrtcvad
import os
import glob
import time

class AudioRecorder:
    def __init__(self, rate=16000, chunk_duration_ms=20, vad_sensitivity=1, output_folder="tmp", min_speech_duration_ms=500):
        self.rate = rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(rate * chunk_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_sensitivity)
        self.output_folder = output_folder
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.speech_started = False
        self.file_counter = 0
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_start_time = None

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        # Clear existing .wav files in the output directory
        self.clear_output_folder()

    def clear_output_folder(self):
        files = glob.glob(os.path.join(self.output_folder, "*.wav"))
        for f in files:
            os.remove(f)

    def start_stream(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def stop_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def save_wav(self):
        filename = os.path.join(self.output_folder, f"{self.file_counter}.wav")
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.file_counter += 1

    def is_speech(self, data):
        return self.vad.is_speech(data, self.rate)

    def run(self):
        self.start_stream()
        try:
            while True:
                data = self.stream.read(self.chunk_size)
                if self.is_speech(data):
                    if not self.speech_started:
                        self.speech_start_time = time.time()
                        self.speech_started = True
                        print("Speech started")
                    self.frames.append(data)
                elif self.speech_started:
                    speech_duration = (time.time() - self.speech_start_time) * 1000  # Convert to milliseconds
                    if speech_duration >= self.min_speech_duration_ms:
                        print("Speech ended, saving file")
                        self.stop_stream()
                        self.save_wav()
                    else:
                        print("Speech ended, but too short to save")
                    self.frames = []
                    self.speech_started = False
                    self.start_stream()
        except KeyboardInterrupt:
            self.stop_stream()
        finally:
            self.audio.terminate()
            print(f"Recording stopped. Audio saved to {self.output_folder}")

if __name__ == "__main__":
    recorder = AudioRecorder(output_folder="tmp", min_speech_duration_ms=500)
    print("Start speaking...")
    recorder.run()
