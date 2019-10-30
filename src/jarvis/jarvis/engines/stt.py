# MIT License

# Copyright (c) 2019 Georgios Papachristou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import logging
import wave
import abc
import traceback
from pprint import pprint

import pyaudio
import grpc
import speech_recognition as sr

from jarvis.utils.console_utils import user_input, clear
from jarvis.engines.recognizer import UnknownValueError
from jarvis.settings import SPEECH_RECOGNITION

from vernacular.vernacular import KaldiServeClient
from vernacular.vernacular_pb2 import RecognitionConfig, RecognizeRequest, RecognitionAudio
from vernacular.vernacular_pb2_grpc import KaldiServeStub


class STTEngine(abc.ABC):
    @abc.abstractmethod
    def recognize_input(self):
        pass


class STTGoogleEngine(STTEngine):
    """
    Speech To Text Engine (STT)
    """

    def __init__(self, pause_threshold, energy_theshold, ambient_duration,
                 dynamic_energy_threshold):
        self.logger = logging
        self.sr = sr
        self.speech_recognizer = sr.Recognizer()
        self.dynamic_energy_ratio = self.speech_recognizer.dynamic_energy_ratio
        self.dynamic_energy_threshold = self.speech_recognizer.energy_threshold
        self.microphone = self._set_microphone(
            pause_threshold=pause_threshold,
            energy_threshold=energy_theshold,
            ambient_duration=ambient_duration,
            dynamic_energy_threshold=dynamic_energy_threshold)

    def recognize_input(self):
        """
        Capture the words from the recorded audio (audio stream --> free text).
        """
        audio_text = self._record()
        try:
            voice_transcript = self.speech_recognizer.recognize_google(
                audio_text).lower()
            self.logger.debug('Recognized words: ' + voice_transcript)
            return voice_transcript
        except self.sr.UnknownValueError:
            self.logger.info('Not recognized text')
        except self.sr.RequestError:
            self.logger.info("Google API was unreachable.")

    def _record(self):
        """
        Capture the user speech and transform it to audio stream (speech --> audio stream --> text).
        """
        self._update_microphone_noise_level()

        with self.microphone as source:
            audio_text = self.speech_recognizer.listen(source)
        return audio_text

    def _update_microphone_noise_level(self):
        """
        Update microphone variables in assistant state.
        """
        self.dynamic_energy_ratio = self.speech_recognizer.dynamic_energy_ratio  # Update dynamic energy ratio
        self.energy_threshold = self.speech_recognizer.dynamic_energy_threshold  # Update microphone energy threshold

        self.logger.debug("Dynamic energy ration value is: {0}".format(
            self.dynamic_energy_ratio))
        self.logger.debug("Energy threshold is: {0}".format(
            self.energy_threshold))

    def _set_microphone(self, pause_threshold, energy_threshold,
                        ambient_duration, dynamic_energy_threshold):
        """
        Setup the assistant microphone.
        """
        microphone_list = self.sr.Microphone.list_microphone_names()

        clear()
        print("=" * 48)
        print("Microphone Setup")
        print("=" * 48)
        print("Which microphone do you want to use a assistant mic:")

        for index, name in enumerate(microphone_list):
            print("{0}) Microphone: {1}".format(index, name))

        choices = "Choice[1-{0}]: ".format(len(microphone_list))
        print(
            "WARNING: "
            "In case of error of 'Invalid number of channels' try again with different micrphone choice"
        )
        index = input(choices)

        while not index.isnumeric():
            index = input(
                "Please select a number between choices[1-{0}]: ".format(
                    len(microphone_list)))

        with self.sr.Microphone(
                device_index=int(index), chunk_size=512) as source:
            self.speech_recognizer.pause_threshold = pause_threshold
            self.speech_recognizer.energy_threshold = energy_threshold

            clear()
            print("-" * 48)
            print("Microphone Calibration")
            print("-" * 48)

            print("Please wait.. for {} seconds ".format(ambient_duration))
            self.speech_recognizer.adjust_for_ambient_noise(
                source, duration=ambient_duration)
            self.speech_recognizer.dynamic_energy_threshold = dynamic_energy_threshold
            print("Microphone calibrated successfully!")

            return source


class STTVernacularEngine(STTEngine):
    def __init__(self, *args, **kwargs):
        self.client = KaldiServeClient()

    def recognize_input(self):
        """
        Capture the words from the recorded audio (audio stream --> free text).
        """
        audio_chunks = self._record()
        response = {}
        encoding = RecognitionConfig.AudioEncoding.LINEAR16

        try:
            print('streaming raw')
            config = lambda chunk_len: RecognitionConfig(
                sample_rate_hertz=SPEECH_RECOGNITION['sample_rate'],
                encoding=encoding,
                language_code=SPEECH_RECOGNITION['language_code'],
                max_alternatives=5,
                model=SPEECH_RECOGNITION['model'],
                raw=True,
                data_bytes=chunk_len
            )
            audio_params = [(config(len(chunk)),
                             RecognitionAudio(content=chunk))
                            for chunk in audio_chunks]
            response = self.client.streaming_recognize_raw(
                audio_params, uuid="")
        except Exception as e:
            traceback.print_exc()

        output = self._parse_response(response)
        return output[0][0]['transcript']

    def _record(self):
        """
        Generate wave audio chunks from microphone worth `secs` seconds.
        """

        p = pyaudio.PyAudio()
        sample_format = pyaudio.paInt16
        sample_rate = SPEECH_RECOGNITION['sample_rate']
        channels = SPEECH_RECOGNITION['num_channels']

        # This is in samples not seconds
        chunk_size = 4000

        stream = p.open(
            format=sample_format,
            channels=channels,
            rate=sample_rate,
            frames_per_buffer=chunk_size,
            input=True)

        sample_width = p.get_sample_size(sample_format)

        for _ in range(
                0,
                int(sample_rate / chunk_size *
                    SPEECH_RECOGNITION['max_audio_length'])):
            yield self._raw_bytes_to_wav(
                stream.read(chunk_size), sample_rate, channels, sample_width)

        stream.stop_stream()
        stream.close()
        p.terminate()

    @staticmethod
    def _raw_bytes_to_wav(data: bytes, frame_rate: int, channels: int,
                          sample_width: int) -> bytes:
        """
        Convert raw PCM bytes to wav bytes (with the initial 44 bytes header)
        """

        out = io.BytesIO()
        wf = wave.open(out, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        wf.writeframes(data)
        wf.close()
        return out.getvalue()

    @staticmethod
    def _parse_response(response):
        output = []

        for res in response.results:
            output.append([{
                "transcript": alt.transcript,
                "confidence": alt.confidence,
                "am_score": alt.am_score,
                "lm_score": alt.lm_score
            } for alt in res.alternatives])
        return output
