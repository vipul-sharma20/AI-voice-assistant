FROM python:3.5

RUN mkdir -p /home/assistant/
WORKDIR /home/assistant/

COPY . /home/assistant/

RUN apt-get update
RUN apt-get install -y portaudio19-dev python-pyaudio python3-pyaudio libasound2-plugins libsox-fmt-all libsox-dev sox ffmpeg espeak
RUN pip install -r requirements.txt

CMD ["python", "src/jarvis/start.py"]

