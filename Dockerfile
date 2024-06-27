FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y libgl1
RUN apt-get install -y cmake
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
