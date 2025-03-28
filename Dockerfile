FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update
RUN apt-get install -y gcc libportaudio2

COPY ./requirements.txt ./requirements.txt

RUN pip3 install torch torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir -r ./requirements.txt

RUN apt-get remove -y gcc

COPY . .

# Runs on port 5005
ENTRYPOINT ["python", "app.py"]

