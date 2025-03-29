FROM pytorch/pytorch@sha256:11691e035a3651d25a87116b4f6adc113a27a29d8f5a6a583f8569e0ee5ff897
COPY . /deep-ltl
WORKDIR /deep-ltl/src/envs/zones/safety-gymnasium
RUN pip install -e .
WORKDIR /deep-ltl
RUN pip install -r requirements.txt
RUN apt update && apt install -y openjdk-11-jre wget unzip
RUN wget https://www7.in.tum.de/~kretinsk/rabinizer4.zip -O rabinizer4.zip && \
    unzip rabinizer4.zip && \
    rm rabinizer4.zip
