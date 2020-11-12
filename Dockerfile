FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
ADD /requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN apt update && apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
WORKDIR /ws_data/nctu/bag_of_words

ENV ROOT=/ws_data/nctu/bag_of_words
ENV RESULT_PATH=/results

ENTRYPOINT ./entrypoint