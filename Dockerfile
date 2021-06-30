FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/mishimario/Research_project

WORKDIR Research_project

RUN pip install -r requirements.txt

CMD python3 -m segmentator train \
    --config \
        configs/unet.yaml \
        configs/additionals/data_options.yaml \
        configs/additionals/deploy_options.yaml \
    --save_path /kw_resources/Research_with_MAZDA/results \
    --data_path \
        /kw_resources/Research_with_MAZDA/segmentator/data/JPEGImages3168 \
        /kw_resources/Research_with_MAZDA/segmentator/data/SegmentationClass3168_yolo \
    --max_steps \
        50 \
