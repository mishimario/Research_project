FROM tensorflow/tensorflow:2.1.0-gpu-py3

RUN apt-get update && apt-get install -y git

WORKDIR /kw_resources/Research_with_MAZDA

RUN cp requirements.txt /tmp

RUN pip install -r requirements.txt

CMD python3 -m segmentator train \
    --config \
        /kw_resources/Research_with_MAZDA/configs/unet.yaml \
        /kw_resources/Research_with_MAZDA/configs/additionals/data_options.yaml \
        /kw_resources/Research_with_MAZDA/configs/additionals/deploy_options.yaml \
    --save_path /kw_resources/Research_with_MAZDA/results \
    --data_path \
        /kw_resources/Research_with_MAZDA/segmentator/data/JPEGImages3168 \
        /kw_resources/Research_with_MAZDA/segmentator/data/SegmentationClass3168_yolo \
    --max_steps \
        50 \
