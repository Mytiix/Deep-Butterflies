FROM cytomineuliege/software-python3-base:v2.2.0-py3.6.8

RUN pip install scikit-learn imageio scipy
RUN pip install https://github.com/Cytomine-ULiege/LandmarkTools/archive/master.zip

ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py
ADD download.py /app/download.py

ENTRYPOINT ["python", "/app/run.py"]