#base image
FROM gcr.io/deeplearning-platform-release/pytorch-gpu
RUN apt update && \
   apt install --no-install-recommends -y build-essential gcc wget curl python3.9 && \
   apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /root
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install dvc 'dvc[gs]'

COPY requirements.txt /tmp/requirements.txt
COPY setup.py setup.py
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install -r /tmp/requirements.txt --no-cache-dir

RUN echo 'GOING TO COPY'
COPY src/ src/
COPY .git/ .git/
COPY .dvc/config .dvc/config

COPY data.dvc data.dvc
COPY models.dvc models.dvc

COPY entry_training.sh entry_training.sh
RUN echo 'ENTRY INCOMING'
ENTRYPOINT ["./entry_training.sh"]