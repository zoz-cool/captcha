FROM python:3.8

RUN groupadd -r yczha && useradd -r -g yczha yczha
WORKDIR /home/yczha

COPY requirements.txt requirements.txt
ARG PYPI=http://mirrors.tencentyun.com/pypi/simple
RUN pip install -r requirements.txt -i ${PYPI} --no-cache-dir --trusted-host mirrors.tencentyun.com\
    && rm -rf requirements.txt /.pip
COPY app.py app.py
COPY model model
RUN chown -R yczha /home/yczha
EXPOSE 5000

USER yczha
CMD exec uvicorn app:app --host 0.0.0.0 --port 5000