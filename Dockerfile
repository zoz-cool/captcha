FROM python:3.8

RUN groupadd -r yczha && useradd -r -g yczha yczha
WORKDIR /home/yczha

COPY requirements.txt requirements.txt
ARG PYPI=https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install pip setuptools --upgrade -i ${PYPI}\
    && pip install -r requirements.txt -i ${PYPI}\
    && rm requirements.txt && rm -rf /.pip

COPY app.py ./
COPY model model
RUN chown -R yczha /home/yczha
ENV FLASK_APP app.py
EXPOSE 5000

USER yczha
CMD exec uvicorn app:app --host 0.0.0.0 --port 5000