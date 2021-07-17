FROM python:3.8

RUN groupadd -r yczha && useradd -r -g yczha yczha
WORKDIR /home/yczha

COPY requirements.txt requirements.txt
COPY torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl
ARG PYPI=https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install pip setuptools --upgrade -i ${PYPI}\
    && pip install torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl\
    && pip install -r requirements.txt -i ${PYPI}\
    && rm requirements.txt torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl && rm -rf /.pip

COPY app.py ./
COPY model model
RUN chown -R yczha /home/yczha
ENV FLASK_APP app.py
EXPOSE 5000

USER yczha
CMD exec gunicorn -w 4 -b :5000 --access-logfile - --error-logfile - app:app