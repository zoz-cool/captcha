FROM python:3.8

MAINTAINER yczha

RUN groupadd -r yczha && useradd -r -g yczha yczha
WORKDIR /home/yczha

COPY requirements.txt requirements.txt
COPY torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl
RUN pip install pip setuptools --upgrade -i https://pypi.mirrors.ustc.edu.cn/simple/\
    && pip install torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl\
    && pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/ \
    && rm requirements.txt torch-1.9.0-cp38-cp38-manylinux1_x86_64.whl && rm -rf /.pip

COPY app.py boot.sh ./
COPY model model
RUN chmod +x boot.sh
RUN chown -R yczha /home/yczha
ENV FLASK_APP app.py
EXPOSE 5000

USER yczha
ENTRYPOINT ["./boot.sh"]
