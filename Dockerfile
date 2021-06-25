FROM python:3.8

RUN groupadd -r yczha && useradd -r -g yczha yczha
WORKDIR /home/yczha

COPY requirements.txt requirements.txt
RUN pip install pip setuptools --upgrade -i https://mirrors.aliyun.com/pypi/simple/\
    && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ \
    && rm requirements.txt && rm -rf /.pip

COPY model app.py boot.sh ./
RUN chmod +x boot.sh
RUN chown -R yczha /home/yczha
ENV FLASK_APP app.py
EXPOSE 5000

USER yczha
ENTRYPOINT ["./boot.sh"]