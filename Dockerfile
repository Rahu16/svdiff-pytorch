FROM python:3.8


ADD requirements.txt .

RUN pip3 install -r requirements.txt

RUN pip install svdiff-pytorch

COPY . .

CMD ["python3", "main.py"]