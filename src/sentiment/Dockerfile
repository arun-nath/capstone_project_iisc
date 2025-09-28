FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY EQUITY_L.csv .

COPY app.py .

CMD ["python" ,"app.py"]