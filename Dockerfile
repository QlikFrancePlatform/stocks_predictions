FROM python:3.9-slim

RUN echo "Europe/Paris" > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && pip install -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8383", "--reload"]