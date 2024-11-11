FROM python:3.10
LABEL maintainer="fjs@usp.br"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./src/main.py" ]
