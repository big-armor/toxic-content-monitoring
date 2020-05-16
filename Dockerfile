FROM python:3.7

# install build utilities
RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

EXPOSE 80

WORKDIR  /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /usr/src/app
COPY data/ /usr/src/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
