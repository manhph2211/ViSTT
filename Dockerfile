FROM python:3.9

WORKDIR /app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y
RUN pip install --upgrade requests

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY /VASR/streamlit/app.py /app/

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]