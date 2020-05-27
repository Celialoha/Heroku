FROM python:3.7

COPY . /Application
WORKDIR /Application

RUN pip install -r "requirements.txt"

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]

CMD ["first_app.py"]