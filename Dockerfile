FROM python:3

WORKDIR /app

COPY . /app

RUN pipenv install
RUN pipenv shell

EXPOSE 5000

CMD py ./spokenlanguageassessment.py