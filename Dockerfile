FROM python:3.9-slim
ENV PYTHONPATH=/usr/lib/python3.9/site-packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils libgl1 libglib2.0-0 \
    python3-pip \
    && apt-get install psmisc \
    && apt-get clean \
    && apt-get autoremove
RUN mkdir /Object-detection
WORKDIR /object-detection
COPY . /object-detection
RUN pip3 install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org python-multipart
EXPOSE 8000
CMD ["python", "app.py"]

