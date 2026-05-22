FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# If a pre-built index exists in data/ it will be copied above.
# Otherwise run `docker run ... make index` before `make bench`.
ENTRYPOINT ["python", "src/search.py"]
