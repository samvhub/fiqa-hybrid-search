FROM python:3.10-slim
WORKDIR /app

# make is not included in python:3.10-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake model weights into the image so docker run needs no internet access.
# The cache lives in the default HF_HOME (~/.cache/huggingface) inside the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source and any pre-built index artifacts already present in data/
COPY . /app

# Build the BM25 + dense index.
# If data/fiqa/bm25_index.pkl and dense_embeddings.npy were COPYed above this
# is a fast no-op (indexer checks for their presence before rebuilding).
# If they were not present, this downloads FiQA (~18 MB) and encodes all
# 57 K passages — expect ~20 min on a typical CI CPU.
RUN python src/index/indexer.py

# Default: reproduce the full benchmark and write results/bench.json
CMD ["make", "bench"]
