import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths
DATA_DIR = os.path.join(ROOT, "data", "fiqa")
RESULTS_DIR = os.path.join(ROOT, "results")

# Dataset
FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

# Dense model
DENSE_MODEL = "all-MiniLM-L6-v2"
DENSE_BATCH_SIZE = 64

# Retrieval defaults
DEFAULT_TOP_K = 10
DEFAULT_ALPHA = 0.5

# Evaluation windows
COLD_WINDOW = 20       # first N queries measured as cold-start
WARMUP_WINDOW = 100    # next N queries discarded (warmup); warm = queries after COLD_WINDOW + WARMUP_WINDOW

# Ablation
ABLATION_ALPHAS = (0.3, 0.5, 0.7)
