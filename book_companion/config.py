"""Application configuration constants"""

MIN_WORD_COUNT = 200
MAX_LOOPS = 3

# Node 1 threshold 
COMPREHENSION_PASS_SCORE = 2 # from NLI
CLARIFICATION_TRIGGER_SCORE = 1 # from NLI

# Embeddings
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
# use get env if makes sense (stored in env.dev)
HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_INFERENCE_API_TIMEOUT_SECONDS = 60
