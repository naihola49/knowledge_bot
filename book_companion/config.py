"""Application configuration constants"""

MIN_WORD_COUNT = 200
MAX_LOOPS = 3

"""
Node 1
"""
# Node 1 threshold 
COMPREHENSION_PASS_SCORE = 2 # from NLI
CLARIFICATION_TRIGGER_SCORE = 1 # from NLI

# Embeddings
EMBEDDING_MODEL_NAME = "google/embeddinggemma-300m"
HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/models"
# use get env if makes sense (stored in env.dev)
HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_INFERENCE_API_TIMEOUT_SECONDS = 60
# Min secs between HF Inference API calls (embeddings + NLI) for rate limiting
HF_INFERENCE_MIN_INTERVAL_SECONDS = 1.5

# NLI
NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_MAX_SEQ_LENGTH = 512 # longer truncate
TOP_K_RETRIEVAL = 3 # embeddings in local cache parameter 

"""
Node 2
"""
HIGH_CONTRADICTION_THRESHOLD = 0.7
LOW_COVERAGE_THRESHOLD = 0.35
MAX_TOPICS_DEFAULT = 3
# Max chars of source/user text quoted inside Node 2 research briefs for the LLM.
RESEARCH_TOPIC_SNIPPET_MAX_CHARS = 320
# minimal llm payload for minimized token costs
NODE2_ANTHROPIC_MAX_WEAK_TOPICS = 4
NODE2_ANTHROPIC_MAX_RETRIEVED_CHUNKS = 2
NODE2_ANTHROPIC_MAX_USER_INPUT_CHARS = 240
NODE2_ANTHROPIC_MAX_TOPICS = 3

"""
Node 3 / premise ingestion
"""
TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"
TAVILY_MAX_RESULTS_PER_QUERY = 5
# "advanced" can be enabled later if needed
TAVILY_EXTRACT_DEPTH: str | None = None

"""
Node 2 Anthropic topic compiler
"""
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
ANTHROPIC_MODEL_NAME = "claude-haiku-4-5-20251001"
