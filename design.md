Personal Research/Knowledge Tool
- Utilizing LangGraph for build
- State schema (flowing through system)
    TypedDict with loops: int, store_ready: bool
- Vector DB + Embedding Model (ChromaDB + lightweight OpenAI)
- React based UI with dominant SSR, lightweight codegen for rendered markdown returned to user for node 3


# Schema
## Node 1: Comprehension Check
Input: Daily_Notes.txt, >200 words long to trigger workflow
Output: JSON payload with fields day=, comprehension_score= [call this output_1 for ease]
Added: Embedding (through HF's Inference Client [Google/gemma])  + Local Natural Language Inference (Local Tokenization [facebook/bart])

If loop: proliferate on output_1x 

## Node 2: Clarification [Conditional ]
Update 4/21: Node ingests output_1 and builds context for downstream, conducted research. 
- Weak_topics arg from output_1 utilized, contradiction (from nli) or weak coverage (based on cosine sim) is then passed into research builder.
- All edge cases between contradiction & weak coverage (if hypothesis incongruous with range(1-k)  semantic chunks), builds confidence score using comprehension score (computed through nli in node 1)
- Lowest confidence scorfes then passed in to research
Max output tokens set extremely low for cost savings.
Input: Output_1
Output: Output_2


Node 3: Research [Conditional ]
Research conducted through access to web browser, builds .md file with research focused on WHY/How does this topic work. Returns this markdown file to user (visually and saved locally), then triggers UI component for user to "try again" with a section space that creates a .txt file to trigger Node 1 once more.

Input: output_2
Output: Output_3 as a markdown file returned to user, saved under research dir. User reads then inputs learnings/notes into a .txt file causing loop. This is an interruption in workflow designed to place human in the loop.

Node 4: Synthesis
Compares against prior nodes to track growth in learning.
Ideas: semantic search against stored vector db (utilize nearest neighbors algo, return nearest 3 chunks for LLM to form text bit), returns tuple of JSON payload and .txt file of synthesis. JSON payload fileds, day= , success= 

Input: Output_1 / 1_x etc
Output: Json payload, synthesis_Day.txt

Node 5: Store
When received JSON payload from Node 4, takes output_1/1_x and persists into vector db for semantic retrieval + potential RAG downstream to "query own brain". Knowledge base builds over time

Appendix: Lightweight Schemas

LangGraph state
```python
from typing import TypedDict, Optional, Literal

class GraphState(TypedDict, total=False):
    run_id: str
    day: str
    loop_count: int
    max_loops: int
    store_ready: bool
    exit_reason: Literal["continue", "done", "max_loops", "manual_review"]
    weak_topics: list[str]

    daily_notes_path: str
    output_1: "Output1"
    output_2: "Output2"
    output_3: "Output3"
    output_4: "Output4"
```

Node payloads
```python
class Output1(TypedDict):
    day: str
    comprehension_score: float
    needs_clarification: bool
    weak_topics: list[str] | None # when looped
    cleaned_summary: str

class Output2(TypedDict, total=False):
    topics: list[dict]
        """ dict contains below key-value pairs
        topic: str
        error_explanation: str
        confidence: float
        """

class Output3(TypedDict):
    research_md_path: str
    prompt_user_retry: bool

class Output4(TypedDict):
    day: str
    success: bool
    synthesis_txt_path: str
    retrieved_chunk_ids: list[str]
    synthesis_text: str
```
