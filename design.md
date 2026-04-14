Personal Research/Knowledge Tool
- Utilizing LangGraph for build
- State schema (flowing through system)
    TypedDict with loops: int, store_ready: bool
- Vector DB + Embedding Model (ChromaDB + lightweight OpenAI)
- React based UI with dominant SSR, lightweight codegen for rendered markdown returned to user for node 3


Schema
Node 1: Comprehension Check
Input: Daily_Notes.txt, >200 words long to trigger workflow
Output: JSON payload with fields day=, comprehension_score= [call this output_1 for ease]
If loop: proliferate on output_1x 

Node 2: Clarification [Conditional ]
If daily notes read weak grammatically/conceptually or inaccurate, node is triggered. To deem inaccurate, a powerful model will read through text, and based on it's training, will determine whether or not my ideas and communication work well. Will provide a directory called agent_docs that specifies what exactly I'm looking for, rather than building monolithinc system prompt.

Idea space is then built with sub-agent with topics attached. Typed payload with fields topic= , error_explanation= (why my initial analysis was wrong) [output_2 for ease]. Max output tokens set extremely low for cost savings.
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
