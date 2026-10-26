JUDGE_PROMPT = """You are evaluating whether a retrieved memory category provides EVIDENCE for a ground-truth behavioral expectation. This is a strict evidence-based judgment, not a topical similarity check.

**Ground truth expectation (GT)**: A specific behavioral tendency that an annotator expects an agent to exhibit at a specific event.
**Retrieved memory category**: A specific label assigned to a memory the agent actually formed.

**Strict Relevance Criteria:**
A retrieved category is RELEVANT only if it:
- Directly SUPPORTS, REPRESENTS, or provides EVIDENCE for the behavioral expectation
- Describes an action that is a concrete INSTANCE of the expected behavior
- A human reader would say "this memory shows the agent actually did what was expected"

A retrieved category is NOT RELEVANT if:
- It merely shares the same topic or domain but does not demonstrate the behavior
- It describes a tangentially related action without direct evidence
- The connection requires inference or assumption beyond what the memory states

GT expectation: "{gt}"
Retrieved memory: "{ret}"

Answer with JSON exactly: {{"relevant": true/false, "confidence": 0.0-1.0}}
"""
