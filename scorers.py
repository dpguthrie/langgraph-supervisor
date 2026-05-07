import braintrust
from pydantic import BaseModel


class StepEfficiencyScorer(BaseModel):
    output: list[dict]


async def step_efficiency_scorer(output):
    # MAX_STEPS=6 covers single-delegation (4 msgs) and hybrid Research→Math (6 msgs).
    # Above SOFT_LIMIT, the agent is looping or over-routing → score 0.
    MAX_STEPS = 6
    SOFT_LIMIT = 12
    messages = output.get("messages", [])
    num_steps = len(messages)
    if num_steps <= MAX_STEPS:
        return 1.0
    if num_steps >= SOFT_LIMIT:
        return 0.0
    return 1.0 - (num_steps - MAX_STEPS) / (SOFT_LIMIT - MAX_STEPS)


project = braintrust.projects.create(name="langgraph-supervisor")

project.scorers.create(
    name="Step Efficiency (Bundled)",
    slug="step-efficiency-bundled",
    description="Evaluates the number of steps taken to answer the question.",
    parameters=StepEfficiencyScorer,
    handler=step_efficiency_scorer,
)
