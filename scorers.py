import braintrust
from pydantic import BaseModel


class StepEfficiencyScorer(BaseModel):
    output: list[dict]


async def step_efficiency_scorer(output):
    MAX_STEPS = 8
    messages = output.get("messages", [])
    num_steps = len(messages)
    if num_steps <= MAX_STEPS:
        return 1.0

    return max(0.0, 1.0 - (num_steps - MAX_STEPS) / MAX_STEPS)


project = braintrust.projects.create(name="langgraph-supervisor")

project.scorers.create(
    name="Step Efficiency (Bundled)",
    slug="step-efficiency-bundled",
    description="Evaluates the number of steps taken to answer the question.",
    parameters=StepEfficiencyScorer,
    handler=step_efficiency_scorer,
)
