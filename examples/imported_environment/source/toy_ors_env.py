from ors import (
    Environment,
    ListToolsOutput,
    RunToolOutput,
    Split,
    TextBlock,
    ToolOutput,
    ToolSpec,
)


class ToyMathEnvironment(Environment):
    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train")]

    @classmethod
    def list_tasks(cls, split):
        return [
            {
                "id": "addition-1",
                "question": "What is 2 + 2?",
                "answer": "4",
            }
        ]

    @classmethod
    def num_tasks(cls, split):
        return len(cls.list_tasks(split))

    @classmethod
    def get_task(cls, split, index):
        return cls.list_tasks(split)[index]

    @classmethod
    def get_task_range(cls, split, start=None, stop=None):
        return cls.list_tasks(split)[slice(start, stop)]

    @classmethod
    def list_tools(cls):
        return ListToolsOutput(
            tools=[
                ToolSpec(
                    name="answer",
                    description="Submit an answer for the math question",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                )
            ]
        )

    def list_task_tools(self):
        return ListToolsOutput(tools=[])

    def get_prompt(self):
        return [TextBlock(text=self.task_spec["question"])]

    def _call_tool(self, name, input):
        guess = str(input.get("value", "")).strip()
        expected = self.task_spec["answer"]
        correct = guess == expected
        return RunToolOutput(
            ToolOutput(
                blocks=[TextBlock(text="correct" if correct else "incorrect")],
                metadata={"expected": expected, "received": guess},
                reward=1.0 if correct else 0.0,
                finished=True,
            )
        )
