import os
import json
import csv
import asyncio
from datetime import datetime
import openai
from pydantic import BaseModel
from typing import Optional, List, Dict
import inspect

openai.api_key = os.getenv("OPENAI_API_KEY")


class Agent(BaseModel):
    name: str
    model: str = "gpt-4"
    instructions: str
    tools: list = []


class Question(BaseModel):
    id: int
    vignette: str = ""
    choices: List[Dict[str, str]] = []  # List of {text: str, explanation: str}
    system: str
    clerkship: str
    difficulty: int
    status: str = "pending"


class Response(BaseModel):
    agent: Optional[Agent]
    messages: list


def function_to_schema(func) -> dict:
    """Convert a Python function to an OpenAI function schema."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    signature = inspect.signature(func)
    parameters = {}
    for param in signature.parameters.values():
        param_type = param.annotation
        # Handle list of dictionaries specifically
        if param_type == list:
            parameters[param.name] = {
                "type": "array",
                "items": {"type": "object"}  # Specify that list items are objects
            }
        else:
            param_type = type_map.get(param_type, "string")
            parameters[param.name] = {"type": param_type}

    required = [
        param.name for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "name": func.__name__,
        "description": (func.__doc__ or "").strip(),
        "parameters": {
            "type": "object",
            "properties": parameters,
            "required": required,
        },
    }


def transfer_to_writer(question_id: int) -> str:
    """Transfer to question writer agent."""
    return "Transferred to Question Writer"


def transfer_to_reviewer(question_id: int) -> str:
    """Transfer to peer reviewer agent."""
    return "Transferred to Peer Reviewer"


def transfer_to_editor(question_id: int) -> str:
    """Transfer to copy editor agent."""
    return "Transferred to Copy Editor"


def write_question(question_id: int) -> str:
    """Write a new medical board question."""
    question = swarm.questions.get(question_id)
    # Generate a question based on the system, clerkship, and difficulty
    if question.system.lower() == "cardiac":
        question.vignette = "A 65-year-old man with a history of hypertension and hyperlipidemia presents with chest pain..."
        question.choices = [
            {"text": "Stable angina", "explanation": "Explanation about stable angina."},
            {"text": "Unstable angina", "explanation": "Explanation about unstable angina."},
            {"text": "Myocardial infarction", "explanation": "Explanation about MI."},
            {"text": "Pericarditis", "explanation": "Explanation about pericarditis."},
        ]
    else:
        # Default or other systems can be added here
        question.vignette = "This is a sample vignette about a patient..."
        question.choices = [
            {"text": "Choice A", "explanation": "Explanation A"},
            {"text": "Choice B", "explanation": "Explanation B"},
            {"text": "Choice C", "explanation": "Explanation C"},
            {"text": "Choice D", "explanation": "Explanation D"},
        ]
    question.status = "written"
    swarm.questions[question_id] = question
    return "Question written successfully"


def review_question(question_id: int) -> str:
    """Review a medical board question for accuracy."""
    question = swarm.questions.get(question_id)
    question.status = "reviewed"
    swarm.questions[question_id] = question
    return "Question reviewed successfully"


def edit_question(question_id: int) -> str:
    """Copy edit a medical board question."""
    question = swarm.questions.get(question_id)
    question.status = "edited"
    swarm.questions[question_id] = question
    return "Question edited successfully"


def save_question(question_id: int, output_path: str) -> str:
    """Save the completed question to a JSON file."""
    question = swarm.questions.get(question_id)
    question.status = "completed"
    json_content = {
        "id": question.id,
        "vignette": question.vignette,
        "choices": question.choices,
        "system": question.system,
        "clerkship": question.clerkship,
        "difficulty": question.difficulty,
        "status": question.status
    }

    json_file_path = os.path.join(output_path, f"question_{question.id}.json")
    with open(json_file_path, 'w') as f:
        json.dump(json_content, f, indent=4)

    swarm.questions[question_id] = question
    return "Question saved successfully"


project_manager = Agent(
    name="Project Manager",
    instructions="""You are the project manager for medical board question writing.
Your role is to:
1. Load questions from CSV
2. Assign questions to writers
3. Track progress
4. Save completed questions
Use the provided tools to perform your tasks.""",
    tools=[transfer_to_writer])

writer_agent = Agent(
    name="Question Writer",
    instructions="""You are a medical subject matter expert writing board questions.
For each question:
1. Write a clinical vignette (1-2 paragraphs)
2. Create 4-5 MCQ answers
3. Write detailed explanations for each choice
4. Transfer to peer reviewer when complete
Use the provided tools to perform your tasks, such as 'write_question' and 'transfer_to_reviewer'.""",
    tools=[write_question, transfer_to_reviewer])

reviewer_agent = Agent(
    name="Peer Reviewer",
    instructions="""You are a medical peer reviewer.
For each question:
1. Review clinical accuracy
2. Verify answer choices
3. Check explanation completeness
4. Transfer to copy editor when complete
Use the provided tools to perform your tasks, such as 'review_question' and 'transfer_to_editor'.""",
    tools=[review_question, transfer_to_editor])

editor_agent = Agent(
    name="Copy Editor",
    instructions="""You are a copy editor for medical board questions.
For each question:
1. Check grammar and spelling
2. Ensure NBME style guidelines
3. Verify question flow and readability
4. Save completed question when done
Use the provided tools to perform your tasks, such as 'edit_question' and 'save_question'.""",
    tools=[edit_question, save_question])


def execute_tool_call(function_name, args, tools_map, agent_name, swarm):
    """Execute a function call and handle agent transfers."""
    # Retrieve the function
    func = tools_map.get(function_name)
    if not func:
        return None, f"Function {function_name} not found"

    # Get question_id from args
    question_id = args.get("question_id")
    if question_id is None:
        return None, "No question_id provided in function arguments"

    # Call the function
    if function_name == 'save_question':
        result = func(question_id, swarm.output_path)
    else:
        result = func(question_id)

    # Handle agent transfers
    if result.startswith("Transferred to"):
        new_agent_name = result.split("Transferred to ")[1]
        new_agent = {
            "Question Writer": writer_agent,
            "Peer Reviewer": reviewer_agent,
            "Copy Editor": editor_agent
        }.get(new_agent_name)
        return new_agent, result
    elif function_name == "save_question" and agent_name == "Copy Editor":
        # Mark question as completed after Copy Editor saves it
        question = swarm.questions.get(question_id)
        question.status = "completed"
        swarm.questions[question_id] = question

    return None, result


def run_full_turn(agent: Agent, messages: list, swarm) -> Response:
    """Run a full turn of the conversation with potential agent transfers."""
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # Convert tools to OpenAI function schemas
        tool_schemas = [
            function_to_schema(tool) for tool in current_agent.tools
        ]
        tools_map = {tool.__name__: tool for tool in current_agent.tools}

        # Get completion from OpenAI
        response = openai.ChatCompletion.create(
            model=current_agent.model,
            messages=[{
                "role": "system",
                "content": current_agent.instructions
            }] + messages,
            functions=tool_schemas,
            function_call="auto",
            temperature=0
        )
        message = response['choices'][0]['message']
        messages.append(message)

        if message.get('content'):
            print(f"{current_agent.name}: {message['content']}")

        if 'function_call' in message:
            # Handle function call
            function_call = message['function_call']
            function_name = function_call['name']
            function_args = function_call['arguments']

            # Parse arguments as JSON
            try:
                args = json.loads(function_args)
            except json.JSONDecodeError:
                # Handle invalid JSON
                print(f"Invalid function arguments: {function_args}")
                break

            # Execute the function
            new_agent, result = execute_tool_call(function_name, args, tools_map, current_agent.name, swarm)

            # Append function's output as assistant's message
            messages.append({
                "role": "function",
                "name": function_name,
                "content": result
            })

            if new_agent:
                current_agent = new_agent
                # Reset messages when transferring to a new agent
                question = swarm.questions[args['question_id']]
                messages = [{
                    "role": "system",
                    "content": current_agent.instructions
                }, {
                    "role": "user",
                    "content":
                    f"Process question {question.id}: {question.system} system, {question.clerkship} clerkship, difficulty {question.difficulty}"
                }]
        else:
            break

    return Response(agent=current_agent, messages=messages[num_init_messages:])


class QuestionSwarm:

    def __init__(self, csv_path: str, output_path: str):
        self.csv_path = csv_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.questions = {}
        self.load_questions()

    def load_questions(self):
        """Load questions from CSV."""
        with open(self.csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.questions[int(row['id'])] = Question(
                    id=int(row['id']),
                    system=row['system'],
                    clerkship=row['clerkship'],
                    difficulty=int(row['difficulty']))

    def save_questions(self):
        """Save completed questions to JSON files."""
        for question_id, question in self.questions.items():
            if question.status == "completed":
                save_question(question_id, self.output_path)

    def process_questions(self):
        """Process all questions through the agent swarm."""
        for question_id, question in self.questions.items():
            if question.status != "pending":
                continue

            print(f"\nProcessing question {question_id}...")

            agent = project_manager
            messages = [{
                "role": "user",
                "content":
                f"Process question {question_id}: {question.system} system, {question.clerkship} clerkship, difficulty {question.difficulty}. Please use the available tools to perform your tasks."
            }]

            while True:
                response = run_full_turn(agent, messages, self)
                agent = response.agent
                messages.extend(response.messages)

                # Check if question is complete
                if question.status == "completed":
                    break

            self.save_questions()


def main():
    global swarm
    swarm = QuestionSwarm(csv_path="questions.csv", output_path="completed_questions")
    swarm.process_questions()


if __name__ == "__main__":
    main()
