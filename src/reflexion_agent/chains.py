import datetime
from dotenv import load_dotenv

from reflexion_agent.schema import AnswerQuestion, ReviseAnswer

load_dotenv()

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os



llm = ChatNVIDIA(model="nvidia/nemotron-3.5-lightning-30b-a3b", api_key=os.getenv("NVIDIA_API_KEY"))
parser = JsonOutputParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        you're a expert researcher.
        Current time: {time}

        1. {first_instruction}
        2. Reflect and critique you answer, Be severe to maximize the improvement.
        3. Recommend Search queries to research information and improve the answer
        """),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed 250 word answer",
    
)

first_responder=first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion]
)

revise_instruction="""
    Revise your answer using the new instruction.
    - You should use the previous critique to improve the answer.
    - You MUST include numerical citations for your sources and ensure it can be verified.
    - Add a references section to the end of the answer.
    - [1] https://example.com
    - [2] https://example.com
    - You should use the previous critique to remove the  superfluous information.
"""
    
revisor = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer])

if __name__ == "__main__":
    human_message=HumanMessage(
        content="write about AI-powered SOC / autonomous soc problem domain," \
        "list startups that do that and raised capital"
    )

    chain=(
    first_responder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | parser_pydantic
    )

    res = chain.invoke({"messages": [human_message]})
    print(res)