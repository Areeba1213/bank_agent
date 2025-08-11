from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, function_tool, GuardrailFunctionOutput, input_guardrail
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio

# Load environment variables
load_dotenv()
set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure OpenAI model
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

# ------------------ Data Models ------------------

class Account(BaseModel):
    name: str
    pin: int

class MyOutputType(BaseModel):
    balance: str

class GuardrailOutput(BaseModel):
    is_not_bank_related: bool

# ------------------ Guardrail Agent ------------------

guardrail_agent = Agent(
    name="GuardrailAgent",
    instructions="You are a guardrail agent. Return true if the query is NOT related to banking.",
    output_type=GuardrailOutput,
    model=model
)

@input_guardrail
async def check_bank_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    print(f"GuardrailAgent decision: is_not_bank_related={result.final_output.is_not_bank_related}")
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_not_bank_related
    )

# ------------------ Tool ------------------

# For now, always enable the tool
def check_user(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    return True

@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> MyOutputType:
    return MyOutputType(balance="100000")

# ------------------ Main Bank Agent ------------------

bank_agent = Agent(
    name="BankAgent",
    instructions="You are a bank agent. You can check account balance and perform transactions.",
    tools=[check_balance],
    output_type=MyOutputType,
    input_guardrails=[check_bank_related],
    model=model
)

# ------------------ Run ------------------

user_context = Account(name="Alice", pin=1234)

result = Runner.run_sync(
    bank_agent,
    input="Check balance for account 12345",
    context=user_context
)

print(f"Final Output: {result.final_output}")



