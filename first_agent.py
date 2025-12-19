import os
from dotenv import load_dotenv
from smolagents import LiteLLMModel, CodeAgent, DuckDuckGoSearchTool, tool
import datetime, pytz
import gradio as gr

# --- Step 1: Load environment variables from the .env file ---
# load_dotenv() searches for the .env file in the current directory and loads
# the key/value pairs into the environment (os.environ).
load_dotenv() 

# --- Step 2: Access the API Key (Optional Check) ---
# You can optionally check if the variable was loaded correctly, 
# but models often do this check internally.
# openai_key = os.getenv("OPENAI_API_KEY")

# if openai_key:
#     print("Open API Key loaded successfully. Length:", len(openai_key))
# else:
#     print("WARNING: OPENAI_API_KEY not found in environment.")

# --- Step 3: Use the Model (it automatically reads the environment) ---
# LiteLLMModel automatically picks up the OPENAI_API_KEY from os.environ

#----------------------------------------------------------------------------------
#       CHECK THAT MODELS CAN BE ACCESSED
#----------------------------------------------------------------------------------

# OPEN AI -------------------------------------------------------------------------
try:
    openai_model = LiteLLMModel(model_id="openai/gpt-4o-mini")
    print(f"Open AI modèle chargé")
except Exception as e:
    print(f"Pb chargement modèle Open AI")
    
# GOOGLE MODEL --------------------------------------------------------------------
try:
    gemini_model = LiteLLMModel(model_id="gemini/gemini-2.5-flash")
    print(f"Google Gemini AI modèle chargé")
except Exception as e:
    print(f"Pb chargement modèle Gemini")
    
# MISTRAL --------------------------------------------------------------------------
try:
    mistral_model = LiteLLMModel(model_id="mistral/mistral-small-latest")
    print(f"Mistral AI modèle chargé")
except Exception as e:
    print(f"Pb chargement modèle Mistral")

#------------------------------------------------------------------------------------
#       TOOLS DEFINITION 
#------------------------------------------------------------------------------------

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

#------------------------------------------------------------------------------------
#        AGENTS
#------------------------------------------------------------------------------------

instructions="""
You are a helpful assistant.
Use your tools whenever relevant.
Prefer short, correct answers.
"""

agent = CodeAgent(
    tools=[
        get_current_time_in_timezone,
    ], 
    model=mistral_model,
    max_steps=5,
    verbosity_level=1,
    additional_authorized_imports=[
        "pytz",
        "datetime"
        ],
    instructions=instructions
)

#-------------------------------------------------------------------------------------
#    "APP"
#-------------------------------------------------------------------------------------

print(f"Agent is powered by: {agent.model.model_id}")
print("-" * 25)

print(f"Gradio version is {gr.__version__}")


def chat_with_agent(message, history):
    # history is now a list of dicts
    history = history or []

    # add user message
    history.append({
        "role": "user",
        "content": message
    })

    # run agent
    response = agent.run(message)

    # add assistant message
    history.append({
        "role": "assistant",
        "content": response
    })

    return history

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 smolagents Chat")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(
        placeholder="Ask something…",
        show_label=False
    )

    msg.submit(
        chat_with_agent,
        inputs=[msg, chatbot],
        outputs=chatbot
    )

demo.launch()