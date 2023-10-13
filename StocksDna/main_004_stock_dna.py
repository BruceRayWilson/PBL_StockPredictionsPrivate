from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import autogen

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")


# config_list_gpt4 = autogen.config_list_from_json(
#     "OAI_CONFIG_LIST",
#     filter_dict={
#         "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
#     },
# )

llm_config = {"config_list": config_list, "seed": 42}


# create an AssistantAgent named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "seed": 42,  # seed for caching and reproducibility
        "config_list": config_list,  # a list of OpenAI API configurations
        "temperature": 0,  # temperature for sampling
    },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # set to True or image name like "python:3" to use docker
    },
)

message = """
# Mission
You are to architect a class to process data that has already been preprocessed for stock market data.

# Context
This is to be part of a script to use an LLM for stock predictions.
The LLM functionality will be implemented later.
Stock symbol validation has been implemented.
Stock data collection has been implemented.
Stock data preprocessing has been implemented.
This needs to be done to perform changes to the stock data.
The stock data will needs to be converted to words.

# Rules
Use the class name StockDna.
Create all necessary methods.

# Instructions
Start with the following code:
class StockDna:
    # Class to process collected stock market data

    @staticmethod
    def exec() -> None:
        '''The method just prints a success message as of now'''
        print("Processing data...")


You are allowed to create additional classes if you deem it necessary.

Chunk size is to be retrieved from StockPreprocessor.chunk_size.

For every stock symbol, use chunk size chunks at a time.  

Create a class variable for number of days and set it to 10.

Verify that the number of days is a multiple of chunk size.
Within each chunk, bin Open,High,Low,Close,Tomorrow_Close into 5 bins.
Change the data such that bin 1 is 'a', bin 2 is 'b', bin 3 is 'c', etc.

Within each chunk, bin Volume into 5 bins.
Change the data such that bin 1 is 'a', bin 2 is 'b', bin 3 is 'c', etc.

Save the processed data to the directory './processed_data'.


# Expected Input
Use the CSV files in the 'preprocess_data' directory.  These files contain the stock data.
Each file contains a line containing a header.
The base of the CSV files are the names of the stock symbol.
Each line represents one day.

# Output Format
Python
"""

# the assistant receives a message from the user_proxy, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message=message,
)
