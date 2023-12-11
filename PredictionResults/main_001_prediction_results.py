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


coder = autogen.AssistantAgent(
    name="Coder",  # the default assistant agent is capable of solving problems with code
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
   name="User_proxy",
#    system_message="A human admin.",
#    code_execution_config={"last_n_messages": 3, "work_dir": "coding"},
   code_execution_config={"work_dir": "coding"},
   human_input_mode="ALWAYS",
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. You are a helpful assistant highly skilled in evaluating the quality of a given visualization code by providing a score from 1 (bad) - 10 (good) while providing clear rationale. YOU MUST CONSIDER VISUALIZATION BEST PRACTICES for each evaluation. Specifically, you can carefully evaluate the code across the following dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- Data transformation (transformation): Is the data transformed appropriately for the visualization type? E.g., is the dataset appropriated filtered, aggregated, or grouped  if needed? If a date field is used, is the date field first converted to a date object etc?
- Goal compliance (compliance): how well the code meets the specified visualization goals?
- Visualization type (type): CONSIDERING BEST PRACTICES, is the visualization type appropriate for the data and intent? Is there a visualization type that would be more effective in conveying insights? If a different visualization type is more appropriate, the score MUST BE LESS THAN 5.
- Data encoding (encoding): Is the data encoded appropriately for the visualization type?
- aesthetics (aesthetics): Are the aesthetics of the visualization appropriate for the visualization type and the data?

YOU MUST PROVIDE A SCORE for each of the above dimensions.
{bugs: 0, transformation: 0, compliance: 0, type: 0, encoding: 0, aesthetics: 0}
Do not suggest code. 
Finally, based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
""",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[user_proxy, coder, critic], messages=[], max_round=20)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

message = """
# Mission
You are to architect a class to check prediction results for stock market data.

# Context
This is to be part of a script to use an LLM for stock predictions.
The LLM functionality will be implemented later.
Stock symbol validation has been implemented.
Stock data collection has been implemented.

# Rules
Use the class name PredictionResults.
Create all necessary methods.

# Instructions
Start with the following code:
class PredictionResults:
    # Class to check stock market prediction results.

    @staticmethod
    def exec() -> None:
        '''Call the other methods to check the stock data prediction results.'''


You are allowed to create additional classes if you deem it necessary.

1. Allow the user to choose a file from the current working directory that starts with 'predictions_output'.
2. Read the file into a pandas dataframe.
3. Select all rows where PredictedClass is 5.  Sort the rows in the dataframe by 'Symbol'.
4. Save the 'Symbol' column in the first data frame to the file 'predictioned_symbols.csv'.
- Create an instance of
```python
class StockData:
    def __init__(self, stock_symbols_path: str, start_time: datetime, end_time: datetime):
```
Using 'Date' column in the first data frame for start_time.
end_time = start_time + datetime.timedelta(days=1)



4. Read the symbol data from the 'data' directory using StockData._get_stock_data(self, symbol: str)
5. Use the 'Date' column in the first data frame to match the 'Date' column in the second data frame.
6. Calculate the PercentGain and DollarGain using the 'Close' and 'Open' columns in the second data frame.
7. Add those columns to the first data frame.
8. Save the dataframe to a CSV file.

# Expected Input
Use the selected CSV file.  The file contains the stock data.
Each file contains a line containing a header.

# Output Format
Python
"""

user_proxy.initiate_chat(manager, message=message)

