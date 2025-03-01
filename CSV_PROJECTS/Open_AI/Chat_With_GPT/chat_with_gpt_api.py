
#       ---- IMPORT_REQUIRED_LIBRARIES ----
# This allows reading and working with JSON 
# files (used to store the API Key). 
import json
# OpenAI library that allows interacting with 
# the OpenAI API (Example: Sending a text prompt 
# to ChatGPT and receiving back responses).
import openai


# OPENAI_API_KEY: Authenticates 
# The code opens the config.json file.
# Reads and loads its contents as a Python dictionary 
# using json.load(config_file).
with open("config.json") as config_file:
    config = json.load(config_file)
    
# Extracts the API key stored under "OPENAI_API_KEY" 
# and assigns it to openai.api_key.
openai.api_key = config["OPENAI_API_KEY"]


#       ---- DEFINE_FUNCTION_chat_with_gpt(prompt) ----
# This function sends a prompt (user input) to ChatGPT 
# and retreives a response 
def chat_with_gpt(prompt):

    try:
        response = openai.ChatCompletion.create(
            # Specifies the GPT-4 model to should be used
            model="gpt-4",  
            # Sends the user input to ChatGPT
            messages=[{"role": "user", "content": prompt}] 
        )
        # The API response is JSON object containing multiple 
        # choices. Extracts the generated text
        return response["choices"][0]["message"]["content"]  
    # If an error occurs while callling on the OpenAI API, 
    # it catches the error and returns an error message, 
    # instead of crashing
    except openai.error.OpenAIError as e:
        return f"Error: {str(e)}"  # Handle API errors gracefully

# Main execution: Get user input and call the function
if __name__ == "__main__":
    # The script enters an infinite loop (while True), to 
    # continuously except the user's input
    while True:
        # Get input from the user
        user_input = input("You: ")  
        # If the user types "exit", "quit", or "bey" 
        # (case-insensitive), the chat ends.
        if user_input.lower() in ["exit", "quit", "bye"]:
            # ChatGPT prints "Goodbey!", and breaks 
            # out of the loop. Ends execution 
            print("ChatGPT: Goodbye!")
            break
        # Otherwise, the function chat_with_gpt(user_input) 
        # is called to generate a response from GPT.
        # The response is printed to the console
        response = chat_with_gpt(user_input) 
        print(f"ChatGPT: {response}")


