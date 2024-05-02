#importing libraries
import openai


claude_key = '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'

import anthropic

# client = anthropic.Anthropic(
#     # defaults to os.environ.get("ANTHROPIC_API_KEY")
#     api_key=claude_key
# )
# message = client.messages.create(
#     model="claude-3-opus-20240229",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, Claude"}
#     ]
# )
# print(message.content)


# import anthropic

def chat_with_claude(messages):
    if not isinstance(messages, list):
        messages = [{"role": "user", "content": messages}]
    client = anthropic.Anthropic(api_key=claude_key)
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=messages,
    )
    refined_message = response.content
    return refined_message











# openai.api_key = 'sk-proj-j75oK7MFpZXqrn7l09rJT3BlbkFJzXoOOrTo6RLnDGCsb1VM'
# key = openai.api_key


# def chat_with_gpt(message):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=message,
#     )
#     return response.choices[0].message["content"]




# message = [
#     {"role": "system", "content": " You are a refiner of text gotten from a sign language translation model."},

# ]

# while True:
#     message = input("User: ")
#     if message:
#         message.append({"role": "user", "content": message})
        
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=message,
#         )
        
#         reply = response.choices[0].message["content"]
#         print(f"Translator: {reply}")
#         message.append({"role": "system", "content": reply})
        