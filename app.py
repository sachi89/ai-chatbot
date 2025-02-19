from flask import Flask, request, jsonify
import os
import logging
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS

# Install dependencies using the following command:
# pip install flask langchain-openai
# pip install flask-cors

app = Flask(__name__)
# enables CORS to make requests between domains on all routes
CORS(app)

# # Initialize LLM outside of the request handler
openai_api_key = os.environ.get("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"

# # Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not openai_api_key:
    logger.error("OpenAI API key is not set properly.")
    raise EnvironmentError("OpenAI API key is not set properly.")

# Initialize the chatbot model (make sure to set your OpenAI API key in the environment)
chat_model = ChatOpenAI(api_key=openai_api_key, model=model_name)

# # Define the chatbot's role prompt
# def get_prompt():
#     return ChatPromptTemplate.from_messages([
#         SystemMessage(content="You are a psychotherapist who specializes in Solution Focused Brief Therapy. Assess the problem and ask solution focused questions to guide the user through their therapy session. Help them discover a solution to their problem, whether it is a skill that has worked in the past or a new solution that will prevent the problem from getting worse or help improve their situation. Facilitate and guide the therapy session to the best of your ability in English."),
#         HumanMessage(content="{user_input}")
#     ])

# Maintain message history using ConversationBufferMemory
memory = ConversationBufferMemory(return_messages=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')  # Get user's message from the request body
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Define initial system prompt
        prompt_messages = [SystemMessage(content=("You are a therapist who specializes in Solution Focused Brief Therapy. Ask simple, short, solution focused questions to guide the user through a brief therapy session and help them discover an efficient solution to their problem. The solution may not resolve the problem entirely, and thats okay, because some improvement is better than no improvement or the problem getting worse. Solutions can also be a skill they already have or something that has worked in the past. Avoid asking too many questions at one time and break them into simple, easy steps. "
                "Some sample questions asked in Solution Focused Brief Therapy are the following: [How often does this problem occur? How long has this problem been occurring? What happens when this problem occurs? What is said or done and by whom? What happens after? "
                "How have you coped or dealt with the problem so far?  How have you already tried to address this problem? What has worked even a little? What changes have happened since you tried to address this problem? Did it get better or worse? When is the problem not occurring or not affecting you as much? Is there something that makes the problem better sometimes? "
                "What are you doing that is different? Is this something you can try doing more of or keep doing? Tell me about a time when the problem did not occur at all. What was different about that time and what were you doing that was different? "
                "Is this something you can try doing again? On a scale from 0 to 10, with 10 being the best in terms of the problem, and 0 being how you felt when the problem was at its worst, where would you place yourself on the scale right now? "
                "10 may not be a realistic goal at the moment, so what number will be an acceptable goal for you? How will you know when you are one point further up the scale? What other differences will there be when you are one point further up? "
                "Who else will notice this? How do you keep from being at a lower number? How have you managed to keep things from getting worse? What else helps? "
                "Suppose you go to sleep tonight and while you are asleep a miracle happens and the problem that brought you here is solved. But you were asleep and dont know that it has been solved. How will you discover that this miracle happened? What will be the first signs you notice are different that will tell you that this miracle has happened and that the problem has been solved? "
                "Who else will notice? What will they notice about you or you doing differently when this happens? What will it be like when the problem is solved? What will be different? What will you be doing instead?]"))]

        # Retrieve and format previous messages
        chat_history = memory.chat_memory.messages  # Directly get stored messages

        # Prepare full conversation history with new user input
        messages = prompt_messages + chat_history + [HumanMessage(content=user_input)]

        # Generate response
        response = chat_model.predict_messages(messages)

        # Store new interaction in memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.content)
        
        # # Retrieve previous messages from memory
        # chat_history = memory.load_memory_variables({}).get('history', [])
        
        # # Create prompt and chain with memory
        # prompt = get_prompt()
        # formatted_prompt = prompt.format(user_input=user_input)
        
        # # Generate response
        # response = chat_model.predict_messages([formatted_prompt] + chat_history)
        
        # # Update memory with new messages
        # memory.save_context({'input': user_input}, {'output': response.content})
        
        return jsonify({'response': response.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    # Endpoint to clear chatbot memory when the page is reloaded
    memory.clear()
    memory.chat_memory.messages = []  # Explicitly reset message history
    return jsonify({'message': 'Chat memory cleared successfully'})

if __name__ == '__main__':
    app.run(debug=True)