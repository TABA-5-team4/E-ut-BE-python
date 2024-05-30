import os
from langchain_openai import ChatOpenAI
from langchain.prompts import StringPromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough


api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key = api_key,
    model = "gpt-3.5-turbo", # 3.5 빠름, 4.0 정확
    temperature = 0.7 # 창의성 (0.0 ~ 2.0)
)

chat_bot_prompt = """
            Your role is a persona who talks to relieve the loneliness of an old man who lives alone
            You can bloom according to the information I give you.
            Please always answer in Korean.
            Say the answer politely.
            First, talk to me in a casual conversation and when I answer, talk to me like a casual conversation to suit your role.
    """

memory = ConversationSummaryBufferMemory(
    llm = llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]

prompt = ChatPromptTemplate.from_messages([
    ("system", chat_bot_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm

def invoke_chain(question: str):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content}
    )
    print(result)
    return result
















PROMPT_TEMPLATE = """
context: {context}
history: {history}
input: {input}

Generate an answer to an input that takes into account context and history.
answer:

"""

class CustomPromptTemplate(StringPromptTemplate):
    context: str
    template: str

    def format(self, **kwargs) -> str:
        context = kwargs.get('context', '')
        return self.template.format(context=context, **kwargs)

PROMPT = CustomPromptTemplate(
    input_variables=["history", "input"],
    template=PROMPT_TEMPLATE,
    context ="""
            Your role is a persona who talks to relieve the loneliness of an old man who lives alone
            You can bloom according to the information I give you.
            Please always answer in Korean.
            Say the answer politely.
            First, talk to me in a casual conversation and when I answer, talk to me like a casual conversation to suit your role.
    """
)

conversation = ConversationChain(
    prompt = PROMPT,
    llm = llm,
)

def chat_with_gpt(text: str):
    history = (
        ""
    )
    formatted_prompt = PROMPT.format(history="", input=text)
    response = llm.invoke(text=formatted_prompt)

    return response