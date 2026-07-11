import os
from dotenv import load_dotenv
from google import genai
from src.vectorization import Vectorizer
from src.retriever import mainretriever
vc =Vectorizer()
vector_db = vc.hfembedder()
mnr = mainretriever()
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
while True:
    question = input("\nQuestion: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        break
    context = mnr.Topkretriever(question,vector_db,3)
    sysprompt = f'''
        You are as Senior Ai Model Advisor 
        You have to give the response in more technichal
        Response by using the Context effeciently
        this is user query {question} and this is 
        context {context} you have to give response in
        small General Explanation with some Key important 
        Details
        '''
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=sysprompt,
    )
    print("\nGemini Response:\n")
    print(response.text)









