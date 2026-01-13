from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agent.prompts import INTENT_CLASSIFICATION_PROMPT
from dotenv import load_dotenv

load_dotenv()

class IntentClassifier:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )

    def classify(self, user_input: str) -> str:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(input=user_input)

        response = self.llm.invoke(
                    [HumanMessage(content=prompt)]
                )


        intent = response.content.strip().upper()

        if intent not in {"GREETING", "INFO", "HIGH_INTENT"}:
            return "INFO"

        return intent
    
