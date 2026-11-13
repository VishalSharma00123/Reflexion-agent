from typing import List

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="The Critique of what is missing in the answer")
    superfluous: str = Field(description="The Critique of what is superfluous in the answer")

class AnswerQuestion(BaseModel):
    """Answer the user's question above using the required format"""

    answer: str = Field(description="250 words detailed answer to the user's question")
    reflection: Reflection = Field(description="Your reflection on the answer")
    search_queries: List[str] = Field(description="serch 1-3 queries to address the critique and improve the answer") 

class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to you question"""

    references: List[str] = Field(description="Citations motivating your updated answer")

    
    