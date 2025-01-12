from pydantic import BaseModel, Field

class TestDataRowInfo:
    """"Parses individual rows from the manual label CSV file"""
    def __init__(self, about_pharmacy_refusals, additional_info=""):
        self.about_pharmacy_refusals = about_pharmacy_refusals
        self.additional_info = additional_info 

    def not_labeled(self):
        return self.about_pharmacy_refusals == ""


class ClassificationResponse(BaseModel):
    """Class to parse the output of the LLM"""
    about_pharmaceutical_refusals: str = Field(description="""Answer with 'YES' if it is about this topic or 'NO' if it is not. **IMPORTANT**: If it is about pharmaceutical refusals, but does not meet all three conditions, you must **ALWAYS** return 'NO'.""") #  but does not meet all three conditions, answer ‘Unclear’.
    additional_information: str = Field(description="Any extra context or details about the classification.")