classify_chat_prompt = """
    We are searching for specific examples of pharmaceutical refusals, defined as the refusal to fulfill a prescription medication at a pharmacy by a pharmacist based on religious or moral objections. Our current corpus contains news articles and legal cases.
    To qualify as a specific example of a pharmaceutical refusal, the news article or legal case must meet all of the following conditions:
    1. Involve a specific person who was refused a prescription at a pharmacy (the name is not necessary).
    2. Mention the drug or type of drug that was refused (e.g., emergency contraception, birth control, abortion medication, hormones, HIV medication, etc.).
    3. State that the refusal was based on moral or religious grounds. It may also relate to an alternative conscientious objection.
    4. The case must have occurred within the United States. 

    Based on these conditions, read each of the attached documents and determine if they describe specific instances of prescriptions being refused on moral or religious grounds.    

    **IMPORTANT**: Unless **EVERY** condition is satisfied, it is **NOT** about pharmacy refusals. 

    **Examples**:
    - A document about a person who was refused a prescription at a pharmacy in Italy due to moral reasons => NOT about pharmacy refusals since it occurred in Italy, not the United States.
    - A document about a person who was refused a prescription at a pharmacy in America, but the reason for the refusal is unclear => NOT about pharmacy refusals since the refusal must be based on moral or religious grounds.
    - A document that mentions some pharmacists refuse to dispense certain drugs based on moral or religious grounds, but no specific person is mentioned => NOT about pharmacy refusals since the case must involve a specific person (the name is not necessary) who was refused a prescription at a pharmacy.
    - A document about pharmacy refusals based on religious grounds but with no **SPECIFIC** examples provided => NOT about pharmacy refusals since the document MUST involve a specific person.
    - A document about pharmacy refusals involving a specific person in America who was refused abortion medication based on religious grounds => This document is valid and about pharmacy refusals. 

    Answer based on the following document: {source}. Do not include any other information in your answer.
"""
