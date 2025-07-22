FACTUALITY_SYSTEM_PROMPT_3D_WITH_TTC = """Given a conversation and a short answer, verify the short answer by referencing the conversation. First, break down the short answer into claims using the `A Step to Extract Claims` below. Next, verify each part of the claim and the relation between each part of the claim using `Steps to Evaluate Each Claim` below.

## A Step to Extract Claims ##
Step 1: Identify claims from the short answer. Example: "Customer was annoyed about slow delivery" -> "There was a delivery", "The delivery was slow", "Customer was annoyed", "Customer was annoyed specifically about slow delivery"

## Steps to Evaluate Each Claim ##
Step 2: In each claim, identify words that have concrete meanings. Example: "There was a delivery" -> "delivery". Verify those words by finding explicit mentions or references. When a word or a phrase can be interpreted in more than one way, see if at least one interpretation can be verified. Example: If a conversation includes discussions of receiving email notifications, this verifies one meaning of "delivery".
Step 3: In each claim, identify words that subjectively describe other words having concrete meanings. These words often describe a product or a service. Example: "The delivery was slow" -> "slow". Verify these words loosely with the context of the conversation.
Step 4: In each claim, identify words that are about subjective interpretation of the conversation. These words often describe sentiments and emotions from a third-person point of view. Example: "Customer was annoyed" -> "annoyed". Verify these words by finding minimal implicit evidence. Example: "annoyed" is verified with implicit evidence reflecting negative sentiment. 
Step 5: In each claim, verify the relation between words. Focus on verifying the relation between words, while ignoring the verifications of the words themselves in this step. Verify the relation with explicit evidence or by inferring the reason behind an action or a message. Example: "Customer was annoyed specifically about slow delivery" -> Verify that the source of a customer's sentiment was indeed the "slow delivery" while ignoring the verifications of "slow" and "annoyed". If a customer asks about filing a complaint after discussing slow delivery without explicitly expressing a negative sentiment, the customer must have been annoyed by the slow delivery. This inferred reason behind the customer's action verifies the relation.

## Output for Each Claim as JSON ## 
1. claim: The exact claim (as identified in Step 1). 
2. reasoning: A detailed reasoning for whether the claim was verified or not.
3. is_claim_verified: True if the claim was verified in Steps 2, 3, 4 and 5; otherwise False.

## Output Format as JSON ##
claims: list of all the claims generated above in the mentioned format.
reasoning: A concise summary of the reasoning for the final answer.
answer: True or False (True if short_answer is verified; otherwise, False)."""

FACTUALITY_SYSTEM_PROMPT_BASIC_WITH_TTC = """Given a conversation and a claim about that conversation, determine if the claim is factual, i.e., supported by the conversation.

### Output Format as JSON:
reasoning: A concise summary of the reasoning for the final answer.
answer: True or False (True if the claim is factual; otherwise, False)."""

FACTUALITY_SYSTEM_PROMPT_3D_NO_TTC = """Given a conversation and a short answer, verify the short answer by referencing the conversation. First, break down the short answer into claims using the `A Step to Extract Claims` below. Next, verify each part of the claim and the relation between each part of the claim using `Steps to Evaluate Each Claim` below.

## A Step to Extract Claims ##
Step 1: Identify claims from the short answer. Example: "Customer was annoyed about slow delivery" -> "There was a delivery", "The delivery was slow", "Customer was annoyed", "Customer was annoyed specifically about slow delivery"

## Steps to Evaluate Each Claim ##
Step 2: In each claim, identify words that have concrete meanings. Example: "There was a delivery" -> "delivery". Verify those words by finding explicit mentions or references. When a word or a phrase can be interpreted in more than one way, see if at least one interpretation can be verified. Example: If a conversation includes discussions of receiving email notifications, this verifies one meaning of "delivery".
Step 3: In each claim, identify words that subjectively describe other words having concrete meanings. These words often describe a product or a service. Example: "The delivery was slow" -> "slow". Verify these words loosely with the context of the conversation.
Step 4: In each claim, identify words that are about subjective interpretation of the conversation. These words often describe sentiments and emotions from a third-person point of view. Example: "Customer was annoyed" -> "annoyed". Verify these words by finding minimal implicit evidence. Example: "annoyed" is verified with implicit evidence reflecting negative sentiment. 
Step 5: In each claim, verify the relation between words. Focus on verifying the relation between words, while ignoring the verifications of the words themselves in this step. Verify the relation with explicit evidence or by inferring the reason behind an action or a message. Example: "Customer was annoyed specifically about slow delivery" -> Verify that the source of a customer's sentiment was indeed the "slow delivery" while ignoring the verifications of "slow" and "annoyed". If a customer asks about filing a complaint after discussing slow delivery without explicitly expressing a negative sentiment, the customer must have been annoyed by the slow delivery. This inferred reason behind the customer's action verifies the relation.

## Output Format as JSON ##
answer: True or False (True if short_answer is verified; otherwise, False)."""

FACTUALITY_SYSTEM_PROMPT_BASIC_NO_TTC = """Given a conversation and a claim about that conversation, determine if the claim is factual, i.e., supported by the conversation.

### Output Format as JSON:
answer: True or False (True if claim is factual; otherwise, False)."""

XML_OUTPUT_SYSTEM_PROMPT_WITH_TTC_ADDITION = """The XML output should be in the following format:
<factuality_response>
    <atomic_claim>
        <claim>...</claim>
        <claim_reasoning>...</claim_reasoning>
        <is_claim_verified>...</is_claim_verified>
    </atomic_claim>
    <atomic_claim>
        <claim>...</claim>
        <claim_reasoning>...</claim_reasoning>
        <is_claim_verified>...</is_claim_verified>
    </atomic_claim>
    ...
    <reasoning>...</reasoning>
    <answer>...</answer>
</factuality_response>"""

XML_OUTPUT_SYSTEM_PROMPT_NO_TTC_ADDITION = """The XML output should be in the following format:
<factuality_response>
    <answer>...</answer>
</factuality_response>"""

XML_OUTPUT_BASIC_SYSTEM_PROMPT_WITH_TTC_ADDITION = """The XML output should be in the following format:
<factuality_response>
    <reasoning>...</reasoning>
    <answer>...</answer>
</factuality_response>"""

XML_OUTPUT_BASIC_SYSTEM_PROMPT_NO_TTC_ADDITION = """The XML output should be in the following format:
<factuality_response>
    <answer>...</answer>
</factuality_response>"""


FACTUALITY_USER_PROMPT = """### Conversation ###
{conversation}

### Short answer ###
{short_answer}"""