from pydantic_xml import BaseXmlModel, element
from pydantic import BaseModel

# --- Pydantic Model for Structured Output ---
class Claim (BaseModel):
    claim: str
    reasoning: str
    is_claim_verified: bool

class FactualityResponse(BaseModel):
    claims: list[Claim]
    reasoning: str
    answer: bool

class FactualityResponseBasic(BaseModel):
    reasoning: str
    answer: bool

class FactualityResponseNoTTC(BaseModel):
    answer: bool

class FactualityResponseBasicXML(BaseXmlModel, tag="factuality_response"):
    reasoning: str = element(tag="reasoning")
    answer: bool = element(tag="answer")

class FactualityResponseXMLNoTTC(BaseXmlModel, tag="factuality_response"):
    answer: bool = element(tag="answer")

class AtomicClaim(BaseXmlModel, tag="atomic_claim"):
    claim: str              = element(tag="claim")
    claim_reasoning: str    = element(tag="claim_reasoning")
    is_claim_verified: bool = element(tag="is_claim_verified")

class FactualityResponseXML(BaseXmlModel, tag="factuality_response"):
    # ← Look for ALL <atomic_claim> children, anywhere under the root
    claims: list[AtomicClaim] = element(tag="atomic_claim", default_factory=list)
    reasoning: str            = element(tag="reasoning")
    answer: bool              = element(tag="answer")