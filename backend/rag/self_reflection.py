from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from .exceptions import ModelError
import logging

logger = logging.getLogger(__name__)

class SelfReflectionChain:
    def __init__(self, llm: Ollama):
        self.llm = llm
        self.reflection_prompt = PromptTemplate(
            template="""Analyze the following medical response for accuracy and completeness:

Response to analyze: {response}
Source documents: {sources}

Follow these steps and provide your analysis in the exact format below:

Verified claims:
- [list verified claims here]

Missing information:
- [list missing information here]

Suggested improvements:
- [list suggested improvements here]

Confidence: [0-100]%
""",
            input_variables=["response", "sources"]
        )
        
        self.improvement_prompt = PromptTemplate(
            template="""Improve the following medical response based on the analysis:

Original response: {response}

Analysis:
- Missing information: {missing_info}
- Unsupported claims: {unsupported_claims}
- Suggested improvements: {improvements}

Please provide an improved response that:
1. Addresses the missing information
2. Removes or clarifies unsupported claims
3. Implements the suggested improvements
4. Maintains a professional medical tone

Improved response:""",
            input_variables=["response", "missing_info", "unsupported_claims", "improvements"]
        )
        
    def analyze_response(self, response: str, source_docs: List[str]) -> Dict:
        """Analyze response quality and trustworthiness"""
        try:
            reflection = self.llm.invoke(
                self.reflection_prompt.format(
                    response=response,
                    sources="\n".join(source_docs)
                )
            )
            
            # Parse reflection output
            analysis = self._parse_reflection(reflection)
            
            # Determine if response needs improvement
            if analysis["confidence_score"] < 80 or analysis["missing_information"]:
                improved_response = self._improve_response(response, analysis)
                return {
                    "original_response": response,
                    "improved_response": improved_response,
                    "analysis": analysis
                }
            
            return {
                "original_response": response,
                "improved_response": None,
                "analysis": analysis
            }
        except Exception as e:
            raise ModelError(f"Failed to analyze response: {str(e)}") from e
    
    def _parse_reflection(self, reflection: str) -> Dict:
        """Parse reflection output into structured format"""
        try:
            lines = reflection.strip().split("\n")
            analysis = {
                "verified_claims": [],
                "missing_information": [],
                "suggested_improvements": [],
                "confidence_score": 0
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for section headers
                if line.lower().startswith("verified claims:"):
                    current_section = "verified_claims"
                    continue
                elif line.lower().startswith("missing information:"):
                    current_section = "missing_information"
                    continue
                elif line.lower().startswith("suggested improvements:"):
                    current_section = "suggested_improvements"
                    continue
                elif line.lower().startswith("confidence:"):
                    try:
                        # Extract number from "Confidence: X%"
                        score_text = line.split(":")[1].strip()
                        score = int(''.join(filter(str.isdigit, score_text)))
                        analysis["confidence_score"] = score
                    except (ValueError, IndexError):
                        analysis["confidence_score"] = 0
                    continue
                    
                # Add items to current section
                if current_section and line.startswith("-"):
                    item = line[1:].strip()
                    if item and item != "[list verified claims here]" and \
                       item != "[list missing information here]" and \
                       item != "[list suggested improvements here]":
                        analysis[current_section].append(item)
                        
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing reflection: {str(e)}")
            return {
                "verified_claims": [],
                "missing_information": [],
                "suggested_improvements": [],
                "confidence_score": 0
            }
    
    def _improve_response(self, original_response: str, analysis: Dict) -> Optional[str]:
        """Generate improved response based on analysis"""
        try:
            if not analysis["missing_information"] and not analysis["suggested_improvements"]:
                return None
                
            improved = self.llm.invoke(
                self.improvement_prompt.format(
                    response=original_response,
                    missing_info="\n".join(analysis["missing_information"]),
                    unsupported_claims="\n".join(analysis["unsupported_claims"]),
                    improvements="\n".join(analysis["suggested_improvements"])
                )
            )
            
            return improved.strip()
            
        except Exception as e:
            raise ModelError(f"Failed to improve response: {str(e)}") from e