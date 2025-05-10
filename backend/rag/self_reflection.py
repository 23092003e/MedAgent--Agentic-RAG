from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from .exceptions import ModelError

class SelfReflectionChain:
    def __init__(self, llm: Ollama):
        self.llm = llm
        self.reflection_prompt = PromptTemplate(
            template="""Analyze the following medical response for accuracy and completeness:

Response to analyze: {response}
Source documents: {sources}

Follow these steps:
1. Verify if all claims are supported by sources
2. Check for any medical inconsistencies
3. Identify missing important information
4. Assess confidence level (0-100%)
5. Suggest improvements if needed

Output format:
- Verified claims:
- Unsupported claims:
- Missing information:
- Medical accuracy:
- Confidence score:
- Suggested improvements:
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
                "unsupported_claims": [],
                "missing_information": [],
                "medical_accuracy": "",
                "confidence_score": 0,
                "suggested_improvements": []
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("- Verified claims:"):
                    current_section = "verified_claims"
                elif line.startswith("- Unsupported claims:"):
                    current_section = "unsupported_claims"
                elif line.startswith("- Missing information:"):
                    current_section = "missing_information"
                elif line.startswith("- Medical accuracy:"):
                    current_section = "medical_accuracy"
                elif line.startswith("- Confidence score:"):
                    try:
                        score = int(line.split(":")[1].strip().rstrip("%"))
                        analysis["confidence_score"] = score
                    except (ValueError, IndexError):
                        analysis["confidence_score"] = 0
                elif line.startswith("- Suggested improvements:"):
                    current_section = "suggested_improvements"
                elif current_section and line.startswith("- "):
                    if isinstance(analysis[current_section], list):
                        analysis[current_section].append(line[2:])
                    else:
                        analysis[current_section] = line[2:]
                        
            return analysis
            
        except Exception as e:
            raise ModelError(f"Failed to parse reflection: {str(e)}") from e
    
    def _improve_response(self, original_response: str, analysis: Dict) -> Optional[str]:
        """Generate improved response based on analysis"""
        try:
            if not analysis["missing_information"] and not analysis["unsupported_claims"] and not analysis["suggested_improvements"]:
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