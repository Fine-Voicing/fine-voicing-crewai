from crewai.tools import BaseTool
from typing import Type
from pydantic import Field, ConfigDict, BaseModel
from fine_voicing.tools.constants import LOGGER_MAIN
from fine_voicing.tools.voice_ai_model_thread import VoiceAIModelThread

class VoiceAIToolInput(BaseModel):
    """Input schema for OpenAIRealtime."""
    role_name: str = Field(..., description="The role to play in the conversation")
    last_message: str = Field(..., description="The last message in the conversation")

class VoiceAIBaseTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Voice AI API Generic Client"
    description: str = "Use a Voice AI API to generate a message"
    voiceai_thread: VoiceAIModelThread = Field(..., description="The thread to use to communicate with the Voice AI API client")
    
    args_schema: Type[BaseModel] = VoiceAIToolInput

    def __init__(self, result_as_answer, voiceai_thread):
        super().__init__(result_as_answer=result_as_answer, voiceai_thread=voiceai_thread)

    def _run(self, role_name: str, last_message: str) -> str:
        """Run the tool synchronously.
        
        Args:
            role_name (str): The name of the role to play in the conversation
            last_message (str): The last message in the conversation
            
        Returns:
            str: The AI's response
        """
        return f"{role_name}: {self.voiceai_thread.send_message(last_message)}"
    
class OpenAIVoiceAI(VoiceAIBaseTool):
    name: str = "OpenAI Voice AI API Client"
    description: str = "Use the OpenAI Voice AI API to generate a message"

    def __init__(self, result_as_answer, voiceai_thread):
        super().__init__(result_as_answer=result_as_answer, voiceai_thread=voiceai_thread)

class UltravoxVoiceAI(VoiceAIBaseTool):
    name: str = "Ultravox Voice AI API Client"
    description: str = "Use the Ultravox Voice AI API to generate a message"

    def __init__(self, result_as_answer, voiceai_thread):
        super().__init__(result_as_answer=result_as_answer, voiceai_thread=voiceai_thread)