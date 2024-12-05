from crewai.tools import BaseTool
from typing import Type
from pydantic import Field, ConfigDict, BaseModel
from fine_voicing.tools.constants import LOGGER_MAIN
from fine_voicing.tools.voice_ai_model_thread import VoiceAIModelThread

class OpenAIRealtimeInput(BaseModel):
    """Input schema for OpenAIRealtime."""
    role_name: str = Field(..., description="The role to play in the conversation")
    last_message: str = Field(..., description="The last message in the conversation")

class OpenAIRealtimeTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "OpenAI Realtime API Client"
    description: str = "Use the OpenAI Realtime API to generate a message"
    
    args_schema: Type[BaseModel] = OpenAIRealtimeInput

    def _run(self, role_name: str, last_message: str) -> str:
        """Run the tool synchronously.
        
        Args:
            role_name (str): The name of the role to play in the conversation
            last_message (str): The last message in the conversation
            
        Returns:
            str: The AI's response
        """
        return f"{role_name}: {VoiceAIModelThread.get_instance().send_message(last_message)}"