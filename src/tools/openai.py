from crewai.tools import BaseTool
from typing import Type
from pydantic import Field, ConfigDict, BaseModel
import websockets
import json
import asyncio
import os
from constants import LOGGER_MAIN, OPENAI_REALTIME_BASE_URL, OPENAI_REALTIME_DEFAULT_MODEL, OPENAI_REALTIME_DEFAULT_VOICE, OPENAI_OBSERVED_EVENTS
import logging

class OpenAIRealtimeInput(BaseModel):
    """Input schema for OpenAIRealtime."""
    last_message: str = Field(..., description="The last message in the conversation")

class OpenAIRealtime(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "OpenAI Realtime API Client"
    description: str = "Use the OpenAI Realtime API to generate a message"
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    model: str = Field(default=OPENAI_REALTIME_DEFAULT_MODEL, description="OpenAI model to use")
    voice: str = Field(default=OPENAI_REALTIME_DEFAULT_VOICE, description="Voice to use for the conversation")
    instructions: str = Field(default="", description="Instructions for the AI")
    ws: websockets.WebSocketClientProtocol = Field(default=None, description="WebSocket connection")
    on_response_done: callable = Field(default=None, description="Callback function for response completion")
    logger: logging.Logger = Field(default=logging.getLogger(LOGGER_MAIN), description="A custom logger instance")
    session_updated: bool = Field(default=False, description="Flag to indicate if the OpenAI session has been be updated")

    args_schema: Type[BaseModel] = OpenAIRealtimeInput

    def __init__(
        self, 
        api_key: str = None,
        model: str = OPENAI_REALTIME_DEFAULT_MODEL,
        instructions: str = '',
        voice: str = OPENAI_REALTIME_DEFAULT_VOICE,
        logger: logging.Logger = logging.getLogger(LOGGER_MAIN)
    ):
        """Initialize the OpenAI Realtime tool.
        
        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable.
            model (str, optional): Model to use. Defaults to OPENAI_REALTIME_DEFAULT_MODEL.
            voice (str, optional): Voice to use. Defaults to OPENAI_REALTIME_DEFAULT_VOICE.
            logger (logging.Logger, optional): Logger instance. Defaults to main logger.
            instructions (str, optional): Instructions to provide to OpenAI.
        """
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            voice=voice,
            logger=logger,
            instructions=instructions
        )

    async def _connect(self):
        """Establish WebSocket connection with OpenAI Realtime API."""
        if not self.ws:
            self.logger.info("Closing websocket connection to OpenAI")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1"
            }
            self.ws = await websockets.connect(
                f"{OPENAI_REALTIME_BASE_URL}?model={self.model}",
                additional_headers=headers,
                ping_interval=None
            )

    async def _disconnect(self):
        if self.ws:
            self.logger.info("Closing websocket connection to OpenAI")
            self.ws.close()
            self.ws = None

    async def _update_session(self):
        if not self.session_updated:
            session_config = {
                "type": "session.update",
                "session": {
                    "turn_detection": None,
                    "voice": self.voice,
                    "instructions": self.instructions,
                    "modalities": ["text"],
                    "temperature": 0.7
                }
            }
            await self.ws.send(json.dumps(session_config))

            # Wait for session.update.completed event
            while True:
                response = await self.ws.recv()
                data = json.loads(response)
                self.logger.debug(f'Session update response: {data}')
                
                if data.get("type") == "session.updated":
                    break
                elif data.get("type") == "error":
                    error_message = data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Session update error: {error_message}")
        else:
            self.logger.info("OpenAI session already up-to-date")

    async def _send_message(self, message: dict):
        """Send a message through the WebSocket connection."""
                    # Send the conversation item
        conversation_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": message
                }]
            }
        }
        # Request response creation
        response_event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": self.instructions,
                "voice": self.voice
            }
        }
        await self.ws.send(json.dumps(conversation_event))
        await self.ws.send(json.dumps(response_event))

    async def _process_responses(self) -> str:
        """Process responses from the WebSocket connection."""
        full_response = ""
        try:
            while True:
                response = await self.ws.recv()
                data = json.loads(response)
                event_type = data.get("type")

                if event_type in OPENAI_OBSERVED_EVENTS:
                    self.logger.debug(f'Received event from OpenAI: {data}')

                    if event_type == "response.done":
                        response = data.get("response")
                        response_status = response.get("status")
                        if response_status == 'completed':
                            full_response = response['output'][0]['content'][0]['text']
                            self.logger.info(f"Response from OpenAI: {full_response}")
                        else:
                            self.logger.info(f"Error in the response from OpenAI")
                            full_response = '<Error>'
                        break
                    if event_type == "response.text.done":
                        message_content = data.get("text", "")
                        full_response += message_content
                        self.logger.info(f"Response from OpenAI: {full_response}")
                        break
                    elif event_type == "error":
                        error_message = data.get("error", {}).get("message", "Unknown error")
                        self.logger.error(f"Error from OpenAI Realtime API: {error_message}")
                        raise Exception(f"OpenAI Realtime API error: {error_message}")
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            
        return full_response

    async def _run_async(self, last_message: str) -> str:
        """Run the tool asynchronously."""
        try:
            breakpoint()
            await self._connect()
            await self._update_session()
            await self._send_message(last_message)
            response = await self._process_responses()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in OpenAI Realtime tool: {str(e)}")
            raise
        finally:
            await self._disconnect()

    def _run(self, last_message: str) -> str:
        """Run the tool synchronously.
        
        Args:
            last_message (str): The last message in the conversation
            
        Returns:
            str: The AI's response
        """
        return asyncio.run(self._run_async(last_message))
    

    def disconnect(self):
        return asyncio.run(self._disconnect())

    