import logging
import os
import json
import websockets
from fine_voicing.tools.constants import LOGGER_MAIN, OPENAI_REALTIME_BASE_URL, OPENAI_REALTIME_DEFAULT_MODEL, OPENAI_REALTIME_DEFAULT_VOICE, OPENAI_OBSERVED_EVENTS, ULTRAVOX_FIRST_SPEAKER_USER
    
class OpenAIRealtimeClient():
    def __init__(
        self, 
        api_key: str = None,
        model: str = OPENAI_REALTIME_DEFAULT_MODEL,
        instructions: str = '',
        voice: str = OPENAI_REALTIME_DEFAULT_VOICE,
        first_speaker: str = ULTRAVOX_FIRST_SPEAKER_USER,
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.logger = logger
        self.instructions = instructions
        self.session_updated = False
        self.ws = None
        self.first_speaker = first_speaker
    async def connect(self):
        """Establish WebSocket connection with OpenAI Realtime API."""
        if not self.ws:
            self.logger.info("Opening websocket connection to OpenAI")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1"
            }
            self.ws = await websockets.connect(
                f"{OPENAI_REALTIME_BASE_URL}?model={self.model}",
                additional_headers=headers,
                ping_interval=None,
                logger=self.logger
            )

    async def disconnect(self):
        if self.ws:
            self.logger.info("Closing websocket connection to OpenAI")
            await self.ws.close()
            self.ws = None

    async def update_session(self):
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

    async def send_message(self, message: dict):
        """Send a message through the WebSocket connection synchronously."""
        self.logger.info(f"Sending message to OpenAI: {message}")
        
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
                "modalities": ["text"]
            }
        }
        await self.ws.send(json.dumps(conversation_event))
        await self.ws.send(json.dumps(response_event))

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
                    elif event_type == "error":
                        error_message = data.get("error", {}).get("message", "Unknown error")
                        self.logger.error(f"Error from OpenAI Realtime API: {error_message}")
                        raise Exception(f"OpenAI Realtime API error: {error_message}")
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            
        return full_response