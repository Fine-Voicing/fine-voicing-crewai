import logging
import os
import json
import websockets
import aiohttp
from fine_voicing.tools.constants import LOGGER_MAIN, ULTRAVOX_BASE_URL, ULTRAVOX_DEFAULT_MODEL, ULTRAVOX_DEFAULT_VOICE, ULTRAVOX_FIRST_SPEAKER_USER, ULTRAVOX_OBSERVED_EVENTS

class UltraVoxClient():
    def __init__(
        self, 
        api_key: str = None,
        model: str = ULTRAVOX_DEFAULT_MODEL,
        instructions: str = '',
        voice: str = ULTRAVOX_DEFAULT_VOICE,
        first_speaker: str = ULTRAVOX_FIRST_SPEAKER_USER,
        logger: logging.Logger = logging.getLogger(LOGGER_MAIN)
    ):
        """Initialize the Ultravox client.
        
        Args:
            api_key (str, optional): Ultravox API key. Defaults to environment variable.
            model (str, optional): Model to use. Defaults to ULTRAVOX_DEFAULT_MODEL.
            voice (str, optional): Voice to use. Defaults to ULTRAVOX_DEFAULT_VOICE.
            logger (logging.Logger, optional): Logger instance. Defaults to main logger.
            instructions (str, optional): System prompt to provide to Ultravox.
        """
        self.api_key = api_key or os.getenv("ULTRAVOX_API_KEY")
        self.model = model
        self.voice = voice
        self.logger = logger
        self.instructions = instructions
        self.session_updated = False
        self.ws = None
        self.first_speaker = first_speaker
    async def connect(self):
        """Establish WebSocket connection with Ultravox."""
        if not self.ws:
            join_url = await self._fetch_join_url()
            self.logger.info(f"Opening websocket connection to Ultravox at {join_url}")
            
            self.ws = await websockets.connect(
                join_url,
                logger=self.logger
            )

            set_output_medium = {
                "type": "set_output_medium",
                "medium": "text"
            }
            await self.ws.send(json.dumps(set_output_medium))

    async def _fetch_join_url(self):
        url = f'{ULTRAVOX_BASE_URL}/api/calls'
        headers = {
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "systemPrompt": self.instructions,
            "model": self.model,
            "voice": self.voice,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": 48000,
                    "outputSampleRate": 48000,
                    "clientBufferSizeMs": 30000
                }
            },
            "firstSpeaker": self.first_speaker
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_data = await response.json()
                join_url = response_data.get('joinUrl')
                return join_url

    async def disconnect(self):
        if self.ws:
            self.logger.info("Closing websocket connection to Ultravox")
            await self.ws.close()
            self.ws = None

    async def update_session(self):
        return True

    async def send_message(self, message: dict):
        """Send a message through the WebSocket connection synchronously."""
        self.logger.info(f"Sending message to Ultravox: {message}")

        conversation_event = {
            "type": "input_text_message",
            "text": message
        }

        await self.ws.send(json.dumps(conversation_event))

        full_response = ""
        try:
            while True:
                try:
                    response = await self.ws.recv()
                    data = json.loads(response)
                    event_type = data.get("type")
                    if event_type in ULTRAVOX_OBSERVED_EVENTS:
                        self.logger.debug(f"Received JSON response from Ultravox: {data}")
                        is_final = bool(data.get("final", False))
                        if is_final:
                            full_response = data.get("text", "")
                            break
                    else:
                        self.logger.debug(f"Received ignored event from Ultravox: {event_type}")
                except json.JSONDecodeError:
                    self.logger.error("Failed to decode JSON response")
                    data = {}
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection to Ultravox closed")
            
        return full_response