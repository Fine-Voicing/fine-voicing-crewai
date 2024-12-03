from crewai.tools import BaseTool
import websockets
import json
import asyncio
import os
from constants import LOGGER_MAIN
import logging

class OpenAIRealtime(BaseTool):
    name: str = "OpenAI Realtime"
    description: str = "Use the OpenAI Realtime API to generate a single response for a conversation."
    def __init__(
        self, 
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        instructions: str = "You are a helpful assistant",
        logger: logging.Logger = None
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.ws = None
        self.extra_event_handlers = {}
        self.on_response_done = None
        self.logger = logger or logging.getLogger(LOGGER_MAIN)

    async def connect(self) -> None:
        """Establish WebSocket connection with the Realtime API."""
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        self.ws = await websockets.connect(url, additional_headers=headers)
        await self._update_session()

    async def _update_session(self) -> None:
        """Update session configuration."""

        session_config = {
            "turn_detection": None,
            "voice": self.voice,
            "instructions": self.instructions,
            "modalities": ["text"],
            "temperature": 0.7
        }

        event = {
            "type": "session.update",
            "session": session_config
        }
        await self.ws.send(json.dumps(event))

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.logger.info("WebSocket connection to OpenAI Realtime closed.")

    async def _create_response(self) -> None:
        """Request a response from the API. Needed when using manual mode."""
        event = {
            "type": "response.create",
            "response": {
                "modalities": ["text"]
            }
        }
            
        self.logger.info("Request OpenAI to generate response...")
        await self.ws.send(json.dumps(event))

    async def _send_text(self, text):
        """Send a prompt and receive a response."""
        if not self.ws:
            raise Exception("WebSocket connection is not established.")

        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        }

        self.logger.info(f"Sending text to OpenAI realtime: {text}")
        await self.ws.send(json.dumps(event))
        await self._create_response()

    async def _handle_messages(self):
        while True:
            try:
                message = await self.ws.recv()
                event = json.loads(message)
                self.logger.debug(f"Received event: {event}")
                if event["type"] == "response.done":
                    if self.on_response_done:
                        self.on_response_done(event['response']['output'][0]['content']['transcript'])
                    break
                elif event["type"] in self.extra_event_handlers:
                    await self.extra_event_handlers[event["type"]]()
            except websockets.ConnectionClosed:
                break

    async def _run(self, argument: str) -> str:
        """
        Uses the OpenAI Realtime API to generate a single response for a conversation.

        Args:
            argument (str): The message to send.
        Returns:
            str: The OpenAI Realtime API response.
        """
        response_text = []
        response_complete = asyncio.Event()
        
        def on_response_done(transcript: str):
            self.logger.info('OpenAI Realtime Transcript:', transcript)
            response_text.append(transcript)
            response_complete.set()
        
        try:
            # Set up callbacks
            self.ws.on_response_done = on_response_done
            
            # Connect and start message handling
            message_handler = asyncio.create_task(self._handle_messages())
            
            # Send the text and wait for response
            await self._send_text(message)
            await response_complete.wait()
            
            return "".join(response_text)
            
        except Exception as e:
            raise Exception(f"Error in realtime chat: {str(e)}")