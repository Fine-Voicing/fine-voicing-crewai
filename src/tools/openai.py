from crewai.tools import BaseTool
from typing import Type
from pydantic import Field, ConfigDict, BaseModel
import websockets
import json
import asyncio
import os
from constants import LOGGER_MAIN, OPENAI_REALTIME_BASE_URL, OPENAI_REALTIME_DEFAULT_MODEL, OPENAI_REALTIME_DEFAULT_VOICE, OPENAI_OBSERVED_EVENTS
import logging
import threading
import queue

class OpenAIRealtimeInput(BaseModel):
    """Input schema for OpenAIRealtime."""
    role_name: str = Field(..., description="The name of the role to play in the conversation")
    last_message: str = Field(..., description="The last message in the conversation")

class OpenAIRealtimeTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "OpenAI Realtime API Client"
    description: str = "Use the OpenAI Realtime API to generate a message"
    
    args_schema: Type[BaseModel] = OpenAIRealtimeInput

    def _run(self, role_name: str, last_message: str) -> str:
        """Run the tool synchronously.
        
        Args:
            last_message (str): The last message in the conversation
            
        Returns:
            str: The AI's response
        """
        return f"{role_name}: {OpenAIRealtimeClientThread(instructions='', logger=None).send_message(last_message)}"

class OpenAIRealtimeClient():
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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.logger = logger
        self.instructions = instructions
        self.session_updated = False
        self.ws = None
    async def connect(self):
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

    async def disconnect(self):
        if self.ws:
            self.logger.info("Closing websocket connection to OpenAI")
            self.ws.close()
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
    

class OpenAIRealtimeClientThread:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, instructions: str = '', logger: logging.Logger = logging.getLogger(LOGGER_MAIN)):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpenAIRealtimeClientThread, cls).__new__(cls)
                cls._instance.__init__(instructions, logger)  # Initialize the instance
        return cls._instance

    def __init__(self, instructions: str, logger: logging.Logger):
        if not hasattr(self, 'initialized'):  # Prevent re-initialization
            self.message_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.running = True
            self.thread = threading.Thread(target=self.run)
            
            self.thread.start()
            self.instructions = instructions
            self.logger = logger
            self.initialized = True  # Mark as initialized

    def run(self):
        """Run the asynchronous client in a separate thread."""
        loop = asyncio.new_event_loop()  # Create a new event loop
        asyncio.set_event_loop(loop)  # Set the new loop as the current loop
        loop.run_until_complete(self.async_run())  # Run the async method

    async def async_run(self):
        """Asynchronous method to handle messages."""
        self.client = OpenAIRealtimeClient(
            instructions=self.instructions,
            logger=self.logger
        )
        await self.client.connect()
        await self.client.update_session()

        while self.running:
            try:
                message = self.message_queue.get(timeout=1)  # Wait for a message
                response = await self.client.send_message(message)  # Call the asynchronous send_message
                self.response_queue.put(response)  # Put the response in the response queue
            except queue.Empty:
                continue  # Continue if no message is received

        await self.client.disconnect()

    def send_message(self, message):
        """Send a message to the OpenAIRealtime client."""
        self.message_queue.put(message)  # Put the message in the queue
        return self.response_queue.get()  # Wait for and return the response

    def stop(self):
        """Stop the thread."""
        self.running = False
        self.thread.join()  # Wait for the thread to finish
        
