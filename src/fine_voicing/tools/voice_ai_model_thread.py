import asyncio
import queue
import threading
import logging
from enum import Enum
from fine_voicing.tools.constants import ULTRAVOX_FIRST_SPEAKER_USER, LOGGER_MAIN
from fine_voicing.tools.openai_realtime_client import OpenAIRealtimeClient
from fine_voicing.tools.ultravox_client import UltraVoxClient

class Provider(Enum):
    OPENAI = 'openai'
    ULTRAVOX = 'ultravox'

class VoiceAIModelThread:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, instructions: str = '', logger: logging.Logger = logging.getLogger(LOGGER_MAIN), provider: Provider = Provider.OPENAI, first_speaker: str = ULTRAVOX_FIRST_SPEAKER_USER):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VoiceAIModelThread, cls).__new__(cls)
                cls._instance.__init__(instructions, logger, provider, first_speaker)  # Initialize the instance
        return cls._instance
    
    @classmethod
    def get_instance(self):
        with self._lock:
            return self._instance

    def __init__(self, instructions: str, logger: logging.Logger, provider: str, first_speaker: str):
        if not hasattr(self, 'initialized'):  # Prevent re-initialization
            self.message_queue = queue.Queue()
            self.response_queue = queue.Queue()
            self.running = True
            self.thread = threading.Thread(target=self.run)
            
            self.thread.start()
            self.instructions = instructions
            self.logger = logger
            self.initialized = True  # Mark as initialized
            self.provider = provider
            self.first_speaker = first_speaker

    def run(self):
        """Run the asynchronous client in a separate thread."""
        loop = asyncio.new_event_loop()  # Create a new event loop
        asyncio.set_event_loop(loop)  # Set the new loop as the current loop
        loop.run_until_complete(self.async_run())  # Run the async method

    async def async_run(self):
        """Asynchronous method to handle messages."""
        if self.provider == Provider.OPENAI:
            self.client = OpenAIRealtimeClient(
                instructions=self.instructions,
                logger=self.logger
            )
        elif self.provider == Provider.ULTRAVOX:
            self.client = UltraVoxClient(
                instructions=self.instructions,
                logger=self.logger
            )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")
        
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

