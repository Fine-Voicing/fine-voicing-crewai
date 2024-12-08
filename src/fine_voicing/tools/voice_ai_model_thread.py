import asyncio
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
    def __init__(self, instructions: str = '', 
                 logger: logging.Logger = logging.getLogger(LOGGER_MAIN),
                 provider: Provider = Provider.OPENAI, 
                 first_speaker: str = ULTRAVOX_FIRST_SPEAKER_USER):
        self.instructions = instructions
        self.logger = logger
        self.provider = provider
        self.first_speaker = first_speaker
        self.client = None
        self._loop = None
        self._thread = None
        self._loop_initialized = threading.Event()  # Added for synchronization
        self._setup_thread()

    def _setup_thread(self):
        """Setup the dedicated thread and event loop for async operations."""
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        self._loop_initialized.wait()  # Wait until the event loop is initialized

    def _run_event_loop(self):
        """Run the event loop in the dedicated thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_initialized.set()  # Signal that the loop is initialized
        self._loop.run_forever()

    def _run_coroutine(self, coro):
        """Helper method to run coroutines in the dedicated event loop."""
        if threading.current_thread() is self._thread:
            return self._loop.run_until_complete(coro)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def send_message(self, message: str) -> str:
        """Thread-safe method to send messages."""
        async def _send():
            if self.client is None:
                await self._initialize()
            return await self.client.send_message(message)
        
        return self._run_coroutine(_send())

    async def _initialize(self):
        """Initialize the AI client."""
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

    def stop(self):
        """Stop the event loop and clean up."""
        async def _cleanup():
            if self.client is not None:
                try:
                    await self.client.disconnect()
                except Exception as e:
                    self.logger.error(f"Error during client disconnect: {e}")
                self.client = None

        try:
            self._run_coroutine(_cleanup())
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        """Ensure resources are cleaned up when the instance is garbage collected."""
        if self._loop is not None and self._loop.is_running():
            self.stop()

