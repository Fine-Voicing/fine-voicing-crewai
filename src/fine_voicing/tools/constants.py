LOGGER_MAIN = "fine-voicing"
LOGGER_TEST_CASE_FILE_PATTERN = "test_case_{test_case_name}_{timestamp}.log"
LOGGER_SESSION_FILE_PATTERN = "test_session_{timestamp}.log"
TEST_CASES_DIR='../../test-cases'

OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"
OPENAI_REALTIME_DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
OPENAI_REALTIME_DEFAULT_VOICE = "alloy"
OPENAI_OBSERVED_EVENTS = ['response.done', 'error']

ULTRAVOX_BASE_URL = "https://api.ultravox.ai"
ULTRAVOX_DEFAULT_MODEL = 'fixie-ai/ultravox'
ULTRAVOX_DEFAULT_VOICE = 'Mark'
ULTRAVOX_OBSERVED_EVENTS = ['transcript']

ULTRAVOX_FIRST_SPEAKER_USER = 'FIRST_SPEAKER_USER'
ULTRAVOX_FIRST_SPEAKER_AGENT = 'FIRST_SPEAKER_AGENT'
EMPTY_HISTORY = '[EMPTY HISTORY]'