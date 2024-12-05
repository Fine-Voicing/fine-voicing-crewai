LOGGER_MAIN = "fine-voicing"
LOGGER_TEST_CASE_FILE_PATTERN = "test_case_{test_case_name}_{timestamp}.log"
LOGGER_SESSION_FILE_PATTERN = "test_session_{timestamp}.log"
TEST_CASES_DIR='../test-cases'

OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1/realtime"
OPENAI_REALTIME_DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-10-01"
OPENAI_REALTIME_DEFAULT_VOICE = "alloy"
OPENAI_OBSERVED_EVENTS = ['response.done', 'error']