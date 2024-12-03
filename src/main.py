from test_runner import TestRunner
from tools import utils
from argparse import ArgumentParser
from constants import LOGGER_MAIN, LOGGER_SESSION_FILE_PATTERN
import asyncio

parser = ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true',
                    help='Enable debug logging level')
args = parser.parse_args()

async def main():
    logger = utils.setup_logging(LOGGER_MAIN, debug=args.debug, file_pattern=LOGGER_SESSION_FILE_PATTERN, console_output=True)

    runner = TestRunner(debug=args.debug)
    transcripts = await runner.run_test_cases()
    logger.info(f"{len(transcripts)} test cases completed")

if __name__ == "__main__":
    asyncio.run(main())