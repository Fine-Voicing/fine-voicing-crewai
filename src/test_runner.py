import logging
import os
import json
from crewai import Agent, Task, Crew, Process
from tools import utils
import concurrent.futures
from typing import List, Dict
from constants import LOGGER_MAIN, TEST_CASES_DIR, LOGGER_TEST_CASE_FILE_PATTERN 

class TestRunner:
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(LOGGER_MAIN)
        self.conversation_generator = None
        self.moderator = None
        self.test_case_loggers = {}
        self.test_case_definitions = {}
        self._setup_agents()

    def run_test_cases(self, test_dir: str = TEST_CASES_DIR) -> Dict[str, List[str]]:
        test_cases_dir = os.path.join(os.path.dirname(__file__), test_dir)
        test_case_files = [os.path.join(test_cases_dir, f) for f in os.listdir(test_cases_dir) if os.path.isfile(os.path.join(test_cases_dir, f))]
        
        transcripts = {}
        futures = []
        future_to_name = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(test_case_files)) as executor:
            for test_case_file in test_case_files:
                test_case_name = os.path.splitext(os.path.basename(test_case_file))[0]
                self.test_case_loggers[test_case_name] = utils.setup_logging(test_case_name, debug=self.debug, file_pattern=LOGGER_TEST_CASE_FILE_PATTERN, test_case_name=test_case_name, console_output=False)
                with open(test_case_file) as f:
                    test_case = json.load(f)
                    self.test_case_definitions[test_case_name] = test_case
                    self.logger.info(f"--- Test case: {test_case_name} submitted for execution ---")
                    future = executor.submit(self.run_test_case, test_case_name)
                    future_to_name[future] = test_case_name
                    futures.append(future)
                    
            for future in concurrent.futures.as_completed(futures):
                test_case_name = future_to_name[future]
                transcript = future.result()
                transcripts[test_case_name] = transcript
                self.logger.info(f"--- Transcript for test case: {test_case_name} ---")
                [self.logger.info(line) for line in transcript]
                self.logger.info(f"--- End transcript for test case: {test_case_name} ---")
                self.logger.info(f"--- Test case: {test_case_name} completed ---")

        return transcripts

    def _setup_agents(self):
        self.conversation_generator = Agent(
            role="Conversation Voice Simulation Agent",
            goal="Generate a single response in the conversation. Voice conversation are often brief and likely to be interrupted by the other person",
            verbose=True,
            memory=True,
            backstory="You specialize in simulating human-like voice interactions, and refining AI prompts.",
        )

        self.moderator = Agent(
            role="Conversation Moderator",
            goal="Evaluate each message in the conversation and decide if the conversation should continue.",
            verbose=True,
            memory=True,
            backstory="You are an expert in evaluating and moderating conversations.",
        )

    def _generate_roles(self, test_case_name: str) -> dict:
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]
        
        generate_roles_task = Task(
            description=(
                "Instructions: {instructions}"
                "Based on the instructions, identify the roles (at least two) involved in the conversation."
                "Based on the instructions, generate a specific set of instructions for each role."
                "Format as a JSON array, where each item has keys: role_name, role_prompt. Output only valid JSON."
            ),
            expected_output="A JSON object with the roles and their descriptions.",
            agent=self.conversation_generator,
        )

        generate_roles_crew = Crew(
            agents=[self.conversation_generator],
            tasks=[generate_roles_task],
        )

        logger.info("Generating conversation roles and instructions")
        sRoles = generate_roles_crew.kickoff(test_case)
        logger.info(f"Conversation Roles: {sRoles}")

        roles = json.loads(str(sRoles))
        logger.debug(f"JSON deserialized roles: {roles}")

        return roles

    def run_test_case(self, test_case_name):
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]

        self.logger.info(f"--- Test case: {test_case_name} starting ---")
        logger.info(f"--- Test case: {test_case_name} starting ---")
        logger.debug(f"Test case definition: {test_case}")

        moderate_message = Task(
            description=(
                "Evaluate the conversation history up to this point: {chat_history} "
                "Decide if the conversation should continue or terminate. Provide reasoning for the decision."
                "Always write in English"
            ),
            expected_output="A decision to 'continue' or 'terminate', with reasoning.",
            agent=self.moderator,
        )

        moderate_crew = Crew(
            agents=[self.moderator],
            tasks=[moderate_message],
        )

        roles = self._generate_roles(test_case_name)
        generate_tasks = []
        for role in roles:
            logger.info(f"Generating task for role: {role}")
            generate_task = Task(
                    description=(
                        "Generate the next message in the conversation, based on the chat history: {chat_history}."
                        f"Play role in the conversation: {role['role_name']}."
                        f"Follow these instructions: {role['role_prompt']}."
                        f"Always format the message as <{role['role_name']}>: <message>."
                        "Do not use placeholders in the conversation, for instance for names."
                        f"Ensure the response is in {test_case['language']} and adheres to the context of the conversation."
                    ),
                    expected_output="A single conversational message, responding to the previous message.",
                    agent=self.conversation_generator,
                    context=generate_tasks[-1:] if generate_tasks else []
                )
            generate_tasks.append(generate_task)

        generate_crew = Crew(
            agents=[self.conversation_generator],
            tasks=generate_tasks,
            process=Process.sequential
        )

        return self._converse(test_case_name, generate_crew, moderate_crew)
    
    def _converse(self, test_case_name: str, generate_crew, moderate_crew):
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]

        transcript = []
        should_terminate = False
        index_turn = 1
        while not should_terminate and index_turn <= test_case['turns']:
            logger.info(f"--- Starting turn {index_turn} ---")
            result = generate_crew.kickoff({"chat_history": transcript})
            [transcript.append(task.raw) for task in result.tasks_output]
            logger.debug(f"--- Intermediary conversation transcript ---")
            [logger.debug(line) for line in transcript]
            logger.debug(f"--- End of intermediary conversation transcript ---")
            
            # Moderate message
            logger.info("Moderating conversation")
            decision = moderate_crew.kickoff({"chat_history": transcript})
            logger.info(f"Moderation Decision: {decision}")
            
            # Check if the moderator wants to terminate the conversation
            decision_str = str(decision)
            should_terminate = "terminate" in decision_str.lower()
            logger.debug(f"Conversation terminated by moderator after {index_turn} turns") if should_terminate else logger.debug(f"Conversation continuing after {index_turn} turns")
            logger.info(f"--- Ending turn {index_turn} ---")
            index_turn += 1

        logger.info("Conversation terminated by moderator") if should_terminate else logger.info(f"Conversation completed after {index_turn} turns")
        return transcript