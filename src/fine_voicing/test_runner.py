import logging
import os
import json
from crewai import Agent, Task, Crew, Process
import asyncio
from typing import List, Dict
from fine_voicing.tools.constants import LOGGER_MAIN, TEST_CASES_DIR, LOGGER_TEST_CASE_FILE_PATTERN, ULTRAVOX_FIRST_SPEAKER_USER, EMPTY_HISTORY
from fine_voicing.tools.voice_ai_model_thread import VoiceAIModelThread, Provider
from fine_voicing.tools import utils, voice_ai

class FineVoicingAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = kwargs.get('logger', logging.getLogger(LOGGER_MAIN))

class TestRunner:
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = logging.getLogger(LOGGER_MAIN)
        self.test_case_loggers = {}
        self.test_case_definitions = {}

    async def run_test_cases(self, test_dir: str = TEST_CASES_DIR) -> Dict[str, List[str]]:
        test_cases_dir = os.path.join(os.path.dirname(__file__), test_dir)
        test_case_files = [os.path.join(test_cases_dir, f) for f in os.listdir(test_cases_dir) if os.path.isfile(os.path.join(test_cases_dir, f)) and not f.endswith('example.json')]
        
        transcripts = {}
        tasks = []

        for test_case_file in test_case_files:
            test_case_name = os.path.splitext(os.path.basename(test_case_file))[0]
            self.test_case_loggers[test_case_name] = utils.setup_logging(test_case_name, debug=self.debug, file_pattern=LOGGER_TEST_CASE_FILE_PATTERN, test_case_name=test_case_name, console_output=False)
            with open(test_case_file) as f:
                test_case = json.load(f)
                self.test_case_definitions[test_case_name] = test_case
                self.logger.info(f"--- Test case: {test_case_name} submitted for execution ---")
                tasks.append(self.run_test_case(test_case_name))
        
        results = await asyncio.gather(*tasks)
        
        for test_case_name, transcript in zip([os.path.splitext(os.path.basename(f))[0] for f in test_case_files], results):
            transcripts[test_case_name] = transcript
            self.logger.info(f"--- Transcript for test case: {test_case_name} ---")
            [self.logger.info(line) for line in transcript]
            self.logger.info(f"--- End transcript for test case: {test_case_name} ---")
            self.logger.info(f"--- Test case: {test_case_name} completed ---")

        return transcripts

    def _setup_agents(self, test_case_name: str) -> Dict[str, FineVoicingAgent]:
        logger = self.test_case_loggers[test_case_name]
        
        voice_ai_model_agent = FineVoicingAgent(
            role="Voice AI Model Agent",
            goal="Use the Voice AI Model to generate conversations.",
            verbose=True,
            memory=True,
            backstory="You specialize in using Voice AI Models to generate conversational messages.",
            logger=logger,
        )

        conversation_generator = FineVoicingAgent(
            role="Conversation Voice Simulation Agent",
            goal="Generate conversational messages in a voice conversation with an AI assistant.",
            verbose=True,
            memory=True,
            backstory="You specialize in simulating human-like voice interactions, and refining AI prompts.",
            logger=logger,
        )

        moderator = FineVoicingAgent(
            role="Conversation Moderator",
            goal="Evaluate each message in the conversation and decide if the conversation should continue.",
            verbose=True,
            memory=True,
            backstory="You are an expert in evaluating and moderating conversations.",
            logger=logger,
        )

        return {
            'voice_ai_model_agent': voice_ai_model_agent,
            'conversation_generator': conversation_generator,
            'moderator': moderator
        }

    def _generate_roles(self, agents: Dict[str, FineVoicingAgent], test_case_name: str) -> dict:
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]

        conversation_roles_agent = FineVoicingAgent(
            role="Conversation Roles Generator",
            goal="Generate the roles and instructions for a conversation based on the provided instructions.",
            verbose=True,
            memory=True,
            backstory="You specialize in generating the roles and instructions for a conversation based on the provided instructions.",
            logger=logger,
        )
        
        generate_roles_task = Task(
            description=(
                "Instructions: {instructions}"
                "Based on the instructions, identify the roles (at least two) involved in the conversation."
                "Based on the instructions, generate a specific set of instructions for each role. Make sure to include the target language in the instructions."
                "Based on the instructions, identify the role to be tested amongst the roles you've identified."
                "Format as a JSON array, where each item has keys: role_name, role_prompt, is_tested_role. Output only valid JSON. No markdown or other formatting."
            ),
            expected_output="A JSON object with the roles and their descriptions.",
            agent=conversation_roles_agent,
        )

        generate_roles_crew = Crew(
            agents=[agents['conversation_generator']],
            tasks=[generate_roles_task],
        )

        logger.info("Generating conversation roles and instructions")
        sRoles = generate_roles_crew.kickoff(test_case)
        logger.info(f"Conversation Roles: {sRoles}")

        roles = json.loads(str(sRoles))
        logger.debug(f"JSON deserialized roles: {roles}")

        return roles

    async def run_test_case(self, test_case_name):
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]
        agents = self._setup_agents(test_case_name)

        self.logger.info(f"--- Test case: {test_case_name} starting ---")
        logger.info(f"--- Test case: {test_case_name} starting ---")
        logger.debug(f"Test case definition: {test_case}")

        moderate_task = Task(
            description=(
                "Evaluate the conversation history up to this point: {chat_history} "
                "Decide if the conversation should continue or terminate. Provide reasoning for the decision."
                "Always write in English"
            ),
            expected_output="A decision to 'continue' or 'terminate', with reasoning.",
            agent=agents['moderator'],
        )

        roles = self._generate_roles(agents, test_case_name)

        tested_role = None
        testing_role = None
        for role in roles:
            if role.get('is_tested_role', False):
                tested_role = role
            else:
                testing_role = role

        if not tested_role or not testing_role:
            raise ValueError("No tested role or testing role identified")

        provider = Provider(test_case['voice_model']['provider'])
        voiceai_thread = VoiceAIModelThread(tested_role['role_prompt'], logger, provider=provider, first_speaker=ULTRAVOX_FIRST_SPEAKER_USER)

        generate_task_tested = Task(
            description=(
                f"Use the {provider.value} Voice AI API Client tool to generate the next message in the conversation."
                f"Provide the role_name parameter of the {provider.value} Voice AI API Client tool: {tested_role['role_name']}"
                f"For Ultravox, when the chat history is {EMPTY_HISTORY}, last_message should be a message generated from the role prompt {tested_role['role_prompt']}."
                f"Otherwise, provide the last_message parameter of the {provider.value} Voice AI API Client tool from the chat history."
                "Chat history, each message is prefixed with a dash (-):"
                "{chat_history}"
            ),
            expected_output=f"The response from the {provider.value} Client tool.",
            agent=agents['voice_ai_model_agent'],
            tools=[voice_ai.OpenAIVoiceAI(result_as_answer=False, voiceai_thread=voiceai_thread), voice_ai.UltravoxVoiceAI(result_as_answer=False, voiceai_thread=voiceai_thread)],
        )
        generate_task_testing = Task(
            description=(
                "Generate the next message in the conversation, based on the chat history." 
                f"Play role in the conversation: {testing_role['role_name']}."
                f"Follow these instructions: {testing_role['role_prompt']}."
                f"Prefix all messages with the role name: {testing_role['role_name']}."
                f"Ensure the response is in {test_case['language']} and adheres to the context of the conversation."
                "Chat history, each message is prefixed with a dash (-):"
                "{chat_history}"
            ),
            expected_output="A single conversational message, responding to the previous message.",
            agent=agents['conversation_generator'],
        )

        transcript = self._converse(test_case_name, agents, generate_task_tested, generate_task_testing, moderate_task)

        voiceai_thread.stop()

        return transcript
    
    def _converse(self, test_case_name: str, agents: Dict[str, FineVoicingAgent], generate_task_tested, generate_task_testing, moderate_task):
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]
        
        generate_tested_crew = Crew(
            agents=[agents['voice_ai_model_agent']],
            tasks=[generate_task_tested],
            process=Process.sequential
        )

        generate_testing_crew = Crew(
            agents=[agents['conversation_generator']],
            tasks=[generate_task_testing],
            process=Process.sequential
        )

        moderate_crew = Crew(
            agents=[agents['moderator']],
            tasks=[moderate_task],
        )

        transcript = []
        should_terminate = False
        index_turn = 1

        while not should_terminate and index_turn <= test_case['turns']:
            logger.info(f"--- Starting turn {index_turn} ---")

            result_tested = generate_tested_crew.kickoff({"chat_history": self._format_transcript(transcript)})
            transcript.append(result_tested.raw)

            result_testing = generate_testing_crew.kickoff({"chat_history": self._format_transcript(transcript)})
            transcript.append(result_testing.raw)

            logger.debug(f"--- Intermediary conversation transcript ---")
            [logger.debug(line) for line in transcript]
            logger.debug(f"--- End of intermediary conversation transcript ---")
            
            # Moderate message
            logger.info("Moderating conversation")
            decision = moderate_crew.kickoff({"chat_history": self._format_transcript(transcript)})
            logger.info(f"Moderation Decision: {decision}")
            
            # Check if the moderator wants to terminate the conversation
            decision_str = str(decision)
            should_terminate = "terminate" in decision_str.lower()
            logger.debug(f"Conversation terminated by moderator after {index_turn} turns") if should_terminate else logger.debug(f"Conversation continuing after {index_turn} turns")
            logger.info(f"--- Ending turn {index_turn} ---")
            index_turn += 1

        logger.info("Conversation terminated by moderator") if should_terminate else logger.info(f"Conversation completed after {index_turn} turns")
        return transcript
    
    def _format_transcript(self, transcript: List[str]) -> str:
        return "\n".join(f"- {line}" for line in transcript) if len(transcript) > 0 else EMPTY_HISTORY