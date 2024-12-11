import logging
import os
import json
from crewai import Agent, Task, Crew, Process, LLM
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
    
    def _setup_llms(self) -> Dict[str, LLM]:
        def _create_llm(model: str, provider: str, temperature: float = 0.7) -> LLM:
            model_name = f"{provider}/{model}"
            match provider:
                case "openrouter":
                    base_url = "https://openrouter.ai/api/v1"
                    api_key = os.getenv("OPENROUTER_API_KEY")
                case _:
                    raise ValueError(f"Invalid provider: {provider}")
            
            return LLM(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key
            )

        return {
            'llama_3_3_70b_instruct': _create_llm(
                provider="openrouter",
                model="meta-llama/llama-3.3-70b-instruct"
                
            ),
            'llama_3_2_3b_instruct': _create_llm(
                provider="openrouter",
                model="meta-llama/llama-3.2-3b-instruct"
            ),
            'qwen_2_5_72b_instruct': _create_llm(
                provider="openrouter",
                model="qwen/qwen-2.5-72b-instruct"
            ),
            'qwen_2_5_coder_32b_instruct': _create_llm(
                provider="openrouter",
                model="qwen/qwen-2.5-coder-32b-instruct"
            ),
            'qwen_2_5_7b_instruct': _create_llm(
                provider="openrouter",
                model="qwen/qwen-2.5-7b-instruct"
            )
        }

    def _setup_agents(self, test_case_name: str) -> Dict[str, FineVoicingAgent]:
        llms = self._setup_llms()
        logger = self.test_case_loggers[test_case_name]
        
        voice_ai_model_agent = FineVoicingAgent(
            role="Voice AI Model Agent",
            goal="Use the Voice AI Model to generate conversations.",
            verbose=True,
            memory=False,
            backstory="You specialize in using Voice AI Models to generate conversational messages.",
            logger=logger,
            **({ 'llm': llms['qwen_2_5_72b_instruct'] } if os.getenv('OPENROUTER_API_KEY') else {})
        )

        conversation_generator = FineVoicingAgent(
            role="Conversation Voice Simulation Agent",
            goal="Generate conversational messages in a voice conversation with an AI assistant.",
            verbose=True,
            memory=False,
            backstory="You specialize in simulating human-like voice interactions, and refining AI prompts.",
            logger=logger,
            **({ 'llm': llms['qwen_2_5_72b_instruct'] } if os.getenv('OPENROUTER_API_KEY') else {})
        )

        moderator = FineVoicingAgent(
            role="Conversation Moderator",
            goal="Evaluate each message in the conversation and decide if the conversation should continue.",
            verbose=True,
            memory=False,
            backstory="You are an expert in evaluating and moderating conversations.",
            logger=logger,
            **({ 'llm': llms['qwen_2_5_7b_instruct'] } if os.getenv('OPENROUTER_API_KEY') else {})
        )

        conversation_roles_agent = FineVoicingAgent(
            role="Conversation Roles Generator",
            goal="Generate the roles and instructions for a conversation based on the provided instructions.",
            verbose=True,
            memory=False,
            backstory="You specialize in generating the roles and instructions for a conversation based on the provided instructions.",
            logger=logger,
            **({ 'llm': llms['qwen_2_5_coder_32b_instruct'] } if os.getenv('OPENROUTER_API_KEY') else {})
        )

        return {
            'voice_ai_model_agent': voice_ai_model_agent,
            'conversation_generator': conversation_generator,
            'moderator': moderator,
            'conversation_roles_agent': conversation_roles_agent
        }

    def _generate_roles(self, agents: Dict[str, FineVoicingAgent], test_case_name: str) -> dict:
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]
        
        generate_roles_task = Task(
            description=(
                "Instructions: {instructions}"
                "Based on the instructions, identify the roles (at least two) involved in the conversation."
                "Based on the instructions, generate a specific set of instructions for each role. Make sure to include the target language in the instructions."
                "Based on the instructions, identify the role to be tested amongst the roles you've identified."
                "Format as a JSON object with root keys as 'tested_role' and 'testing_role'. Each role has keys: role_name, role_prompt. Output only valid JSON. No markdown or other formatting."
            ),
            expected_output="A JSON object with the roles and their descriptions.",
            agent=agents['conversation_roles_agent'],
        )

        generate_roles_crew = Crew(
            agents=[agents['conversation_roles_agent']],
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
        
        try:
            moderate_task = Task(
            description=(
                    "Decide if pursuing the conversation would be useful, based on the conversation history."
                    "Answer clearly: continue OR terminate."
                    "Provide reasoning for the decision."
                    "Always write in English"
                    "Chat history, each message is prefixed with a dash (-):"
                    "{chat_history}"
                ),
                expected_output="A decision to 'continue' or 'terminate', with reasoning.",
                agent=agents['moderator'],
            )

            roles = self._generate_roles(agents, test_case_name)

            tested_role = roles['tested_role']
            testing_role = roles['testing_role']

            if not tested_role or not testing_role:
                raise ValueError("No tested role or testing role identified")
            
            provider = Provider(test_case['voice_model']['provider'])
            voiceai_thread = VoiceAIModelThread(tested_role['role_prompt'], logger, provider=provider, first_speaker=ULTRAVOX_FIRST_SPEAKER_USER)

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

            tasks = {
                #'generate_task_tested': generate_task_tested,
                'generate_task_testing': generate_task_testing,
                'moderate_task': moderate_task
            }

            transcript = await self._converse(test_case_name, voiceai_thread, agents, tasks, roles)
        finally:
            voiceai_thread.stop()
        
        return transcript
    
    async def _converse(self, test_case_name: str, voiceai_thread: VoiceAIModelThread, agents: Dict[str, FineVoicingAgent], tasks: Dict[str, Task], roles: dict):
        logger = self.test_case_loggers[test_case_name]
        test_case = self.test_case_definitions[test_case_name]

        generate_testing_crew = Crew(
            agents=[agents['conversation_generator']],
            tasks=[tasks['generate_task_testing']],
            process=Process.sequential
        )

        moderate_crew = Crew(
            agents=[agents['moderator']],
            tasks=[tasks['moderate_task']],
        )

        transcript = []
        should_terminate = False
        index_turn = 1
        run_next_turn = True
        while run_next_turn:
            logger.info(f"--- Starting turn {index_turn} ---")

            last_message = transcript[-1] if len(transcript) > 0 else ""
            message = voiceai_thread.send_message(last_message)
            transcript.append(f"{roles['tested_role']['role_name']}: {message}")

            result_testing = await generate_testing_crew.kickoff_async({"chat_history": self._format_transcript(transcript)})
            transcript.append(result_testing.raw)

            logger.debug(f"--- Turn {index_turn} conversation transcript ---")
            [logger.debug(line) for line in transcript]
            logger.debug(f"--- End of turn {index_turn} conversation transcript ---")
            
            # Moderate message
            logger.info(f"--- Turn {index_turn}: Moderating conversation ---")
            decision = await moderate_crew.kickoff_async({"chat_history": self._format_transcript(transcript)})
            logger.info(f"Moderation Decision: {decision}")
            
            # Check if the moderator wants to terminate the conversation
            decision_str = str(decision)
            should_terminate = "terminate" in decision_str.lower()
            logger.debug(f"Conversation terminated by moderator after {index_turn} turns") if should_terminate else logger.debug(f"Conversation continuing after {index_turn} turns")
            logger.info(f"--- Ending turn {index_turn} ---")
            index_turn += 1
            run_next_turn = not should_terminate and index_turn <= test_case['turns']

        logger.info("Conversation terminated by moderator") if should_terminate else logger.info(f"Conversation completed after {index_turn-1} turns")
        return transcript
    
    def _format_transcript(self, transcript: List[str] | str) -> str:
        return "\n".join(f"- {line}" for line in transcript) if len(transcript) > 0 else EMPTY_HISTORY