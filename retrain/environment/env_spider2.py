import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal, TYPE_CHECKING

from loguru import logger
import torch

from .env_fastmcp import FastMCPEnv
from .types import EnvironmentObservation, LLMAction
from retrain.reward.types import RawRolloutData
from retrain.utils.parser.smol_agent_parser import ParsedSmolLLMOutput

if TYPE_CHECKING:
    ModelObject = Any
    TokenizerObject = Any
    SamplingParams = Dict[str, Any]

class Spider2Env(FastMCPEnv):
    """
    Environment for training on the Spider2 benchmark using MCP Alchemy for database interaction.
    This environment inherits from FastMCPEnv to leverage its tool discovery and interaction
    loop and specializes it for the text-to-SQL task.
    """

    def __init__(
        self,
        spider2_data_path: str = "data/spider/dev.jsonl",
        spider2_type: Literal["lite", "snow", "dbt"] = "lite",
        mcp_server_url: Optional[str] = None,
        mcp_server_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 20,
        enable_external_knowledge: bool = True,
        target_db_id: Optional[str] = None,
    ):
        super().__init__(
            server_url=mcp_server_url,
            mcp_server_config=mcp_server_config,
            max_steps=max_steps,
            initial_prompt_template="This will be overridden in reset."
        )

        self.spider2_data_path = Path(spider2_data_path)
        self.spider2_type = spider2_type
        self.enable_external_knowledge = enable_external_knowledge
        self.target_db_id = target_db_id

        self.questions: List[Dict[str, Any]] = []
        self.current_question_index = -1
        self.current_question: Optional[Dict[str, Any]] = None
        self.current_db: Optional[str] = None

    async def setup(self):
        """
        Initializes the environment by loading Spider2 data and then setting up
        the connection to the FastMCP server to discover database tools.
        """
        await self._load_spider2_data()
        await super().setup()
        logger.info(
            f"[Spider2Env] Setup complete. Loaded {len(self.questions)} questions. "
            f"Active tools from server: {list(self.active_tools.keys())}"
        )

    async def _load_spider2_data(self):
        """Load Spider2 questions from a JSONL file."""
        if not self.spider2_data_path.exists():
            raise FileNotFoundError(f"Spider2 data file not found at {self.spider2_data_path}")

        logger.info(f"Loading Spider2 data from {self.spider2_data_path}...")
        with open(self.spider2_data_path, 'r') as f:
            for line in f:
                if line.strip():
                    question = json.loads(line)
                    # If a target_db_id is specified, only load questions for that DB
                    if self.target_db_id:
                        if question.get('db_id') == self.target_db_id:
                            self.questions.append(question)
                    else:
                        self.questions.append(question)

        if not self.questions:
            if self.target_db_id:
                raise ValueError(f"No questions found for the target database '{self.target_db_id}' in {self.spider2_data_path}. Please check the db_id and data file.")
            else:
                raise ValueError(f"No questions were loaded from {self.spider2_data_path}. The file might be empty.")

        logger.info(f"Loaded {len(self.questions)} questions.")

    def _create_initial_prompt(self) -> str:
        """Create the initial prompt for the current question."""
        if not self.current_question or not self.current_db:
            return "No question loaded."

        prompt_parts = [
            f"You are a SQL expert. You need to answer the following question about the '{self.current_db}' database:",
            "",
            f"Question: {self.current_question['question']}",
            ""
        ]

        if self.enable_external_knowledge and self.current_question.get('external_knowledge'):
            prompt_parts.extend([
                f"External Knowledge: {self.current_question['external_knowledge']}",
                ""
            ])

        prompt_parts.extend([
            "IMPORTANT FORMAT REQUIREMENTS:",
            "- Start your response with <answer> for your final SQL query",
            "- Use <tool>{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool> to call database tools",
            "- Use <think>your reasoning</think> for intermediate thoughts",
            "- Do NOT put any other text before these tags",
            ""
        ])

        if self.active_tools:
            tool_descs = self._get_formatted_tool_descriptions()
            prompt_parts.extend([
                "You have access to the following database tools:",
                tool_descs,
                "",
                "Process:",
                "1. First, explore the database schema using available tools",
                "2. Then, write a SQL query to answer the question",
                "3. Provide your final answer as: <answer>SELECT ...</answer>"
            ])
        else:
            prompt_parts.extend([
                "Generate a SQL query that correctly answers the question.",
                "Provide your answer as: <answer>SELECT ...</answer>"
            ])

        return "\n".join(prompt_parts)

    async def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[EnvironmentObservation, Dict[str, Any]]:
        """
        Resets the environment with a new question from the Spider2 dataset.
        """
        if seed is not None:
            random.seed(seed)

        if options and 'question_index' in options:
            self.current_question_index = options['question_index']
        else:
            if not self.questions:
                raise RuntimeError("No questions loaded. Make sure setup() is called before reset().")
            self.current_question_index = random.randint(0, len(self.questions) - 1)

        self.current_question = self.questions[self.current_question_index]
        self.current_db = self.current_question.get('db_id') or self.current_question.get('db')
        
        if not self.current_db:
             logger.error(f"Could not find 'db_id' or 'db' key for question index {self.current_question_index}.")
             self.current_db = "unknown_db"


        initial_prompt = self._create_initial_prompt()

        self.current_turn = 0
        self._conversation_history = [{"role": "user", "content": initial_prompt}]

        logger.info(f"[Spider2Env] Reset. Question {self.current_question.get('instance_id', self.current_question_index)} for DB '{self.current_db}'.")

        initial_obs = EnvironmentObservation(
            observation_type="initial",
            content=None,
            tool_result=None,
            current_conversation=list(self._conversation_history),
            available_tools=list(self.tool_schemas_for_prompt),
            requires_llm_action=True
        )
        
        info = {
            'question_id': self.current_question.get('instance_id', self.current_question_index),
            'database_id': self.current_db,
            'question': self.current_question['question'],
            'ground_truth_query': self.current_question.get('query'),
            'external_knowledge': self.current_question.get('external_knowledge')
        }
        
        return initial_obs, info

    async def step(self, action: LLMAction) -> Tuple[EnvironmentObservation, float, bool, bool, Dict[str, Any]]:
        """
        Executes a step in the environment. It uses the parent FastMCPEnv's step method
        and then computes a custom reward for the text-to-SQL task.
        """
        obs, _, terminated, truncated, info = await super().step(action)
        
        reward = self._calculate_reward(action, info)

        if not terminated: 
            parsed_output = self.parser.parse(action["raw_llm_output"])
            if self._check_termination(parsed_output):
                terminated = True
                obs["requires_llm_action"] = False 
                info["termination_reason"] = "final_answer_detected"

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: LLMAction, step_info: Dict[str, Any]) -> float:
        """
        Calculates a simple intrinsic reward.
        """
        if step_info.get("step_error"):
            return -1.0 

        if step_info.get("tool_result_status") == "error":
            return -0.5

        if step_info.get("tool_result_status") == "success":
            return 0.1

        parsed_output = self.parser.parse(action["raw_llm_output"])
        if parsed_output.final_answer:
            if "select" in parsed_output.final_answer.lower():
                return 1.0
            else:
                return -0.2
        
        return 0.0

    def _check_termination(self, parsed_action: ParsedSmolLLMOutput) -> bool:
        """
        Checks if the episode should terminate.
        """
        return parsed_action.final_answer is not None

    def close(self) -> None:
        """Close the environment. Parent's close is sufficient."""
        super().close() 