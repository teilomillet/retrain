import re
import sqlparse
from typing import Dict, Any, List
from .reward import reward

@reward("sql_quality")
def sql_quality(prompt: str, completion: str, **kwargs) -> float:
    """
    Format-based reward for SQL quality assessment.
    
    Checks:
    - SQL syntax validity
    - Proper SQL structure (SELECT, FROM, etc.)
    - Query complexity and completeness
    - Relevance to the prompt
    """
    score = 0.0
    
    # Extract SQL from completion (look for SQL patterns)
    sql_pattern = r'(?i)(select\s+.*?(?:;|$))'
    sql_matches = re.findall(sql_pattern, completion, re.DOTALL | re.IGNORECASE)
    
    if not sql_matches:
        # No SQL found, check if it's a reasonable explanation instead
        if any(keyword in completion.lower() for keyword in ['select', 'from', 'where', 'join', 'group by']):
            return 0.1  # Small reward for SQL-related content
        return -0.5
    
    # Take the first SQL query found
    sql_query = sql_matches[0].strip()
    
    # 1. Syntax validation using sqlparse (basic check)
    try:
        parsed = sqlparse.parse(sql_query)
        if parsed and len(parsed) > 0:
            score += 0.3  # Basic syntax reward
        else:
            return -0.3
    except Exception:
        return -0.5
    
    # 2. Structure checks
    sql_lower = sql_query.lower()
    
    # Must have SELECT
    if 'select' in sql_lower:
        score += 0.2
    else:
        return -0.3
    
    # Must have FROM
    if 'from' in sql_lower:
        score += 0.2
    else:
        score -= 0.1
    
    # 3. Complexity and completeness rewards
    complexity_keywords = ['where', 'join', 'group by', 'having', 'order by', 'limit']
    complexity_score = sum(0.1 for keyword in complexity_keywords if keyword in sql_lower)
    score += min(complexity_score, 0.3)  # Cap at 0.3
    
    # 4. Length and detail check (avoid trivial queries)
    if len(sql_query.split()) >= 5:
        score += 0.1
    
    # 5. Check relevance to prompt (basic keyword matching)
    prompt_keywords = re.findall(r'\b\w+\b', prompt.lower())
    sql_keywords = re.findall(r'\b\w+\b', sql_query.lower())
    
    # Count keyword overlaps
    overlap = len(set(prompt_keywords) & set(sql_keywords))
    if overlap > 2:  # Some relevance
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

@reward("tool_usage")
def tool_usage(prompt: str, completion: str, **kwargs) -> float:
    """
    Reward for proper tool usage in SQL generation context.
    
    Checks:
    - Evidence of database exploration
    - Use of schema information
    - Structured approach to query building
    """
    score = 0.0
    
    # Check for evidence of tool usage patterns
    tool_indicators = [
        'describe', 'show tables', 'schema', 'information_schema',
        'explain', 'table', 'column', 'database'
    ]
    
    completion_lower = completion.lower()
    
    # Reward mentions of database exploration
    for indicator in tool_indicators:
        if indicator in completion_lower:
            score += 0.1
    
    # Reward structured thinking
    structure_indicators = [
        'first', 'then', 'next', 'finally', 'step',
        'need to', 'should', 'will', 'can'
    ]
    
    structure_score = sum(0.05 for indicator in structure_indicators if indicator in completion_lower)
    score += min(structure_score, 0.2)
    
    # Bonus for mentioning specific SQL elements that suggest exploration
    sql_exploration = ['table_name', 'column_name', 'data_type', 'primary_key', 'foreign_key']
    exploration_score = sum(0.1 for element in sql_exploration if element in completion_lower)
    score += min(exploration_score, 0.3)
    
    # Check if there's actual SQL - tool usage should lead to query
    if 'select' in completion_lower and 'from' in completion_lower:
        score += 0.2
    
    return min(score, 1.0)

@reward("final_answer_quality")
def final_answer_quality(prompt: str, completion: str, **kwargs) -> float:
    """
    Reward for overall answer quality and completeness.
    
    Checks:
    - Answer completeness
    - Clear SQL query provided
    - Explanation quality
    - Format adherence
    """
    score = 0.0
    
    # 1. Must contain SQL
    sql_pattern = r'(?i)(select\s+.*?(?:;|$))'
    has_sql = bool(re.search(sql_pattern, completion, re.DOTALL))
    
    if has_sql:
        score += 0.4
    else:
        # If no SQL, it might be an explanation or partial answer
        if any(word in completion.lower() for word in ['query', 'select', 'database', 'table']):
            score += 0.1
        else:
            return -0.5
    
    # 2. Length and detail assessment
    word_count = len(completion.split())
    if word_count >= 20:  # Substantial answer
        score += 0.2
    elif word_count >= 10:  # Moderate answer
        score += 0.1
    
    # 3. Explanation quality
    explanation_indicators = [
        'because', 'since', 'due to', 'in order to', 'this will',
        'the reason', 'explanation', 'approach', 'solution'
    ]
    
    explanation_score = sum(0.05 for indicator in explanation_indicators if indicator.lower() in completion.lower())
    score += min(explanation_score, 0.2)
    
    # 4. Professional formatting
    formatting_indicators = ['```', 'sql', '`', '\n', ':']
    formatting_score = sum(0.02 for indicator in formatting_indicators if indicator in completion)
    score += min(formatting_score, 0.1)
    
    # 5. Question answering directness
    question_words = ['how many', 'what', 'which', 'when', 'where', 'who', 'why']
    prompt_lower = prompt.lower()
    
    # Check if prompt contains a question
    has_question = any(qword in prompt_lower for qword in question_words) or '?' in prompt
    
    if has_question:
        # Look for answer patterns in completion
        answer_patterns = ['the answer is', 'result:', 'total:', 'count:', 'number:']
        if any(pattern in completion.lower() for pattern in answer_patterns):
            score += 0.1
    
    return min(score, 1.0) 