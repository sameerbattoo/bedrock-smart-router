#!/usr/bin/env python3
"""Generate code generation benchmark prompts.

Each prompt contains:
  - id: unique identifier
  - category: "code_generation"
  - difficulty: "simple" | "medium" | "complex"
  - system_prompt: instructions for the AI
  - user_prompt: the coding task
  - context: requirements, specs, or existing code
  - expected_answer: reference implementation
"""
import json
import os

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)


def save(filename, data):
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  {filename}: {len(data)} prompts")


prompts = [
    # ── Simple (1-17) ────────────────────────────────────────────
    {"id": "code_001", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that checks if a string is a palindrome, ignoring case and non-alphanumeric characters.",
     "context": "Function signature: def is_palindrome(s: str) -> bool",
     "expected_answer": "def is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]"},
    {"id": "code_002", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that finds the two numbers in a list that add up to a target sum.",
     "context": "Function signature: def two_sum(nums: list[int], target: int) -> tuple[int, int]\nReturn the indices of the two numbers.",
     "expected_answer": "def two_sum(nums: list[int], target: int) -> tuple[int, int]:\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return (seen[complement], i)\n        seen[num] = i\n    return (-1, -1)"},
    {"id": "code_003", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that flattens a nested list of arbitrary depth.",
     "context": "Function signature: def flatten(lst: list) -> list\nExample: flatten([1, [2, [3, 4]], 5]) -> [1, 2, 3, 4, 5]",
     "expected_answer": "def flatten(lst: list) -> list:\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"},
    {"id": "code_004", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that implements binary search on a sorted list.",
     "context": "Function signature: def binary_search(arr: list[int], target: int) -> int\nReturn the index of target, or -1 if not found.",
     "expected_answer": "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"},
    {"id": "code_005", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that groups a list of strings by their first character.",
     "context": "Function signature: def group_by_first_char(words: list[str]) -> dict[str, list[str]]\nExample: group_by_first_char(['apple', 'banana', 'avocado']) -> {'a': ['apple', 'avocado'], 'b': ['banana']}",
     "expected_answer": "def group_by_first_char(words: list[str]) -> dict[str, list[str]]:\n    groups = {}\n    for word in words:\n        if word:\n            key = word[0].lower()\n            groups.setdefault(key, []).append(word)\n    return groups"},
    {"id": "code_006", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a decorator that caches function results (memoization).",
     "context": "The decorator should work with any function that takes hashable arguments.",
     "expected_answer": "def memoize(func):\n    cache = {}\n    def wrapper(*args, **kwargs):\n        key = (args, tuple(sorted(kwargs.items())))\n        if key not in cache:\n            cache[key] = func(*args, **kwargs)\n        return cache[key]\n    return wrapper"},
    {"id": "code_007", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that validates an email address using regex.",
     "context": "Function signature: def is_valid_email(email: str) -> bool\nShould handle standard email formats like user@domain.com",
     "expected_answer": "import re\n\ndef is_valid_email(email: str) -> bool:\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))"},
    {"id": "code_008", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that converts a Roman numeral string to an integer.",
     "context": "Function signature: def roman_to_int(s: str) -> int\nHandle values I, V, X, L, C, D, M and subtractive notation (IV=4, IX=9, etc.)",
     "expected_answer": "def roman_to_int(s: str) -> int:\n    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    result = 0\n    for i in range(len(s)):\n        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:\n            result -= values[s[i]]\n        else:\n            result += values[s[i]]\n    return result"},
    {"id": "code_009", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that implements a stack using a list with push, pop, peek, and is_empty methods.",
     "context": "Implement as a class: class Stack",
     "expected_answer": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item):\n        self._items.append(item)\n\n    def pop(self):\n        if self.is_empty():\n            raise IndexError('pop from empty stack')\n        return self._items.pop()\n\n    def peek(self):\n        if self.is_empty():\n            raise IndexError('peek from empty stack')\n        return self._items[-1]\n\n    def is_empty(self) -> bool:\n        return len(self._items) == 0\n\n    def __len__(self) -> int:\n        return len(self._items)"},
    {"id": "code_010", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that merges two sorted lists into one sorted list.",
     "context": "Function signature: def merge_sorted(list1: list[int], list2: list[int]) -> list[int]\nDo not use built-in sort.",
     "expected_answer": "def merge_sorted(list1: list[int], list2: list[int]) -> list[int]:\n    result = []\n    i, j = 0, 0\n    while i < len(list1) and j < len(list2):\n        if list1[i] <= list2[j]:\n            result.append(list1[i])\n            i += 1\n        else:\n            result.append(list2[j])\n            j += 1\n    result.extend(list1[i:])\n    result.extend(list2[j:])\n    return result"},
    {"id": "code_011", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that finds the longest common prefix among a list of strings.",
     "context": "Function signature: def longest_common_prefix(strs: list[str]) -> str",
     "expected_answer": "def longest_common_prefix(strs: list[str]) -> str:\n    if not strs:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix"},
    {"id": "code_012", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that generates all permutations of a list.",
     "context": "Function signature: def permutations(lst: list) -> list[list]",
     "expected_answer": "def permutations(lst: list) -> list[list]:\n    if len(lst) <= 1:\n        return [lst]\n    result = []\n    for i, item in enumerate(lst):\n        rest = lst[:i] + lst[i+1:]\n        for perm in permutations(rest):\n            result.append([item] + perm)\n    return result"},
    {"id": "code_013", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that implements retry logic with exponential backoff.",
     "context": "Function signature: def retry(func, max_retries=3, base_delay=1.0)\nShould catch exceptions and retry with exponential backoff.",
     "expected_answer": "import time\n\ndef retry(func, max_retries=3, base_delay=1.0):\n    for attempt in range(max_retries + 1):\n        try:\n            return func()\n        except Exception as e:\n            if attempt == max_retries:\n                raise\n            delay = base_delay * (2 ** attempt)\n            time.sleep(delay)"},
    {"id": "code_014", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that converts a dictionary to a query string.",
     "context": "Function signature: def dict_to_query_string(params: dict) -> str\nExample: {'name': 'John', 'age': 30} -> 'name=John&age=30'\nHandle URL encoding.",
     "expected_answer": "from urllib.parse import quote\n\ndef dict_to_query_string(params: dict) -> str:\n    parts = []\n    for key, value in params.items():\n        parts.append(f'{quote(str(key))}={quote(str(value))}')\n    return '&'.join(parts)"},
    {"id": "code_015", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that calculates the nth Fibonacci number using dynamic programming.",
     "context": "Function signature: def fibonacci(n: int) -> int\nUse bottom-up DP, not recursion. Handle n=0 and n=1.",
     "expected_answer": "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    prev, curr = 0, 1\n    for _ in range(2, n + 1):\n        prev, curr = curr, prev + curr\n    return curr"},
    {"id": "code_016", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a context manager that measures execution time of a code block.",
     "context": "Usage: with Timer() as t: ... ; print(t.elapsed)",
     "expected_answer": "import time\n\nclass Timer:\n    def __init__(self):\n        self.elapsed = 0.0\n\n    def __enter__(self):\n        self._start = time.perf_counter()\n        return self\n\n    def __exit__(self, *args):\n        self.elapsed = time.perf_counter() - self._start"},
    {"id": "code_017", "category": "code_generation", "difficulty": "simple",
     "system_prompt": "You are an expert Python developer. Write clean, well-documented code. Return only the code, no explanation.",
     "user_prompt": "Write a function that chunks a list into sublists of a given size.",
     "context": "Function signature: def chunk(lst: list, size: int) -> list[list]\nExample: chunk([1,2,3,4,5], 2) -> [[1,2], [3,4], [5]]",
     "expected_answer": "def chunk(lst: list, size: int) -> list[list]:\n    return [lst[i:i + size] for i in range(0, len(lst), size)]"},

]


if __name__ == "__main__":
    save("code_generation.json", prompts)
    print(f"Generated {len(prompts)} code_generation prompts")
