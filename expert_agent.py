# expert_agent.py

import re
import json
from google.api_core.exceptions import ResourceExhausted

class ExpertAgent:
    """
    The "Expert" Agent (CORE). 
    A Neuro-Symbolic reasoning engine designed for dependency constraint optimization.
    It separates deterministic signal extraction (Regex) from stochastic planning (LLM).
    """
    def __init__(self, llm_client):
        self.llm = llm_client
        self.llm_available = True

    def _clean_json_response(self, text: str) -> str:
        """Sanitizes LLM output to ensure valid JSON parsing."""
        cleaned = text.strip()
        # Remove markdown code blocks if present (e.g. ```json ... ```)
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        return cleaned.strip()

    def _extract_key_constraints(self, error_log: str) -> list:
        """
        Extracts high-signal constraint lines from verbose pip logs.
        Used to create a dense context for the LLM summary.
        """
        key_lines = []
        patterns = [
            r"^\s*([a-zA-Z0-9\-_]+.* requires .*)$",
            r"^\s*([a-zA-Z0-9\-_]+.* depends on .*)$",
            r"^\s*(The user requested .*)$",
            r"^\s*(Incompatible versions: .*)$",
            r"^\s*(Conflict: .*)$"
        ]
        
        for pat in patterns:
            for match in re.finditer(pat, error_log, re.MULTILINE):
                key_lines.append(match.group(1).strip())
        
        # Deduplicate and return top 10 to maintain context window efficiency
        return list(set(key_lines))[:10]

    def summarize_error(self, error_message: str) -> str:
        """Generates a concise, one-sentence summary of the root cause."""
        if not self.llm_available: return "(LLM summary unavailable)"
        
        key_constraints = self._extract_key_constraints(error_message)
        if key_constraints:
            context = "Key constraint lines:\n" + "\n".join(key_constraints)
        else:
            context = error_message[:2000]

        prompt = (
            "Summarize the root cause of the following Python dependency conflict "
            "in a single, concise sentence. Focus explicitly on the package names involved. "
            f"Context: {context}"
        )
        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip().replace('\n', ' ')
        except Exception:
            return "Failed to get summary from LLM."

    def diagnose_conflict_from_log(self, error_log: str) -> list[str]:
        """
        Extracts ALL conflicting package names using rigorous Regex matching.
        
        Methodology:
        We prioritize Regex over LLMs for this step to ensure 'High Recall'.
        In a Co-Resolution scenario (especially for the Greedy Heuristic), 
        we need to identify every package involved in the conflict graph 
        (including 'bystander' packages), not just the root cause.
        """
        # 1. Define a regex for standard python package specs (e.g. "package>=1.0", "pkg==2.0")
        #    Matches a valid name followed immediately by a version operator.
        package_pattern = re.compile(
            r"(?P<name>[a-zA-Z0-9\-_]+)(?:==|>=|<=|~=|!=|<|>)"
        )
        
        found_packages = set()
        
        # 2. Extract standard specifiers from the log
        for match in package_pattern.finditer(error_log):
            name = match.group('name').lower()
            # Filter out build tools/noise keywords
            if name not in ['python', 'pip', 'setuptools', 'wheel', 'setup', 'dependencies', 'versions', 'requirement']:
                found_packages.add(name)
        
        # 3. Extract natural language conflicts (e.g. "Conflict between package-a and package-b")
        #    This captures packages that might appear without a version number in the error text.
        conflict_pattern = re.compile(r"conflict.*(?:between|dependencies|among)\s+(`|')?([a-zA-Z0-9\-_,\s]+)(`|')?", re.IGNORECASE)
        for match in conflict_pattern.finditer(error_log):
            raw_text = match.group(2)
            # Split by comma or space
            tokens = re.split(r'[,\s]+', raw_text)
            for t in tokens:
                clean_t = t.strip("`'").lower()
                # Filter out stop words
                if clean_t and len(clean_t) > 2 and clean_t not in ['and', 'the', 'dependencies', 'versions', 'package']:
                     found_packages.add(clean_t)

        return list(found_packages)

    def propose_co_resolution(
        self, 
        target_package: str, 
        error_log: str, 
        available_updates: dict,
        current_versions: dict = None,
        history: list = None
    ) -> dict | None:
        """
        Iterative Co-Resolution Planner.
        Generates a multi-package update plan to resolve dependency deadlocks.
        
        Args:
            target_package: The package we initially tried to update.
            error_log: The stderr from the failed installation.
            available_updates: Dict of {pkg: latest_ver} (The "Ceiling").
            current_versions: Dict of {pkg: installed_ver} (The "Floor").
            history: List of previous (plan, outcome) tuples for iterative refinement.
        """
        if not self.llm_available: return None

        # Construct constraints strings
        floor_constraints = json.dumps(current_versions, indent=2) if current_versions else "{}"
        ceiling_constraints = json.dumps(available_updates, indent=2)

        # Build History Context (Reinforcement Learning signal)
        history_text = ""
        if history:
            history_text = "--- PREVIOUS FAILED ATTEMPTS (DO NOT REPEAT) ---\n"
            for i, (attempt_plan, failure_reason) in enumerate(history):
                history_text += f"Attempt {i+1} Plan: {attempt_plan}\nResult: FAILED. Reason: {failure_reason}\n\n"

        prompt = f"""
        You are CORE (Constraint Optimization & Resolution Expert).
        Your task is to solve a dependency deadlock by synthesizing a "Co-Resolution Plan".

        OBJECTIVE:
        Find a set of version configurations that allows '{target_package}' to be updated (if possible) while strictly satisfying all dependency graph constraints.
        
        OPTIMIZATION RULES:
        1. MONOTONICITY: Do not propose versions lower than the CURRENT INSTALLED VERSIONS unless absolutely necessary to solve the conflict.
        2. MAXIMIZATION: Prefer the HIGHEST possible version from AVAILABLE UPDATES.
        3. COMPLETENESS: The plan MUST include a version for '{target_package}'.
        4. PLAUSIBILITY: Only use versions listed in AVAILABLE UPDATES or CURRENT INSTALLED VERSIONS. Do not invent versions.

        CONTEXT:
        1. Target Package: {target_package}
        2. Current Installed Versions (The Floor): {floor_constraints}
        3. Available Updates (The Ceiling): {ceiling_constraints}
        
        THE CONFLICT LOG:
        {error_log}

        {history_text}

        YOUR TASK:
        Analyze the conflict. Return a JSON object with:
        - "plausible": boolean (true if a solution exists within constraints)
        - "proposed_plan": list of strings ["package==version", ...]
        """

        try:
            response = self.llm.generate_content(prompt)
            clean_text = self._clean_json_response(response.text)
            
            match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if not match:
                print(f"  -> LLM_WARNING: Invalid JSON structure.")
                return None
            
            plan = json.loads(match.group(0))
            
            if plan.get("plausible") and isinstance(plan.get("proposed_plan"), list):
                # STRICT VALIDATION: Anti-Hallucination Check
                valid_plan = []
                for requirement in plan.get("proposed_plan", []):
                    try:
                        pkg, ver = requirement.split('==')
                        
                        # Case A: It's a known "New" version (from Ceiling)
                        if pkg in available_updates and available_updates[pkg] == ver:
                            valid_plan.append(requirement)
                        
                        # Case B: It's a known "Current" version (from Floor)
                        # The Expert decided we must hold this package back to solve the conflict.
                        elif current_versions and pkg in current_versions and current_versions[pkg] == ver:
                            valid_plan.append(requirement)
                        
                        else:
                            print(f"  -> LLM_WARNING: Hallucinated version detected and filtered: {requirement}")
                            # We filter it out to prevent crashing pip. 
                            # If the plan relied on this hallucination, validation will likely fail later, which is fine.
                    except ValueError:
                        continue
                
                if not valid_plan:
                    return {"plausible": False, "proposed_plan": []}
                
                plan["proposed_plan"] = valid_plan
                return plan
                
            return None
        except (json.JSONDecodeError, AttributeError, Exception) as e:
            print(f"  -> LLM_ERROR: Co-Resolution generation failed: {e}")
            return None