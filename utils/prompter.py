# utils/prompter.py
class Prompter(object):
    """Prompt template helper."""

    def __init__(self, template_name: str = "alpaca", template_path: str = None):
        self.template_name = template_name
        self.template = self._load_template(template_name, template_path)

    def _load_template(self, template_name, template_path):
        """Load a prompt template by name."""
        if template_name == "kg_completion":
            return {
                "prompt_listwise": "Given a head entity and a relation, select the correct tail entity\nfrom the candidate list.\n\nHead Entity:\n{head_entity}\n\nRelation:\n{relation}\n\nCandidates:\n{candidates}\n\nAnswer with the index only:"
            }
        else:
            return {}

    def generate_prompt(
            self,
            head_entity: str = None,
            relation: str = None,
            candidate_entities: list = None
    ) -> str:
        """Generate a listwise prompt for Top-1 selection.
        
        Args:
            head_entity: head entity name
            relation: relation name
            candidate_entities: candidate entity names, e.g. ["a", "b", ...]
        
        Returns:
            Prompt string.
        """
        if self.template_name != "kg_completion":
            raise ValueError(f"Listwise format only supports 'kg_completion', got '{self.template_name}'")
        
        if head_entity is None or relation is None:
            raise ValueError("head_entity and relation are required for Listwise format")
        
        if candidate_entities is None or len(candidate_entities) == 0:
            raise ValueError("candidate_entities is required for Listwise format")
        
        candidates_str = ""
        for i, cand_name in enumerate(candidate_entities, start=1):
            candidates_str += f"({i}) {cand_name}\n"
        candidates_str = candidates_str.rstrip("\n")
        
        return self.template["prompt_listwise"].format(
            head_entity=head_entity,
            relation=relation,
            candidates=candidates_str
        )

    def get_response(self, output: str) -> str:
        """Extract the chosen index from model output (listwise format)."""
        import re
        match = re.search(r'\d+', output)
        if match:
            return match.group()
        return output.strip()