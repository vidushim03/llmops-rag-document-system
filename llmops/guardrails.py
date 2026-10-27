"""
LLMOps Guardrails Engine
Responsible for:
1. Input Prompt Injection & Jailbreak Detection
2. PII Redaction / Masking (Emails, API Keys, Phone Numbers, Credit Cards, SSN)
3. Output PII Leak Redaction
"""

import re
from typing import Tuple, Dict, Any

class LLMGuardrails:
    # Common prompt injection signatures
    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"disregard all previous",
        r"you are now an unrestricted",
        r"dan mode",
        r"system prompt leak",
        r"reveal your secret prompt",
        r"bypass rules",
        r"as an ai with no boundaries"
    ]

    # Sensitive PII patterns
    PII_PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "API_KEY": r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer)[\s:=]+['\"]?([a-zA-Z0-9_\-]{16,})['\"]?",
        "PHONE_NUMBER": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
    }

    def __init__(self, block_injections: bool = True, mask_pii: bool = True):
        self.block_injections = block_injections
        self.mask_pii = mask_pii

    def validate_input(self, user_query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates user query before passing to RAG retriever.
        Returns: (is_safe, sanitized_query, audit_details)
        """
        audit = {
            "prompt_injection_detected": False,
            "detected_injection_patterns": [],
            "pii_detected": False,
            "pii_types_found": []
        }

        # 1. Injection check
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_query, flags=re.IGNORECASE):
                audit["prompt_injection_detected"] = True
                audit["detected_injection_patterns"].append(pattern)

        if audit["prompt_injection_detected"] and self.block_injections:
            return False, "Violation: Potential prompt injection or adversarial jailbreak attempt detected.", audit

        # 2. PII Sanitization
        sanitized_query = user_query
        if self.mask_pii:
            for pii_name, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, sanitized_query, flags=re.IGNORECASE):
                    audit["pii_detected"] = True
                    audit["pii_types_found"].append(pii_name)
                    sanitized_query = re.sub(pattern, f"[REDACTED_{pii_name}]", sanitized_query, flags=re.IGNORECASE)

        return True, sanitized_query, audit

    def validate_output(self, generated_response: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Redact PII that the model reproduced before it is stored or shown.

        Returns ``(is_safe, sanitized_response, audit)``. ``is_safe`` describes
        the returned text (redacted text is safe to display); the audit records
        whether the raw output actually contained PII so callers can warn the
        user and count the incident.
        """
        audit = {
            "output_pii_detected": False,
            "output_pii_types_found": [],
            "output_pii_leaked": False,
        }

        sanitized_response = generated_response or ""
        for pii_name, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, sanitized_response, flags=re.IGNORECASE):
                audit["output_pii_detected"] = True
                audit["output_pii_types_found"].append(pii_name)
                sanitized_response = re.sub(
                    pattern,
                    f"[REDACTED_{pii_name}]",
                    sanitized_response,
                    flags=re.IGNORECASE,
                )

        audit["output_pii_leaked"] = audit["output_pii_detected"]
        return True, sanitized_response, audit
