#!/usr/bin/env python3
"""
PromptShield AI — Terminal Command-Line Interface (CLI).
Unified security analysis and risk assessment platform for prompts.

Evaluates prompts automatically across three integrated layers:
1. Prompt Security Model: Detects adversarial prompt injection, jailbreaks, and malice.
2. Application Risk Model: Predicts architectural attack surfaces implied by requested features,
   correlated with explicitly declared security controls and feature negations.
3. Sensitive Data Detector: Flags hardcoded API keys, secrets, and credentials (rule-based proof).
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyzers.prompt_analyzer import analyze_prompt
from analyzers.sensitive_data_detector import PATTERNS, redact_secret


# ANSI Colors with automatic TTY / NO_COLOR detection
class Colors:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    @property
    def RESET(self) -> str:
        return "\033[0m" if self.enabled else ""

    @property
    def BOLD(self) -> str:
        return "\033[1m" if self.enabled else ""

    @property
    def DIM(self) -> str:
        return "\033[2m" if self.enabled else ""

    @property
    def RED(self) -> str:
        return "\033[91m" if self.enabled else ""

    @property
    def GREEN(self) -> str:
        return "\033[92m" if self.enabled else ""

    @property
    def YELLOW(self) -> str:
        return "\033[93m" if self.enabled else ""

    @property
    def BLUE(self) -> str:
        return "\033[94m" if self.enabled else ""

    @property
    def CYAN(self) -> str:
        return "\033[96m" if self.enabled else ""

    @property
    def MAGENTA(self) -> str:
        return "\033[95m" if self.enabled else ""


def is_color_supported() -> bool:
    """Checks whether ANSI color output is appropriate."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Risk Metadata Mapping for the 15 Application Risk Labels
RISK_METADATA: Dict[str, Dict[str, str]] = {
    "sqli": {
        "title": "SQL Injection (SQLi)",
        "description": "Risk of SQL query manipulation via unparameterized database inputs.",
        "recommendation": "Use parameterized queries or an ORM (e.g., SQLAlchemy) for all DB operations; never concatenate raw user input into SQL."
    },
    "xss": {
        "title": "Cross-Site Scripting (XSS)",
        "description": "Risk of client-side script execution in user browsers via unescaped dynamic output.",
        "recommendation": "Contextually encode and sanitize dynamic output in HTML templates; implement a strict Content Security Policy (CSP)."
    },
    "ssrf": {
        "title": "Server-Side Request Forgery (SSRF)",
        "description": "Risk of unauthorized outbound server requests targeting internal infrastructure.",
        "recommendation": "Validate and whitelist destination URLs/IPs; block requests to internal subnets (127.0.0.1, 10.0.0.0/8, 169.254.169.254)."
    },
    "idor": {
        "title": "Insecure Direct Object References (IDOR)",
        "description": "Risk of accessing unauthorized records by manipulating object identifiers.",
        "recommendation": "Enforce object-level access control verifying the requesting user owns or is authorized for the target record ID."
    },
    "auth_weakness": {
        "title": "Authentication Weakness",
        "description": "Risk of credential brute-forcing, weak password hashing, or improper session creation.",
        "recommendation": "Use adaptive hashing (bcrypt, Argon2id), enforce strong password complexity policies, require Multi-Factor Authentication (MFA), and rate-limit login endpoints."
    },
    "authz_weakness": {
        "title": "Authorization Weakness",
        "description": "Risk of privilege escalation or missing role enforcement on sensitive endpoints.",
        "recommendation": "Implement centralized Role-Based Access Control (RBAC) or Attribute-Based Access Control (ABAC); default to deny-by-default on all endpoints."
    },
    "file_upload": {
        "title": "Insecure File Upload",
        "description": "Risk of arbitrary file upload, webshell execution, or storage exhaustion.",
        "recommendation": "Validate uploaded files by magic bytes and MIME type, enforce size limits, randomize stored file names, and store files outside the public web root in isolated object storage."
    },
    "path_traversal": {
        "title": "Path Traversal / Directory Traversal",
        "description": "Risk of reading arbitrary filesystem files via dot-dot-slash (../) manipulation.",
        "recommendation": "Canonicalize file paths and validate against a strict directory allowlist; avoid passing raw user input into filesystem APIs."
    },
    "command_injection": {
        "title": "Command Injection",
        "description": "Risk of executing arbitrary OS commands on the host server.",
        "recommendation": "Avoid shell execution (shell=True); use parameterized subprocess calls with structured argument lists and sanitize all input."
    },
    "csrf": {
        "title": "Cross-Site Request Forgery (CSRF)",
        "description": "Risk of unauthorized state-changing actions performed on behalf of authenticated users.",
        "recommendation": "Implement anti-CSRF tokens for all state-changing HTTP methods (POST, PUT, DELETE) and set SameSite=Lax or SameSite=Strict on session cookies."
    },
    "sensitive_data": {
        "title": "Sensitive Data Exposure",
        "description": "Risk of leaking PII, credentials, or internal system details via logs or unencrypted channels.",
        "recommendation": "Encrypt sensitive data at rest (AES-256) and in transit (TLS 1.3); redact secrets and PII from application logs and error traces."
    },
    "prompt_injection": {
        "title": "Prompt Injection / AI Evasion",
        "description": "Risk of untrusted external input overriding intended LLM system instructions.",
        "recommendation": "Isolate untrusted application inputs with clear delimiter boundaries before passing to LLMs; implement input/output guardrail checks."
    },
    "data_poisoning": {
        "title": "Data Poisoning",
        "description": "Risk of adversarial data corrupting ML training sets, fine-tuning pipelines, or RAG indexes.",
        "recommendation": "Verify provenance and cryptographic hashes of training datasets; audit data sources and sanitize inputs."
    },
    "model_extraction": {
        "title": "Model Extraction / Inversion",
        "description": "Risk of proprietary model reverse engineering via repetitive systematic querying.",
        "recommendation": "Enforce strict query rate limits, monitor for systematic extraction patterns, and avoid returning raw log probabilities."
    },
    "ai_tool_risk": {
        "title": "AI Tool / Agent Execution Risk",
        "description": "Risk of autonomous AI agents executing destructive tool actions without verification.",
        "recommendation": "Apply least privilege to agent tool capabilities; require explicit human-in-the-loop authorization for sensitive or irreversible operations."
    }
}


def mask_sensitive_text(prompt: str) -> str:
    """
    Safely redacts detected sensitive credentials in prompt text to prevent
    secret leakage in terminal displays or logs.
    """
    masked = prompt
    for pattern_def in PATTERNS:
        regex = pattern_def["regex"]
        matches = list(regex.finditer(masked))
        for m in reversed(matches):
            if pattern_def["type"] == "hardcoded_password_assignment":
                span = m.span(1)
            elif pattern_def["type"] == "db_connection_uri":
                span = m.span(0)
            else:
                span = m.span(0)
            val = masked[span[0]:span[1]]
            masked = masked[:span[0]] + redact_secret(val) + masked[span[1]:]
    return masked


def generate_recommendations(analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Synthesizes targeted, actionable security recommendations based on actual
    findings from all three analyzers, distinguishing declared controls from
    unmitigated risks and decoupling severity from raw model scores.
    """
    recommendations: List[Dict[str, str]] = []

    # 1. Deterministic Rule-Based Evidence (Sensitive Data Exposure)
    sensitive_data = analysis_result.get("sensitive_data") or {}
    if sensitive_data.get("detected"):
        findings = sensitive_data.get("findings", [])
        types_detected = list({f.get("type", "unknown") for f in findings})
        types_str = ", ".join(types_detected)
        recommendations.append({
            "priority": "CRITICAL",
            "category": "Credential Protection",
            "title": f"Revoke and remove exposed credentials ({types_str})",
            "guidance": (
                f"Identified {len(findings)} exposed secret pattern(s). Immediately revoke/rotate "
                "these keys. Remove all credentials from prompt text; inject secrets securely at runtime "
                "via environment variables or a secret manager (e.g., Vault, AWS Secrets Manager)."
            )
        })

    # 2. Prompt Security Analysis (Adversarial Malice & Injections)
    prompt_sec = analysis_result.get("prompt_security") or {}
    sec_status = prompt_sec.get("status")

    if sec_status == "success":
        if prompt_sec.get("is_malicious") is True:
            prob = prompt_sec.get("malicious_probability", 0.0)
            prio = "HIGH" if prob >= 0.80 else "MEDIUM"
            recommendations.append({
                "priority": prio,
                "category": "Prompt Security",
                "title": f"Mitigate adversarial prompt injection / jailbreak ({prob * 100:.1f}% confidence)",
                "guidance": (
                    "Prompt matches patterns characteristic of adversarial injection, jailbreaks, or instruction override. "
                    "Do not pass this prompt to downstream LLMs without strict input sandboxing, delimiter tags "
                    "(e.g., <user_input>), and an instruction-hierarchy safety layer."
                )
            })
        else:
            prob = prompt_sec.get("benign_probability", 0.0)
            recommendations.append({
                "priority": "INFORMATIONAL",
                "category": "Prompt Security",
                "title": f"Prompt appears non-adversarial ({prob * 100:.1f}% confidence)",
                "guidance": (
                    "Prompt demonstrates low adversarial pattern correlation. Continue applying standard defensive "
                    "delimiters when passing dynamic user inputs to AI components."
                )
            })
    elif sec_status in {"model_unavailable", "error"}:
        recommendations.append({
            "priority": "HIGH",
            "category": "Model Integrity",
            "title": "Prompt Security Model is offline or unverified",
            "guidance": (
                "Unable to evaluate adversarial prompt risk. DO NOT assume this prompt is benign. "
                "Train or restore the model artifact using: '.venv/bin/python ml/train_prompt_security_model.py'."
            )
        })

    # 3. Application Security: Declared Defenses vs. Unmitigated Surfaces
    app_risk = analysis_result.get("application_risks") or {}
    app_status = app_risk.get("status")

    if app_status == "success":
        detected_risks = app_risk.get("detected_risks") or []
        mitigated_surfaces = app_risk.get("mitigated_surfaces") or {}
        if "unmitigated_surfaces" in app_risk:
            unmitigated_surfaces = app_risk.get("unmitigated_surfaces") or []
        elif "mitigated_surfaces" in app_risk:
            unmitigated_surfaces = [r for r in detected_risks if r not in mitigated_surfaces]
        else:
            unmitigated_surfaces = detected_risks
        declared_controls = app_risk.get("declared_controls") or []

        # 3A. Surfaces with Declared Defenses (Residual Risk Review)
        if mitigated_surfaces:
            for risk_label, controls_list in mitigated_surfaces.items():
                meta = RISK_METADATA.get(risk_label, {"title": risk_label.replace("_", " ").title()})
                ctrl_str = ", ".join(controls_list)
                recommendations.append({
                    "priority": "DEFENSE-DECLARED",
                    "category": "Residual Risk Audit",
                    "title": f"{meta['title']} — Declared Controls Identified ({ctrl_str})",
                    "guidance": (
                        f"Prompt specifies defensive controls ({ctrl_str}). Residual risk review: "
                        "Audit algorithm parameters (e.g., memory/time costs for hashing, session expiration, "
                        "strict token revocation) during code implementation."
                    )
                })

        # 3B. Unmitigated Attack Surfaces
        if unmitigated_surfaces:
            for risk_label in unmitigated_surfaces:
                meta = RISK_METADATA.get(risk_label, {
                    "title": risk_label.replace("_", " ").title(),
                    "recommendation": "Review and harden application implementation for this risk area."
                })
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Application Hardening",
                    "title": meta["title"],
                    "guidance": meta["recommendation"]
                })

        # 3C. Informational when no active attack surfaces
        if not detected_risks:
            recommendations.append({
                "priority": "INFORMATIONAL",
                "category": "Application Architecture",
                "title": "No active application attack surfaces detected by baseline model",
                "guidance": (
                    "The application description does not trigger common risk categories or explicitly negates them. "
                    "Regularly audit your codebase against the OWASP Top 10 as additional features are introduced."
                )
            })

    elif app_status in {"model_unavailable", "error"}:
        recommendations.append({
            "priority": "HIGH",
            "category": "Model Integrity",
            "title": "Application Risk Model is offline or unverified",
            "guidance": (
                "Unable to predict application attack surfaces. DO NOT assume the requested application is risk-free. "
                "Train or restore the model artifact using: '.venv/bin/python ml/train_security_model.py'."
            )
        })

    return recommendations


def get_banner(colors: Colors) -> str:
    """Returns the PromptShield AI banner formatted for terminal display."""
    banner_text = r"""
================================================================================
  ____                            _   ____  _     _      _     _      _    ___ 
 |  _ \ _ __ ___  _ __ ___  _ __ | |_/ ___|| |__ (_) ___| | __| |    / \  |_ _|
 | |_) | '__/ _ \| '_ ` _ \| '_ \| __\___ \| '_ \| |/ _ \ |/ _` |   / _ \  | | 
 |  __/| | | (_) | | | | | | |_) | |_ ___) | | | | |  __/ | (_| |  / ___ \ | | 
 |_|   |_|  \___/|_| |_| |_| .__/ \__|____/|_| |_|_|\___|_|\__,_| /_/   \_\___|
                           |_|                                                  
      PromptShield AI — Unified Security & Risk Intelligence Platform
================================================================================
"""
    return f"{colors.CYAN}{banner_text}{colors.RESET}"


def format_security_report(
    prompt: str,
    analysis_result: Dict[str, Any],
    colors: Optional[Colors] = None
) -> str:
    """
    Formats the complete PromptShield AI unified security report.
    Integrates Prompt Security, Application Risks, Declared Controls, and Sensitive Data.
    """
    if colors is None:
        colors = Colors(enabled=False)

    lines: List[str] = []
    rule = "=" * 80
    subrule = "-" * 80

    lines.append(rule)
    lines.append(f"{colors.BOLD}{colors.CYAN}              PROMPTSHIELD AI — UNIFIED SECURITY REPORT{colors.RESET}")
    lines.append(rule)

    # Prompt Summary
    sanitized_prompt = mask_sensitive_text(prompt)
    preview = sanitized_prompt if len(sanitized_prompt) <= 300 else f"{sanitized_prompt[:297]}..."
    
    char_count = len(prompt)
    word_count = len(prompt.split())
    line_count = len(prompt.splitlines())

    lines.append(f"\n{colors.BOLD}[ PROMPT SUMMARY ]{colors.RESET}")
    lines.append(f"  Length: {char_count} chars | {word_count} words | {line_count} line(s)")
    lines.append(f"  Prompt Content (Secrets Redacted for Display):")
    for pline in preview.splitlines():
        lines.append(f"    {pline}")

    # Section 1: Prompt Security Analysis
    prompt_sec = analysis_result.get("prompt_security") or {}
    sec_status = prompt_sec.get("status", "unknown")

    lines.append(f"\n{subrule}")
    lines.append(f"{colors.BOLD}1. PROMPT SECURITY ANALYSIS (Adversarial Malice & Injection){colors.RESET}")
    lines.append(subrule)

    if sec_status == "success":
        is_mal = prompt_sec.get("is_malicious")
        mal_prob = prompt_sec.get("malicious_probability", 0.0)
        ben_prob = prompt_sec.get("benign_probability", 0.0)

        if is_mal:
            verdict_badge = f"{colors.BOLD}{colors.RED}[ ADVERSARIAL / MALICIOUS ]{colors.RESET}"
        else:
            verdict_badge = f"{colors.BOLD}{colors.GREEN}[ BENIGN ]{colors.RESET}"

        lines.append(f"  Analyzer Status:       {colors.GREEN}SUCCESS{colors.RESET}")
        lines.append(f"  Classification:        {verdict_badge}")
        lines.append(f"  Malicious Probability: {mal_prob * 100:.2f}%")
        lines.append(f"  Benign Probability:    {ben_prob * 100:.2f}%")
        lines.append(f"  Model Context:         {prompt_sec.get('disclaimer', '')}")
    elif sec_status == "model_unavailable":
        lines.append(f"  Analyzer Status:       {colors.RED}MODEL UNAVAILABLE{colors.RESET}")
        lines.append(f"  Classification:        {colors.YELLOW}[ UNVERIFIED — MODEL OFFLINE ]{colors.RESET}")
        lines.append(f"  {colors.RED}[!] CRITICAL WARNING: Model artifact missing!{colors.RESET}")
        lines.append("      DO NOT assume this prompt is benign.")
        lines.append("      To train/restore: python ml/train_prompt_security_model.py")
        if prompt_sec.get("error"):
            lines.append(f"      Details: {prompt_sec['error']}")
    else:
        lines.append(f"  Analyzer Status:       {colors.RED}ERROR{colors.RESET}")
        lines.append(f"  Classification:        {colors.YELLOW}[ UNVERIFIED — INFERENCE ERROR ]{colors.RESET}")
        lines.append(f"  {colors.RED}[!] WARNING: Inference failed; prompt safety cannot be confirmed.{colors.RESET}")
        if prompt_sec.get("error"):
            lines.append(f"      Error: {prompt_sec['error']}")

    # Section 2: Application Attack Surfaces & Declared Controls
    app_risk = analysis_result.get("application_risks") or {}
    app_status = app_risk.get("status", "unknown")

    lines.append(f"\n{subrule}")
    lines.append(f"{colors.BOLD}2. APPLICATION ATTACK SURFACES & SECURITY CONTROLS (Architectural Evaluation){colors.RESET}")
    lines.append(subrule)

    if app_status == "success":
        declared_controls = app_risk.get("declared_controls") or []
        declared_negations = app_risk.get("declared_negations") or []
        suppressed_risks = app_risk.get("suppressed_risks") or []
        detected_risks = app_risk.get("detected_risks") or []
        mitigated_surfaces = app_risk.get("mitigated_surfaces") or {}
        unmitigated_surfaces = app_risk.get("unmitigated_surfaces") or []

        # 2A. Explicitly Stated Security Controls
        lines.append(f"  {colors.BOLD}Explicitly Stated Security Controls:{colors.RESET}")
        if declared_controls:
            for ctrl in declared_controls:
                lines.append(f"    {colors.GREEN}[✓]{colors.RESET} {colors.BOLD}{ctrl['name']}{colors.RESET} (Matched: \"{ctrl['matched_text']}\")")
                lines.append(f"        {colors.DIM}{ctrl['description']}{colors.RESET}")
        else:
            lines.append(f"    {colors.DIM}None explicitly stated in prompt.{colors.RESET}")

        # 2B. Declared Feature Negations
        if declared_negations:
            lines.append(f"\n  {colors.BOLD}Declared Feature Negations (Excluded Attack Surfaces):{colors.RESET}")
            for neg in declared_negations:
                lines.append(f"    {colors.CYAN}[i]{colors.RESET} Excluded: {colors.BOLD}{neg['feature']}{colors.RESET} — \"{neg['matched_text']}\"")
            if suppressed_risks:
                lines.append(f"    {colors.DIM}Suppressed model risk predictions: {', '.join(suppressed_risks)}{colors.RESET}")

        # 2C. Model Predicted Potential Attack Surfaces
        lines.append(f"\n  {colors.BOLD}Predicted Potential Attack Surfaces ({len(detected_risks)} active):{colors.RESET}")
        if detected_risks:
            for risk in detected_risks:
                meta = RISK_METADATA.get(risk, {"title": risk, "description": "Application attack surface"})
                if risk in mitigated_surfaces:
                    ctrls_str = ", ".join(mitigated_surfaces[risk])
                    lines.append(f"    {colors.GREEN}[✓]{colors.RESET} {colors.BOLD}{risk:<18}{colors.RESET} — {meta['title']} {colors.GREEN}[Declared Controls: {ctrls_str}]{colors.RESET}")
                else:
                    lines.append(f"    {colors.YELLOW}[!]{colors.RESET} {colors.BOLD}{risk:<18}{colors.RESET} — {meta['title']} {colors.YELLOW}[No Declared Defenses]{colors.RESET}")
                lines.append(f"        {colors.DIM}{meta['description']}{colors.RESET}")
        else:
            lines.append(f"    {colors.GREEN}✓ No active application attack surfaces detected by baseline model.{colors.RESET}")

        lines.append("")
        lines.append(f"  {colors.CYAN}[i] IMPORTANT ARCHITECTURAL DISTINCTION:{colors.RESET}")
        lines.append("      These findings represent POTENTIAL ATTACK SURFACES in the requested")
        lines.append("      software design (e.g., authentication, database, uploads). They are")
        lines.append("      architectural areas to defend, NOT confirmed vulnerabilities or proof of malice.")
        lines.append(f"  Model Context:         {app_risk.get('disclaimer', '')}")
    elif app_status == "model_unavailable":
        lines.append(f"  Analyzer Status:       {colors.RED}MODEL UNAVAILABLE{colors.RESET}")
        lines.append(f"  Identified Surfaces:   {colors.YELLOW}[ UNVERIFIED — MODEL OFFLINE ]{colors.RESET}")
        lines.append(f"  {colors.RED}[!] CRITICAL WARNING: Model artifact missing!{colors.RESET}")
        lines.append("      DO NOT assume the requested application has no attack surfaces.")
        lines.append("      To train/restore: python ml/train_security_model.py")
        if app_risk.get("error"):
            lines.append(f"      Details: {app_risk['error']}")
    else:
        lines.append(f"  Analyzer Status:       {colors.RED}ERROR{colors.RESET}")
        lines.append(f"  Identified Surfaces:   {colors.YELLOW}[ UNVERIFIED — INFERENCE ERROR ]{colors.RESET}")
        lines.append(f"  {colors.RED}[!] WARNING: Inference failed; risk assessment unavailable.{colors.RESET}")
        if app_risk.get("error"):
            lines.append(f"      Error: {app_risk['error']}")

    # Section 3: Sensitive Data Detection
    sensitive_data = analysis_result.get("sensitive_data") or {}
    has_sensitive = sensitive_data.get("detected", False)
    findings = sensitive_data.get("findings", [])

    lines.append(f"\n{subrule}")
    lines.append(f"{colors.BOLD}3. SENSITIVE DATA & CREDENTIAL EXPOSURE DETECTION (Deterministic Rule Evidence){colors.RESET}")
    lines.append(subrule)

    if has_sensitive:
        badge = f"{colors.BOLD}{colors.RED}[ EXPOSURE DETECTED — EMPIRICAL EVIDENCE ]{colors.RESET}"
        lines.append(f"  Status:                {badge}")
        lines.append(f"  Findings Count:        {len(findings)}")
        lines.append("  Detected Sensitive Strings (Redacted):")
        for finding in findings:
            desc = finding.get("description", "Unknown Secret")
            ftype = finding.get("type", "generic")
            redacted = finding.get("redacted", "[REDACTED]")
            lines.append(f"    {colors.RED}[!] {desc} ({ftype}): {redacted}{colors.RESET}")
    else:
        badge = f"{colors.BOLD}{colors.GREEN}[ CLEAN — NO SECRETS FOUND ]{colors.RESET}"
        lines.append(f"  Status:                {badge}")
        lines.append(f"  {colors.GREEN}✓ No API keys, passwords, private keys, or tokens detected.{colors.RESET}")

    lines.append(f"  Detector Note:         {sensitive_data.get('disclaimer', '')}")

    # Section 4: Actionable Security Recommendations
    recommendations = generate_recommendations(analysis_result)

    lines.append(f"\n{subrule}")
    lines.append(f"{colors.BOLD}4. ACTIONABLE SECURITY RECOMMENDATIONS & RESIDUAL RISK REVIEW{colors.RESET}")
    lines.append(subrule)

    if recommendations:
        for rec in recommendations:
            prio = rec["priority"]
            if prio == "CRITICAL":
                prio_str = f"{colors.BOLD}{colors.RED}[CRITICAL]{colors.RESET}"
            elif prio == "HIGH":
                prio_str = f"{colors.BOLD}{colors.YELLOW}[HIGH]{colors.RESET}"
            elif prio in {"MEDIUM", "DEFENSE-DECLARED"}:
                prio_str = f"{colors.BOLD}{colors.BLUE}[DEFENSE-DECLARED]{colors.RESET}"
            else:
                prio_str = f"{colors.BOLD}{colors.CYAN}[INFO]{colors.RESET}"

            lines.append(f"  {prio_str} {colors.BOLD}{rec['title']}{colors.RESET}")
            lines.append(f"         Category: {rec['category']}")
            lines.append(f"         Action:   {rec['guidance']}")
            lines.append("")
    else:
        lines.append("  No specific security recommendations generated for this prompt.")

    # Section 5: Diagnostic Summary
    avail = analysis_result.get("model_availability") or {}
    sec_model_ok = avail.get("prompt_security_model", False)
    app_model_ok = avail.get("application_risk_model", False)
    overall_status = analysis_result.get("status", "unknown")

    lines.append(subrule)
    lines.append(f"{colors.BOLD}5. SYSTEM & MODEL DIAGNOSTICS{colors.RESET}")
    lines.append(subrule)
    lines.append(f"  Prompt Security Model:     {colors.GREEN}ONLINE{colors.RESET}" if sec_model_ok else f"  Prompt Security Model:     {colors.RED}OFFLINE{colors.RESET}")
    lines.append(f"  Application Risk Model:    {colors.GREEN}ONLINE{colors.RESET}" if app_model_ok else f"  Application Risk Model:    {colors.RED}OFFLINE{colors.RESET}")
    lines.append(f"  Sensitive Data Detector:   {colors.GREEN}ACTIVE (Rule-based heuristics){colors.RESET}")
    
    status_color = colors.GREEN if overall_status == "success" else (colors.YELLOW if overall_status == "partial_success" else colors.RED)
    lines.append(f"  Overall Orchestrator:      {status_color}{overall_status.upper()}{colors.RESET}")
    lines.append(rule)

    return "\n".join(lines)


def read_multiline_prompt(colors: Colors) -> Optional[str]:
    """
    Interactively reads a multiline prompt from the terminal until
    the user types 'END' on a line by itself.
    Returns None if the user issues an exit command or interrupts.
    """
    print(f"\n{colors.BOLD}Enter custom application prompt below.{colors.RESET}")
    print("• You may paste or enter multi-line text.")
    print(f"• Type {colors.BOLD}'END'{colors.RESET} on a new line when finished to run security analysis.")
    print(f"• Type {colors.BOLD}'exit'{colors.RESET} or {colors.BOLD}'quit'{colors.RESET} on the first line to quit.")
    print("-" * 80)

    lines: List[str] = []
    while True:
        try:
            line = input(f"{colors.CYAN}>{colors.RESET} ")
        except (KeyboardInterrupt, EOFError):
            return None

        # Check for immediate exit
        if not lines and line.strip().lower() in {"exit", "quit", "q"}:
            return None

        # Check for END sentinel (case-insensitive)
        if line.strip().upper() == "END":
            break

        lines.append(line)

    prompt_text = "\n".join(lines).strip()
    return prompt_text


def show_prompt_improvement_notice(colors: Colors) -> None:
    """
    Displays the planned feature notice for automated prompt improvement.
    """
    print("\n" + "=" * 80)
    print(f"{colors.BOLD}{colors.CYAN}       PROMPT IMPROVEMENT & OPTIMIZATION (PLANNED FEATURE){colors.RESET}")
    print("=" * 80)
    print(f"  {colors.YELLOW}[!] Feature Status: Scheduled for future development phase.{colors.RESET}")
    print("\n  Currently Implemented Capabilities in PromptShield AI:")
    print(f"    {colors.GREEN}✓{colors.RESET} Calibrated Prompt Security Model (Adversarial / Malice Classification)")
    print(f"    {colors.GREEN}✓{colors.RESET} Application Risk Assessment (15 Multi-label Architectural Risks)")
    print(f"    {colors.GREEN}✓{colors.RESET} Security Controls & Constraints Extractor (Argon2id, CSP, RBAC, etc.)")
    print(f"    {colors.GREEN}✓{colors.RESET} Sensitive Data Detection (Credentials, API Keys, Private Keys)")
    print(f"    {colors.GREEN}✓{colors.RESET} Decoupled Severity & Actionable Remediation Engine")
    print(f"    {colors.GREEN}✓{colors.RESET} Unified FastAPI REST Endpoints & Terminal CLI")
    print("\n  Planned Future Modules:")
    print(f"    {colors.DIM}• Automated Prompt Rewriting & Hardening")
    print("    • Token Usage & Cost Estimation")
    print(f"    • Multi-Model (GPT / Claude / Gemini) Comparative Scoring{colors.RESET}")
    print("=" * 80)


def handle_next_action_menu(colors: Colors) -> str:
    """
    Presents the post-report action menu and returns the selected action.
    Returns: 'improve', 'another', or 'exit'.
    """
    while True:
        print("\n" + "=" * 80)
        print(f"{colors.BOLD}NEXT ACTION MENU{colors.RESET}")
        print("=" * 80)
        print("  [1] Improve this prompt")
        print("  [2] Analyze another prompt")
        print("  [3] Exit PromptShield AI")
        print("=" * 80)

        try:
            choice = input(f"{colors.CYAN}Select an option (1-3) or type 'exit':{colors.RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return "exit"

        if choice in {"1", "improve", "improve this prompt"}:
            show_prompt_improvement_notice(colors)
            try:
                sub_choice = input(f"\n{colors.CYAN}Press [Enter] to return to menu, or 'exit' to quit:{colors.RESET} ").strip().lower()
                if sub_choice in {"exit", "quit", "q"}:
                    return "exit"
            except (KeyboardInterrupt, EOFError):
                return "exit"
            continue
        elif choice in {"2", "another", "analyze another prompt", "a"}:
            return "another"
        elif choice in {"3", "exit", "quit", "q"}:
            return "exit"
        else:
            print(f"{colors.RED}[!] Invalid choice. Please enter 1, 2, or 3.{colors.RESET}")


def run_interactive_cli(colors: Colors) -> None:
    """
    Runs the main interactive loop for PromptShield AI terminal workflow.
    """
    print(get_banner(colors))
    print("Welcome to PromptShield AI — Unified Prompt Security Analysis Platform.")

    while True:
        prompt = read_multiline_prompt(colors)

        if prompt is None:
            print(f"\n{colors.CYAN}Exiting PromptShield AI. Goodbye!{colors.RESET}\n")
            sys.exit(0)

        if not prompt:
            print(f"\n{colors.YELLOW}[!] Warning: Prompt cannot be empty or contain only whitespace.{colors.RESET}")
            print("    Please enter a valid application prompt to analyze.\n")
            continue

        print(f"\n{colors.CYAN}Running unified security analysis...{colors.RESET}")
        try:
            analysis_result = analyze_prompt(prompt, analysis_mode="both")
        except Exception as e:
            print(f"\n{colors.RED}[ERROR] Unexpected error during analysis: {e}{colors.RESET}")
            analysis_result = {
                "prompt": prompt,
                "analysis_mode": "both",
                "status": "error",
                "error": str(e),
                "model_availability": {"prompt_security_model": False, "application_risk_model": False},
                "prompt_security": {"status": "error", "error": str(e)},
                "application_risks": {"status": "error", "error": str(e)},
                "sensitive_data": {"detected": False, "findings_count": 0, "findings": []}
            }

        # Print report
        report_text = format_security_report(prompt, analysis_result, colors=colors)
        print(report_text)

        # Next action menu
        action = handle_next_action_menu(colors)
        if action == "exit":
            print(f"\n{colors.CYAN}Thank you for using PromptShield AI. Goodbye!{colors.RESET}\n")
            sys.exit(0)
        elif action == "another":
            print("\n" + "=" * 80)
            continue


def main():
    parser = argparse.ArgumentParser(
        description="PromptShield AI — Unified prompt security and application risk terminal interface."
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Analyze a single prompt directly in non-interactive mode."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to a text file containing the prompt to analyze."
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output."
    )

    args = parser.parse_args()
    colors = Colors(enabled=(not args.no_color and is_color_supported()))

    # Direct file input
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"{colors.RED}Error: File not found: {file_path}{colors.RESET}", file=sys.stderr)
            sys.exit(1)
        prompt = file_path.read_text(encoding="utf-8").strip()
        if not prompt:
            print(f"{colors.RED}Error: File is empty: {file_path}{colors.RESET}", file=sys.stderr)
            sys.exit(1)

        print(get_banner(colors))
        print(f"Analyzing prompt from file: {file_path}...\n")
        result = analyze_prompt(prompt, analysis_mode="both")
        print(format_security_report(prompt, result, colors=colors))
        sys.exit(0)

    # Direct string input
    if args.prompt is not None:
        prompt = args.prompt.strip()
        if not prompt:
            print(f"{colors.RED}Error: Provided prompt is empty or contains only whitespace.{colors.RESET}", file=sys.stderr)
            sys.exit(1)

        print(get_banner(colors))
        print("Analyzing provided prompt...\n")
        result = analyze_prompt(prompt, analysis_mode="both")
        print(format_security_report(prompt, result, colors=colors))
        sys.exit(0)

    # Interactive mode (default when started with python cli.py)
    run_interactive_cli(colors)


if __name__ == "__main__":
    main()
