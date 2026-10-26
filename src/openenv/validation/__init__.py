"""RFC 008 environment validation: contracts and the local `openenv validate` pipeline.

Ships signature detection, parsers, one normalized manifest, a grader registry,
a report schema, a versioned severity policy, and the local runner. Graders read
the manifest, never the signature.
"""

from .graders import ENTRY_POINT_GROUP, Grader, GraderRegistry, Subject
from .manifest import (
    CapabilitiesSpec,
    JudgePin,
    ManifestError,
    NetworkPolicy,
    NormalizedManifest,
    OracleDeclaration,
    ResourceDeclaration,
    RewardDeclaration,
    TaskDistributionPin,
    TypeSpec,
    VerifierBinding,
)
from .parsers import Parser, ParserRegistry
from .policy import (
    apply_policy,
    DeclarationBounds,
    load_policy,
    PolicyEntry,
    PolicyError,
    SeverityPolicy,
)
from .providers import ExecResult, RunningSubject, ValidationProvider
from .report import CheckResult, ValidationReport, write_report
from .runner import run_validation
from .signature import (
    detect_signature,
    SignatureError,
    UNSUPPORTED_CATEGORIES,
    UnsupportedPackageError,
    WELL_KNOWN_FILES,
)
from .types import (
    CheckStatus,
    Lane,
    Level,
    ProviderCapability,
    Severity,
    SignatureKind,
    Verdict,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "UNSUPPORTED_CATEGORIES",
    "WELL_KNOWN_FILES",
    "CapabilitiesSpec",
    "CheckResult",
    "CheckStatus",
    "DeclarationBounds",
    "ExecResult",
    "Grader",
    "GraderRegistry",
    "JudgePin",
    "Lane",
    "Level",
    "ManifestError",
    "NetworkPolicy",
    "NormalizedManifest",
    "OracleDeclaration",
    "Parser",
    "ParserRegistry",
    "PolicyEntry",
    "PolicyError",
    "ProviderCapability",
    "ResourceDeclaration",
    "RewardDeclaration",
    "RunningSubject",
    "Severity",
    "SeverityPolicy",
    "SignatureError",
    "SignatureKind",
    "Subject",
    "TaskDistributionPin",
    "TypeSpec",
    "UnsupportedPackageError",
    "ValidationProvider",
    "ValidationReport",
    "Verdict",
    "VerifierBinding",
    "apply_policy",
    "detect_signature",
    "load_policy",
    "run_validation",
    "write_report",
]
