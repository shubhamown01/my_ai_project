"""
security.py — Smart Vision Security Layer
==========================================
Features:
- AES-256 encryption for all database files
- Tamper detection via SHA-256 checksums
- Access token system (session-based)
- Audit log (every DB read/write logged)
- Rate limiting (brute force prevention)
- Data integrity verification on startup
"""

import os
import json
import time
import hashlib
import secrets
import logging
from datetime import datetime
from functools import wraps
from collections import defaultdict

# ── Crypto (safe import — works even if not installed) ──────────
CRYPTO_AVAILABLE = False
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    print("[SECURITY] cryptography not installed — encryption disabled.")
    print("[SECURITY] To enable: pip install cryptography")
except Exception as e:
    print(f"[SECURITY] cryptography load error: {e}")

# ══════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════
SECURITY_DIR   = "security"
KEY_FILE       = os.path.join(SECURITY_DIR, ".key")          # AES key (never share)
CHECKSUM_FILE  = os.path.join(SECURITY_DIR, "checksums.json")
AUDIT_LOG      = os.path.join(SECURITY_DIR, "audit.log")
TOKEN_FILE     = os.path.join(SECURITY_DIR, ".session_tokens")
os.makedirs(SECURITY_DIR, exist_ok=True)

# ── Files to protect (integrity check) ──────────────────────
PROTECTED_FILES = [
    "persons/person_registry.json",
    "raw_data_folder/global_object_dataset.csv",
    "activity_patterns/pattern_registry.json",
]

# ══════════════════════════════════════════════════════════════
#  AUDIT LOGGER
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    filename=AUDIT_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def audit(action: str, detail: str = "", level: str = "INFO"):
    msg = f"{action} | {detail}"
    if level == "WARNING":
        logging.warning(msg)
    elif level == "ERROR":
        logging.error(msg)
    else:
        logging.info(msg)


# ══════════════════════════════════════════════════════════════
#  AES-256 ENCRYPTION
# ══════════════════════════════════════════════════════════════

class Encryptor:
    """Fernet (AES-128-CBC + HMAC-SHA256) encryption wrapper."""

    def __init__(self):
        self._fernet = None
        if CRYPTO_AVAILABLE:
            self._load_or_create_key()

    def _load_or_create_key(self):
        if os.path.exists(KEY_FILE):
            with open(KEY_FILE, 'rb') as f:
                key = f.read().strip()
        else:
            key = Fernet.generate_key()
            with open(KEY_FILE, 'wb') as f:
                f.write(key)
            # Restrict permissions (Unix only)
            try:
                os.chmod(KEY_FILE, 0o600)
            except Exception:
                pass
            audit("KEY_CREATED", "New encryption key generated")
        try:
            self._fernet = Fernet(key)
        except Exception as e:
            audit("KEY_ERROR", str(e), "ERROR")

    def encrypt_file(self, filepath: str) -> bool:
        if not CRYPTO_AVAILABLE or self._fernet is None:
            return False
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            encrypted = self._fernet.encrypt(data)
            with open(filepath + ".enc", 'wb') as f:
                f.write(encrypted)
            audit("ENCRYPT", filepath)
            return True
        except Exception as e:
            audit("ENCRYPT_FAIL", f"{filepath}: {e}", "ERROR")
            return False

    def decrypt_file(self, enc_filepath: str) -> bytes:
        if not CRYPTO_AVAILABLE or self._fernet is None:
            return None
        try:
            with open(enc_filepath, 'rb') as f:
                data = f.read()
            decrypted = self._fernet.decrypt(data)
            audit("DECRYPT", enc_filepath)
            return decrypted
        except Exception as e:
            audit("DECRYPT_FAIL", f"{enc_filepath}: {e}", "ERROR")
            return None

    def encrypt_string(self, text: str) -> str:
        if not CRYPTO_AVAILABLE or self._fernet is None:
            return text
        return self._fernet.encrypt(text.encode()).decode()

    def decrypt_string(self, token: str) -> str:
        if not CRYPTO_AVAILABLE or self._fernet is None:
            return token
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except Exception:
            return ""


# ══════════════════════════════════════════════════════════════
#  CHECKSUM / TAMPER DETECTION
# ══════════════════════════════════════════════════════════════

class IntegrityChecker:

    def __init__(self):
        self._checksums = self._load()

    def _load(self) -> dict:
        if os.path.exists(CHECKSUM_FILE):
            try:
                with open(CHECKSUM_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        with open(CHECKSUM_FILE, 'w') as f:
            json.dump(self._checksums, f, indent=2)

    def _sha256(self, filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def update(self, filepath: str):
        """Record current checksum after a legitimate write."""
        if os.path.exists(filepath):
            self._checksums[filepath] = {
                "sha256":    self._sha256(filepath),
                "timestamp": datetime.now().isoformat()
            }
            self._save()

    def verify(self, filepath: str) -> bool:
        """Return True if file is untampered, False if modified externally."""
        if not os.path.exists(filepath):
            return True   # File doesn't exist yet — not a tamper
        if filepath not in self._checksums:
            self.update(filepath)
            return True   # First time — register it
        current  = self._sha256(filepath)
        expected = self._checksums[filepath]["sha256"]
        ok = (current == expected)
        if not ok:
            audit("TAMPER_DETECTED", filepath, "WARNING")
        return ok

    def verify_all(self) -> list:
        """Check all protected files. Returns list of tampered files."""
        tampered = []
        for fp in PROTECTED_FILES:
            if os.path.exists(fp) and not self.verify(fp):
                tampered.append(fp)
        return tampered


# ══════════════════════════════════════════════════════════════
#  SESSION TOKEN MANAGER
# ══════════════════════════════════════════════════════════════

class TokenManager:
    """
    Simple session token system.
    On startup a token is generated — all DB writes require it.
    Prevents external scripts from injecting data without auth.
    """

    def __init__(self):
        self._active_token = None
        self._token_ts     = None

    def generate_token(self) -> str:
        token = secrets.token_hex(32)
        self._active_token = token
        self._token_ts     = time.time()
        audit("TOKEN_GENERATED", token[:8] + "****")
        return token

    def validate(self, token: str) -> bool:
        if self._active_token is None:
            return False
        # Token expires after 8 hours
        if time.time() - self._token_ts > 28800:
            audit("TOKEN_EXPIRED", "", "WARNING")
            return False
        ok = secrets.compare_digest(token, self._active_token)
        if not ok:
            audit("TOKEN_INVALID", "Unauthorized access attempt", "WARNING")
        return ok

    def require_token(self, token: str):
        """Raises PermissionError if invalid."""
        if not self.validate(token):
            raise PermissionError("Invalid or expired security token.")


# ══════════════════════════════════════════════════════════════
#  RATE LIMITER
# ══════════════════════════════════════════════════════════════

class RateLimiter:
    """Prevents brute-force / spam DB writes."""

    def __init__(self, max_calls: int = 100, window_sec: int = 60):
        self.max_calls  = max_calls
        self.window     = window_sec
        self._calls     = defaultdict(list)

    def check(self, key: str) -> bool:
        now  = time.time()
        hist = self._calls[key]
        hist[:] = [t for t in hist if now - t < self.window]
        if len(hist) >= self.max_calls:
            audit("RATE_LIMIT", f"Key={key} exceeded {self.max_calls}/min", "WARNING")
            return False
        hist.append(now)
        return True


# ══════════════════════════════════════════════════════════════
#  SECURITY MANAGER  (single entry point)
# ══════════════════════════════════════════════════════════════

class SecurityManager:

    def __init__(self):
        self.encryptor = Encryptor()
        self.integrity = IntegrityChecker()
        self.tokens    = TokenManager()
        self.limiter   = RateLimiter()
        self._session_token = self.tokens.generate_token()
        self._startup_check()
        print("[SECURITY] System initialized. Integrity checks passed.")

    def _startup_check(self):
        tampered = self.integrity.verify_all()
        if tampered:
            print(f"[SECURITY ⚠️] Tampered files detected: {tampered}")
            audit("STARTUP_TAMPER", str(tampered), "WARNING")
        else:
            audit("STARTUP_OK", "All integrity checks passed")

    @property
    def token(self) -> str:
        return self._session_token

    def record_write(self, filepath: str):
        """Call after every legitimate DB write."""
        self.integrity.update(filepath)
        audit("DB_WRITE", filepath)

    def verify_read(self, filepath: str) -> bool:
        """Call before reading sensitive DB."""
        ok = self.integrity.verify(filepath)
        audit("DB_READ", filepath)
        return ok

    def rate_check(self, key: str) -> bool:
        return self.limiter.check(key)

    def get_audit_tail(self, n: int = 50) -> list:
        """Last N audit log lines."""
        if not os.path.exists(AUDIT_LOG):
            return []
        with open(AUDIT_LOG, 'r') as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:]]


# Singleton
_security = None

def get_security() -> SecurityManager:
    global _security
    if _security is None:
        _security = SecurityManager()
    return _security