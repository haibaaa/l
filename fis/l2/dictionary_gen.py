"""
Overview:
------------------
1. Define the character set: 26 lowercase ASCII letters (a-z).
2. Use a set for uniqueness checks during generation.
3. Repeatedly sample 6 characters (with replacement) from the charset
   using Python's secrets module (cryptographically secure RNG) until
   we accumulate exactly TARGET_COUNT unique passwords.
4. Write the passwords to dictionary.txt, one per line.
5. For each password, compute its SHA-1 digest (hex-encoded) using hashlib.
6. Write only the hex hashes to hashes.txt, one per line (hashcat format).
"""

import hashlib
import secrets
import string
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
CHARSET: str = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
PASSWORD_LENGTH: int = 6
TARGET_COUNT: int = 10_000
DICT_FILE: Path = Path("dictionary.txt")
HASH_FILE: Path = Path("hashes.txt")


# ── Step 1: Generate unique passwords ────────────────────────────────────────
def generate_passwords(
    charset: str,
    length: int,
    count: int,
) -> list[str]:
    """
    Return `count` unique random strings of `length` chars drawn from `charset`.

    Strategy:
      - Use a set to track seen passwords (O(1) lookup).
      - secrets.choice() gives an unbiased, cryptographically strong pick.
      - Join `length` individual characters to form each candidate.
    """
    seen: set[str] = set()
    passwords: list[str] = []

    while len(passwords) < count:
        candidate: str = "".join(secrets.choice(charset) for _ in range(length))
        if candidate not in seen:  # guarantee uniqueness
            seen.add(candidate)
            passwords.append(candidate)

    return passwords


# ── Step 2: Hash each password with SHA-1 ────────────────────────────────────
def sha1_hash(password: str) -> str:
    """
    Return the lowercase hex SHA-1 digest of `password`.

    hashlib.sha1() returns a HASH object; .hexdigest() gives the 40-char
    hex string that hashcat expects as input.
    """
    return hashlib.sha1(password.encode("utf-8")).hexdigest()


def compute_hashes(passwords: list[str]) -> list[str]:
    """Map every plaintext password to its SHA-1 hex digest."""
    return [sha1_hash(pw) for pw in passwords]


# ── Step 3: Persist to disk ───────────────────────────────────────────────────
def save_lines(path: Path, lines: list[str]) -> None:
    """Write each item in `lines` as a separate line in the file at `path`."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✔  Saved {len(lines):,} entries → {path.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  CSD356 — Password Dictionary & Hash Generator")
    print("=" * 60)

    # 1. Generate
    print(
        f"\n[1/3] Generating {TARGET_COUNT:,} unique {PASSWORD_LENGTH}-char passwords …"
    )
    passwords: list[str] = generate_passwords(CHARSET, PASSWORD_LENGTH, TARGET_COUNT)
    print(f"      Total unique passwords generated : {len(passwords):,}")

    # 2. Save dictionary
    print(f"\n[2/3] Writing plaintext passwords to '{DICT_FILE}' …")
    save_lines(DICT_FILE, passwords)

    # 3. Hash & save
    print(f"\n[3/3] Computing SHA-1 hashes and writing to '{HASH_FILE}' …")
    hashes: list[str] = compute_hashes(passwords)
    save_lines(HASH_FILE, hashes)

    print("\n" + "=" * 60)
    print("  Done!  Both files are ready for Hashcat.")
    print("=" * 60)


if __name__ == "__main__":
    main()
