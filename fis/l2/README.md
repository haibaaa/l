# Programming Assignment 1: Hashcat Password Cracking

## Step 1 — Set Up the `uv` Project

```bash
# Create a new uv project (run once)
uv init csd356-hashcat
cd csd356-hashcat

# Copy dictionary_gen.py into this folder, then run directly — no extra
# dependencies needed (hashlib, secrets, string are all stdlib).
```

---

## Step 2 — Generate the Password Dictionary & Hashes

```bash
uv run dictionary_gen.py
```

### What the script does (algorithm summary)

| Step | Action |
|------|--------|
| **1** | Define charset = `a-z` (26 lowercase letters) |
| **2** | Initialise an empty `set` for O(1) uniqueness tracking |
| **3** | Loop: pick 6 chars at random via `secrets.choice()` (cryptographically secure RNG) |
| **4** | Add to the set only if not seen before; repeat until **10,000 unique** passwords collected |
| **5** | Write all passwords to `dictionary.txt` (one per line) |
| **6** | For each password compute `SHA-1` with `hashlib.sha1().hexdigest()` |
| **7** | Write all 40-char hex digests to `hashes.txt` (one per line) |

### Expected terminal output

```
============================================================
  CSD356 — Password Dictionary & Hash Generator
============================================================

[1/3] Generating 10,000 unique 6-char passwords …
      Total unique passwords generated : 10,000

[2/3] Writing plaintext passwords to 'dictionary.txt' …
  ✔  Saved 10,000 entries → /path/to/dictionary.txt

[3/3] Computing SHA-1 hashes and writing to 'hashes.txt' …
  ✔  Saved 10,000 entries → /path/to/hashes.txt

============================================================
  Done!  Both files are ready for Hashcat.
============================================================
```

> **Screenshot checkpoint ①** — Take a screenshot of the above terminal output.  
> Make sure your **terminal title bar / username / hostname** is visible.

---

## Step 3 — Verify the Files

```bash
# Count lines in each file (both should print 10000)
wc -l dictionary.txt hashes.txt

# Preview first 5 passwords
head -5 dictionary.txt

# Preview first 5 hashes
head -5 hashes.txt
```

> **Screenshot checkpoint ②** — Capture the `wc -l` output and a sample of both files.

---

## Crack Hashes with Hashcat

### Command

```bash
hashcat -m 100 -a 0 hashes.txt dictionary.txt --outfile cracked.txt --outfile-format 2
```

### Flag breakdown

| Flag | Value | Meaning |
|------|-------|---------|
| `-m` | `100` | Hash mode = **SHA-1** |
| `-a` | `0` | Attack mode = **Straight / Dictionary** |
| `hashes.txt` | — | Input file containing one SHA-1 hash per line |
| `dictionary.txt` | — | Wordlist (our generated password dictionary) |
| `--outfile` | `cracked.txt` | Save cracked plaintext passwords here |
| `--outfile-format` | `2` | Output only the plaintext (no hash prefix) |

### Optional: show cracked results in the terminal

```bash
hashcat -m 100 -a 0 hashes.txt dictionary.txt --show
```

> **Screenshot checkpoint ③** — Capture the Hashcat progress/status screen and the  
> final **"Recovered"** line (e.g., `Recovered........: 9998/10000`).

---

## Step 5 — Success Rate Calculation

$$\text{Success Rate} = \frac{\text{Cracked Passwords}}{\text{Total Passwords}} \times 100\%$$

### Quick calculation in Python

```python
total   = 10_000
cracked = int(input("Enter number of cracked passwords from Hashcat: "))
rate    = (cracked / total) * 100
print(f"Success Rate: {cracked}/{total} = {rate:.2f}%")
```

> **Screenshot checkpoint ④** — Capture the success-rate terminal output.

---

## Submission Checklist

> ⚠️ **Per the assignment instructions**, every screenshot **must show your terminal or browser ID** (username, hostname, or student ID visible in the prompt or title bar). Screenshots without this will **not be evaluated**.

| # | Item | File / Command | Screenshot required |
|---|------|---------------|-------------------|
| ① | Algorithm explanation | `dictionary_gen.py` comments + this README | Terminal running the script |
| ② | Generated `dictionary.txt` with total count (≥ 1,000) | `wc -l dictionary.txt` | Yes — show count & sample entries |
| ③ | Generated `hashes.txt` | `wc -l hashes.txt` + `head hashes.txt` | Yes — show count & sample hashes |
| ④ | Hashcat command | `hashcat -m 100 -a 0 hashes.txt dictionary.txt` | Yes — show full command + status |
| ⑤ | Success rate | Python snippet above | Yes — show calculation output |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `hashcat: command not found` | Add hashcat to `$PATH` or use full path, e.g. `./hashcat` |
| `No hashes loaded` | Ensure `hashes.txt` has no blank lines; re-run the generator |
| GPU not detected | Add `--force` flag *(not recommended for real use, OK for coursework)* |
| Hashcat says "All hashes found in potfile" | Run `hashcat --potfile-disable …` to ignore the cache |

---

*CSD356 — Foundation of Information Security | Programming Assignment 1*
