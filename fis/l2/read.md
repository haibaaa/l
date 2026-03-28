# Documentation: SHA-1 Hash Recovery Analysis
This document details the recovery of SHA-1 hashes using a dictionary-based attack. The process involved programmatic dictionary generation, environment configuration via Nix, and high-performance cracking using Hashcat. The operation achieved a 100% success rate.

---

## 2. Methodology

### 2.1 Dictionary Generation
The search space was defined and populated using a custom Python script. This ensured that the dictionary was optimized for the target hash set.

* **Execution Command:** `uv run dictionary_gen.py`
* **Primary Output:** `dictionary.txt`

!(./gen.png)

### 2.2 Data Verification
Prior to the attack, a quantitative analysis of the input files was performed to verify the scale of the target hashes against the generated dictionary.

* **Files Analyzed:** `hashes.txt`, `dictionary.txt`

!(./ss.png)

### 2.3 Hashcat Configuration
Hashcat (v7.1.2).
* **Hash Type:** SHA-1 (Mode 100)
* **Attack Mode:** Straight Dictionary (Mode 0)
* **Command:**
```bash
hashcat -m 100 -a 0 hashes.txt dictionary.txt --outfile cracked.txt --outfile-format 2
```

!(./hashcat.png)
> [!note] since the output was too big please refer to ./hashcatout.txt

---
## 3. Results and Performance

The recovery operation was completed with the following metrics:

| Metric | Status / Value |
| :--- | :--- |
| **Target Algorithm** | SHA-1 |
| **Total Hashes** | [See hashes.txt] |
| **Recovery Rate** | 100% |
| **Output Format** | Hash:Pass |
| **Result File** | `cracked.txt` |

