
"""Converting the URL string to fixed-length integer sequence"""

from typing import List

# Maximum length of a URL sequence (longer ones will be cut,
# shorter ones will be padded with zeros)
MAX_LEN = 75 # each url represeented as exactly 75 chars

# I create a simple vocabulary for all printable ASCII characters
# ASCII codes 32–126 will cover characters like letters, digits, symbols, etc.
# Example: 'A' -> 34, 'a' -> 66, '/' -> 17, etc.
# Subtracting 31 shifts them to start from 1 instead of 32
VOCAB = {chr(i): i-31 for i in range(32, 127)}

# Special token IDs:
PAD = 0   # padding token (for short URLs)
UNK = len(VOCAB) + 1 # unknown character token (for chars not in VOCAB)

def url_to_ids(url : str , max_len : int = MAX_LEN) -> List[int]:
    """
    Convert a URL string into a fixed-length list of integers.

    Steps:
      1. Map each character to an integer ID using VOCAB.
         - If a character isn't in VOCAB, use UNK (unknown token).
      2. Truncate to `max_len` if the URL is longer.
      3. Pad with PAD (0) on the right if it's shorter.

    Returns:
        List[int]: List of integer IDs, length = max_len
    """

    # Convert each character to its numeric ID (or UNK if missing)
    ids = [VOCAB.get(c, UNK) for c in url[:max_len]]

    # Add padding tokens to reach max_len
    ids += [PAD] * (max_len - len(ids)) # pad right

    # Ensure the final list has exactly max_len elements
    return ids[:max_len] 

# sanity check

if __name__ == "__main__":
    # Test conversation
    print(url_to_ids("https://google.com"))