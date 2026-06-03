import hashlib


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


for sample in ["hello", "hello!", "HELLO"]:
    print(f"{sample:8} -> {sha256(sample)[:16]}...")