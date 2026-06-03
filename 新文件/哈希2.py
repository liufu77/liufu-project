import hashlib


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


left = "刘付"
right = "冬琦"
h1 = sha256(left)
h2 = sha256(right)
diff = sum(a != b for a, b in zip(h1, h2))

print(left, "->", h1)
print(right, "->", h2)
print(f"64 位十六进制中有 {diff} 位不同")