import secrets

# Generate a 24-byte key for 3DES
secure_key = secrets.token_bytes(24)
print(secure_key)
