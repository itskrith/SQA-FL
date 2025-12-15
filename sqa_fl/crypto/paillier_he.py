# sqa_fl/crypto/paillier_he.py
from __future__ import annotations

from typing import Tuple

from phe import paillier

# Scale float -> int to use with Paillier (which works on integers)
_SCALE = 10**6


def generate_keypair() -> Tuple[paillier.PaillierPublicKey, paillier.PaillierPrivateKey]:
    """Generate a Paillier public/private keypair."""
    pub, priv = paillier.generate_paillier_keypair()
    return pub, priv


def _encode_float(x: float) -> int:
    """Encode a float into an integer with fixed-point scaling."""
    return int(round(x * _SCALE))


def _decode_float(n: int) -> float:
    """Decode an integer back into float."""
    return n / _SCALE


def encrypt_quality(
    pubkey: paillier.PaillierPublicKey,
    q: float,
) -> paillier.EncryptedNumber:
    """Encrypt a scalar quality score q \\in [0, 1]."""
    return pubkey.encrypt(_encode_float(q))


def decrypt_sum_quality(
    privkey: paillier.PaillierPrivateKey,
    enc_sum,
) -> float:
    """
    Decrypt an encrypted sum of encoded qualities and return it as float.

    enc_sum is typically a Paillier EncryptedNumber representing
    sum_i (encoded q_i * something).
    """
    decoded_int = privkey.decrypt(enc_sum)
    return _decode_float(decoded_int)
