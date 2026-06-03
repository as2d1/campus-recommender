"""Compatibility wrapper for the normalized mock data generator."""

from __future__ import annotations

from mock_data import generate_all, parse_args


if __name__ == "__main__":
    generate_all(parse_args())
