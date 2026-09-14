"""Offline consolidation & SFT training for Ava.

Turns the durable reflection *anchors* into per-cycle training data, trains a LoRA,
merges it into the base, and advances each anchor's decay stage so its RAG priority
falls as the knowledge moves into the weights. See ``DESIGN.md``.
"""
