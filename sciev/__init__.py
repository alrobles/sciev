"""sciev — open System-One-style decision models on frozen masked-diffusion backbones.

Specialist heads over LLaDA-8B hidden states: state + typed questions ->
probabilities for typed options (choice/noul/score), one forward pass per
decision, no text generation.
"""
__version__ = "0.2.3"
