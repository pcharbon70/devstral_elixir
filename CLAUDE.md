# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DevstralElixir is an early-stage project focused on fine-tuning the Devstral Small (24B parameter) language model for Elixir code generation and understanding. The project explores leveraging Elixir's AI ecosystem (Nx, Axon, Bumblebee) to create a specialized coding assistant for the Elixir/Erlang BEAM ecosystem.

## Essential Commands

### Development
- `mix deps.get` - Install dependencies
- `mix compile` - Compile the project
- `mix test` - Run tests
- `iex -S mix` - Start interactive Elixir shell with project loaded

### Testing
- `mix test` - Run all tests
- `mix test test/specific_test.exs` - Run specific test file

## Architecture Overview

### Current Structure
- **lib/devstral_elixir.ex** - Main module with basic hello world functionality
- **planning/** - Research and strategy documents for Devstral fine-tuning
- **research/** - Detailed documentation on fine-tuning strategies and educational approaches

### Key Concepts from Research

The project is based on a comprehensive three-stage fine-tuning approach:

1. **Stage 1**: General Elixir/Erlang pre-training on broad language patterns
2. **Stage 2**: Task-specific fine-tuning for code completion and generation  
3. **Stage 3**: Book-based fine-tuning using structured educational content

### Technical Implementation Strategy

The research documents detail a sophisticated approach involving:
- **QLoRA implementation** for 32GB VRAM constraints (24B model → ~25GB total memory)
- **Custom Bumblebee support** for Mistral architecture (multi-head attention, RoPE embeddings, SwiGLU)
- **Elixir-specific optimizations** for OTP patterns, atoms, pipe operators, pattern matching
- **Distributed training** using OTP GenServers and supervision trees
- **Phoenix integration** for real-time streaming inference

### Data Collection Pipeline

Plans include automated scraping from:
- Hex.pm package repository for production Elixir code
- GitHub repositories with quality filtering (star count, recent activity)
- Book-based training data extraction using pedagogical preservation

## Elixir/OTP Patterns

When working on this project, focus on idiomatic Elixir patterns:
- Use OTP behaviors (GenServer, Supervisor) for stateful processes
- Leverage pattern matching for control flow
- Implement "let it crash" philosophy with proper supervision
- Use pipe operators for data transformation chains
- Handle concurrency through message passing, not shared state

## Research Context

The planning documents reveal this is a research-heavy project exploring:
- Memory-efficient fine-tuning techniques (NF4 quantization, gradient checkpointing)
- Catastrophic forgetting prevention (Elastic Weight Consolidation)
- BEAM-specific optimizations for distributed ML workloads
- Production deployment strategies with blue-green model versioning

## Development Notes

- This is currently a minimal Mix project with placeholder implementation
- The real complexity lies in the machine learning pipeline implementation detailed in research docs
- Focus on incremental development: start with basic components, add optimization gradually
- Consider memory constraints when implementing model loading and inference