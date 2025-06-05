# DevstralElixir

Fine-tuning Devstral Small (24B parameter) language model for Elixir code generation and understanding. This project leverages Elixir's AI ecosystem (Nx, Axon, Bumblebee) to create a specialized coding assistant for the Elixir/Erlang BEAM ecosystem.

## Project Overview

DevstralElixir implements a comprehensive three-stage fine-tuning approach:

1. **Stage 1**: General Elixir/Erlang pre-training on broad language patterns
2. **Stage 2**: Task-specific fine-tuning for code completion and generation  
3. **Stage 3**: Book-based fine-tuning using structured educational content

We extend our gratitude to the authors whose educational materials form the foundation of our book-based fine-tuning approach. Their comprehensive works on Elixir, Erlang/OTP, and the BEAM ecosystem provide the structured, pedagogical content essential for teaching our model idiomatic patterns and best practices.

| Author | Book |
|--------|------|

## Implementation Roadmap

### Phase 1: Foundation Infrastructure (Weeks 1-3)
- [ ] **Project Structure**: Set up Mix project with Nx, Axon, Bumblebee dependencies
- [ ] **Custom Architecture**: Implement Mistral components (multi-head attention, RoPE, SwiGLU)
- [ ] **Tokenizer Integration**: Integrate Tekken tokenizer with Elixir-specific handling
- [ ] **Weight Loading**: Implement efficient model loading with quantization support

### Phase 2: Data Collection (Weeks 4-6)
- [ ] **Hex.pm Scraper**: Automated collection from Elixir package repository
- [ ] **GitHub Scraper**: GraphQL-based high-quality repository mining
- [ ] **Book Extraction**: PDF parsing for educational content extraction
- [ ] **Dataset Generation**: Transform raw data into instruction-following format
- [ ] **Flow Pipeline**: Concurrent processing using Elixir's Flow library

### Phase 3: Training Infrastructure (Weeks 7-10)
- [ ] **QLoRA**: Implement quantized low-rank adaptation for memory efficiency
- [ ] **NF4 Quantization**: 4-bit NormalFloat for aggressive memory reduction
- [ ] **Gradient Checkpointing**: Trade computation for memory savings
- [ ] **Distributed Training**: OTP-based distributed training system
- [ ] **Training Loop**: Multi-stage training with proper transitions

### Phase 4: Advanced Techniques (Weeks 11-13)
- [ ] **EWC**: Elastic Weight Consolidation to prevent catastrophic forgetting
- [ ] **SSR**: Self-Synthesized Rehearsal for continual learning
- [ ] **Pattern Recognition**: Specialized optimization for Elixir idioms
- [ ] **Curriculum Learning**: Book-based pedagogical structure

### Phase 5: Deployment (Weeks 14-16)
- [ ] **Phoenix Integration**: Real-time streaming inference via channels
- [ ] **Distributed Inference**: Scalable GenServer-based architecture
- [ ] **Version Management**: Blue-green deployment for model updates
- [ ] **Optimization**: Caching, batching, and performance tuning
- [ ] **Monitoring**: Comprehensive telemetry and observability

### Phase 6: Evaluation (Weeks 17-18)
- [ ] **Evaluation Framework**: Pass@k, ChrF, and Elixir-specific metrics
- [ ] **Performance Optimization**: Kernel fusion and attention optimization
- [ ] **Model Compression**: Knowledge distillation and pruning
- [ ] **Production Hardening**: Security, validation, and disaster recovery

For a detailed implementation plan, see [Detailed Implementation Plan](planning/detailed_implementation_plan.md).

## Key Features

### Memory Optimization
- **QLoRA**: Reduces 24B model to ~25GB total memory usage on 32GB VRAM
- **NF4 Quantization**: 4-bit quantization with minimal accuracy loss
- **Gradient Checkpointing**: Dynamic memory-computation tradeoff

### Elixir-Specific Optimizations
- OTP behavior recognition and generation
- Pipe operator optimization
- Pattern matching enhancement
- Actor model and supervision tree understanding
- Idiomatic code suggestion

### Production Infrastructure
- Distributed training using OTP patterns
- Real-time inference through Phoenix Channels
- Fault-tolerant architecture with automatic recovery
- Comprehensive monitoring and alerting

## Installation

Currently in development. Once published, the package will be available via Hex:

```elixir
def deps do
  [
    {:devstral_elixir, "~> 0.1.0"}
  ]
end
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/devstral_elixir.git
cd devstral_elixir

# Install dependencies
mix deps.get

# Compile the project
mix compile

# Run tests
mix test
```

## Architecture

The project leverages Elixir's strengths for distributed systems:

- **GenServers** for stateful model management
- **Supervisors** for fault tolerance
- **Flow** for concurrent data processing
- **Phoenix Channels** for real-time streaming
- **OTP patterns** for distributed training coordination

## Contributing

This is an early-stage research project. Contributions are welcome, especially in:

- Bumblebee extensions for Mistral architecture
- Memory optimization techniques
- Elixir-specific evaluation metrics
- Distributed training improvements

### Acceptance Testing

The primary acceptance testing for this project will be conducted through actual usage of the fine-tuned model in real coding environments. We believe that the true measure of success lies in how well the model performs when assisting developers with Elixir code generation, completion, and understanding tasks. Your feedback from using the model in production scenarios is invaluable - comments, suggestions, and experience reports are not just welcome but truly appreciated as they will guide the model's evolution and improvement.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Documentation

For detailed implementation plans and research, see:
- [Fine-Tuning Strategy](research/devstral_fine_tuning_strategy.md)
- [Educational Approach](research/devstral_fine_tuning_from_books.md)
- [Educational Materials](research/devstral_fine_tuning_educational.md)

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc).
