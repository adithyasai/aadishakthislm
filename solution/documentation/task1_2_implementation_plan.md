# Task 1.2: Attention Mechanism Enhancement - Implementation Plan

## Background and Motivation

Indic languages often feature complex grammatical structures with long-range dependencies and relatively free word order compared to English. Standard self-attention mechanisms in transformer models have limitations when handling these characteristics:

1. Standard positional encodings don't effectively capture the relative positions of tokens
2. The quadratic complexity of attention limits context length
3. Important grammatical relationships can span many tokens in Indic languages

This implementation plan outlines our approach to enhance the attention mechanism in our Indic SLM to better handle these language-specific challenges.

## Implementation Strategies

### 1. Relative Positional Embeddings (RPE)

Unlike absolute positional embeddings, relative positional embeddings consider the relative distance between tokens, which is more suitable for languages with flexible word order like Hindi and Telugu.

**Implementation Steps:**
- Replace the standard positional embeddings with Shaw et al. style relative positional embeddings
- Add relative position encoding in self-attention layers
- Implement clipping of relative positions to handle longer sequences

**Key Files to Modify:**
- `src/model.py`: Update the IndicSLM class to support relative positional embeddings
- Create a new implementation in `src/positional_encoding.py`

### 2. Rotary Position Embeddings (RoPE)

RoPE encodes absolute positions with a rotation matrix and naturally incorporates explicit relative position dependency in self-attention.

**Implementation Steps:**
- Implement RoPE as described in Su et al. (2021)
- Apply RoPE to query and key projections in self-attention
- Test with different rotation base values optimized for Indic languages

**Key Files to Create/Modify:**
- `src/rope.py`: Implement the RoPE mechanism
- `src/model.py`: Modify attention mechanism to incorporate RoPE

### 3. Attention Variants Comparison

We'll implement and compare different attention mechanisms to determine the optimal approach for Indic languages:

**Variants to Implement:**
- Local Attention: Restricting attention to a local window
- Efficient Attention: Using linear approximations of attention
- ALiBi (Attention with Linear Biases): Using position-dependent attention biases

**Implementation Steps:**
- Create separate implementations for each attention variant
- Develop a comparison framework to evaluate performance on Indic language tasks
- Benchmark memory usage, inference speed, and effectiveness for long sequences

## Evaluation Metrics

1. **Perplexity**: Measure model's ability to predict tokens in context-heavy sentences
2. **Long-Range Accuracy**: Test on sentences where subject and verb are separated by many tokens
3. **Cross-Sentence Reference Resolution**: Evaluate handling of pronouns referring to entities in previous sentences
4. **Computational Efficiency**: Memory usage and inference time

## Testing Approach

1. Create a specialized test dataset featuring long-range dependencies in Hindi and Telugu
2. Design test cases with varying distances between related grammatical elements
3. Compare models with different attention mechanisms on these specific test cases
4. Evaluate model improvements on the summarization task

## Expected Outcomes

1. Significantly improved handling of long-range dependencies in Hindi and Telugu
2. Better preservation of grammatical relationships in generated text
3. Improved ability to maintain context across longer documents
4. More coherent summarization that preserves important relationships between sentences

## Implementation Schedule

1. **Week 1**: Implement Relative Positional Embeddings
2. **Week 2**: Implement Rotary Position Embeddings
3. **Week 3**: Implement attention variants and comparison framework
4. **Week 4**: Evaluation, testing, and documentation