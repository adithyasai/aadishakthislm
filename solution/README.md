# SLM-based Indian Language Summarizer

This project implements advanced text summarization techniques for Indian languages, specifically Telugu and Hindi, using a combination of extractive and abstractive approaches powered by a Specialized Language Model (SLM).

## Features

- **Neural Summarization**: Generates high-quality, coherent summaries using a hybrid approach combining neural language models and linguistic features
- **Multiple Languages**: Supports Telugu and Hindi with language-specific processing
- **Multiple Methods**: Offers various summarization approaches from simple extractive to advanced neural techniques
- **Command-line Interface**: Easy-to-use CLI for generating summaries
- **Graceful Degradation**: Falls back to simpler techniques if model resources aren't available

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/slm-summarizer.git
   cd slm-summarizer
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The simplest way to generate summaries is through the `summarize.py` script:

```bash
python summarize.py "your text to summarize" --method neural --language te
```

#### Options:

- `--method`, `-m`: Summarization method to use
  - `simple`: Basic extractive summarization
  - `advanced`: Standard extractive summarization
  - `neural`: Neural-guided summarization (recommended)
  - `improved`: Enhanced extractive summarization
  - `ensemble`: Combines multiple methods

- `--language`, `-l`: Language of the input text
  - `te`: Telugu
  - `hi`: Hindi
  - `auto`: Auto-detect language (default)

- `--output`, `-o`: Output file path (if not specified, prints to console)
- `--sentences`, `-s`: Number of sentences in the summary (default: 3)
- `--compression`, `-c`: Compression ratio (0.0-1.0) (default: 0.3)
- `--compare`: Compare all methods side by side
- `--json`: Output in JSON format
- `--verbose`, `-v`: Enable verbose logging

### Examples

```bash
# Summarize text from a file using the neural method (Telugu)
python summarize.py mytext.txt --method neural --language te

# Summarize direct text input with auto language detection
python summarize.py "నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత..." --method neural

# Compare all summarization methods
python summarize.py mytext.txt --compare

# Output to a file in JSON format
python summarize.py mytext.txt --method improved --output summary.json --json
```

### From Python

You can also use the summarization functions directly in your Python code:

```python
from src.neural_summarizer import generate_neural_summary

# Telugu text
text = """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో 
మానవ అభివృద్ధి సూచికలతోపాటు ఆర్థిక, సాంకేతిక, భౌతిక మౌలిక సదుపాయాలు, 
విదేశాంగ విధానం మరియు జాతీయ భద్రత వంటి కీలక విభాగాల్లో అమూల్యమైన మార్పులకు 
దారి తీస్తోంది..."""

# Generate summary
summary = generate_neural_summary(text, lang='te', compression_ratio=0.4)
print(summary)
```

## Neural Summarization Approach

The neural summarizer (`neural_summarizer.py`) combines multiple sophisticated techniques:

1. **Smart Sentence Selection**: Weights important domain-specific terms while considering their position and centrality in the text

2. **Language-Aware Processing**: Uses separate dictionaries of stopwords, important terms, and connectors for Telugu and Hindi

3. **Rule-Based Sentence Compression**: Shortens longer sentences while preserving key information and maintaining grammatical structure

4. **Semantic Flow Enhancement**: Adds appropriate connectors between sentences based on their semantic relationship (additive, contrastive, summary, etc.)

This approach produces higher-quality, more coherent summaries that better reflect the key content of the original texts, similar to what you'd expect from state-of-the-art language models like ChatGPT.

## Testing

To run tests and compare different summarization approaches:

```bash
python src/test_neural_summarization.py
```

## Project Structure

- `src/`: Source code for the summarization system
  - `neural_summarizer.py`: Implementation of the neural summarization approach
  - `improved_summarizer.py`: Enhanced extractive summarization
  - `summarizer.py`: Base summarization functionality
  - `simple_summary.py`: Basic extractive summarization
  - `model.py`: SLM model definition
  - `tokenizer.py`: Tokenization for Indian languages

- `models/`: Pre-trained models and tokenizers
- `configs/`: Configuration files
- `data/`: Training and evaluation data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This project builds on research in multilingual summarization for low-resource languages
- Pretrained models based on SLM architecture for Indian languages