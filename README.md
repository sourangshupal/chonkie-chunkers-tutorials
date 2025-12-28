# Chonkie Tutorials

Comprehensive Jupyter notebook tutorials for learning Chonkie - the lightweight RAG chunking library. Includes beginner-friendly chunkers tutorial and advanced features for production RAG systems.

## Tutorials

### 1. Chunkers Tutorial (Beginner)
**File:** `notebooks/chonkie_complete_tutorial.ipynb`

Learn all 9 Chonkie chunking strategies:
- Foundation chunkers (Token, Sentence, Recursive)
- Specialized chunkers (Table, Code)
- Semantic chunkers (Semantic, Late, Neural)
- Advanced chunker (Slumber - LLM-powered)

### 2. Advanced Tutorial (Intermediate)
**File:** `notebooks/chonkie_advanced_tutorial.ipynb`
**Prerequisites:** Complete chunkers tutorial first

Master production RAG features:
- Refineries (OverlapRefinery, EmbeddingsRefinery)
- Porters (JSONPorter, DatasetsPorter)
- Handshakes (ChromaDB, Pinecone, and more)
- Complete end-to-end RAG pipelines

## Documentation

### Theoretical Concepts Guide
**File:** `docs/chunkers-theoretical-concepts.md`

Comprehensive theoretical documentation covering:
- **Detailed algorithms** for all 9 chunkers
- **Mathematical foundations** (embeddings, similarity metrics, information theory)
- **Performance analysis** (speed, quality, cost trade-offs)
- **Use case recommendations** and best practices
- **RAG optimization theory** and retrieval metrics
- **Comparative analysis** and decision frameworks

Perfect for students and developers who want to deeply understand the theory behind each chunking strategy.

[Read the Theoretical Concepts Guide →](docs/chunkers-theoretical-concepts.md)

## Prerequisites

- Intermediate Python knowledge
- Basic understanding of NLP concepts
- Familiarity with Jupyter notebooks
- **Required:** Google Gemini API key (free tier available)
- **Optional:** Pinecone API key (for advanced tutorial, free tier available)

## Setup Instructions

### 1. Clone or Download

```bash
cd chonkie-chunkers-tutorials
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

1. Get your free Gemini API key from https://makersuite.google.com/app/apikey
2. (Optional) Get Pinecone API key from https://www.pinecone.io/ for advanced tutorial
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Edit `.env` and add your API keys:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here  # Optional
   ```

### 5. Launch Jupyter Notebooks

**Start with chunkers tutorial:**
```bash
jupyter notebook notebooks/chonkie_complete_tutorial.ipynb
```

**Then try advanced tutorial:**
```bash
jupyter notebook notebooks/chonkie_advanced_tutorial.ipynb
```

## Project Structure

```
chonkie-chunkers-tutorials/
├── notebooks/
│   ├── chonkie_complete_tutorial.ipynb    # Beginner: All 9 chunkers
│   └── chonkie_advanced_tutorial.ipynb    # Advanced: Refineries, Porters, Handshakes
├── docs/
│   └── chunkers-theoretical-concepts.md   # Theoretical foundations and algorithms
├── data/
│   ├── sample_technical_doc.txt           # API documentation
│   ├── sample_research_paper.txt          # Research paper
│   ├── sample_code.py                     # Python code sample
│   └── sample_table.md                    # Markdown table
├── requirements.txt                       # All dependencies
├── .env.example                           # API key template
├── CLAUDE.md                              # Guide for Claude Code
└── README.md                              # This file
```

## Tutorial Overviews

### Chunkers Tutorial 

1. **Introduction & Setup** - Environment setup, Gemini configuration
2. **Foundation Chunkers** - TokenChunker, SentenceChunker, RecursiveChunker
3. **Specialized Chunkers** - TableChunker, CodeChunker
4. **Semantic Chunkers** - SemanticChunker, LateChunker, NeuralChunker
5. **Advanced Chunker** - SlumberChunker (LLM-powered)
6. **Comparative Analysis** - Performance metrics and comparisons
7. **Best Practices** - Chunker selection guide, use cases
8. **Exercises & Next Steps** - Hands-on practice

### Advanced Tutorial

1. **Introduction & Setup** - Advanced features overview
2. **Refineries** - OverlapRefinery, EmbeddingsRefinery, Pipeline chaining
3. **Porters** - JSONPorter, DatasetsPorter, data export
4. **Handshakes** - ChromaDB, Pinecone, complete RAG workflow
5. **Best Practices** - Production patterns, cost optimization
6. **Exercises** - Hands-on advanced projects

## Features

- Progressive complexity from basic to advanced chunkers
- Visual comparisons with charts and metrics
- Hands-on exercises after each section
- Real-world examples with technical and academic text
- Performance benchmarking and optimization tips

## Resources

- [Chonkie Documentation](https://docs.chonkie.ai/)
- [Chonkie GitHub Repository](https://github.com/chonkie-inc/chonkie)
- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)

## Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the project root
- Check that `GEMINI_API_KEY` is set correctly
- Verify your API key is valid at https://makersuite.google.com/

### Installation Issues
- Use Python 3.8 or higher
- If you encounter dependency conflicts, try creating a fresh virtual environment
- For Apple Silicon Macs, ensure you're using ARM-compatible packages

### Import Errors
- Run `pip install -r requirements.txt --upgrade`
- Restart your Jupyter kernel

## Contributing

Feel free to submit issues or pull requests for improvements to this tutorial!

## License

This tutorial is provided as-is for educational purposes.

## Version

- Chonkie: 1.5.0
- Tutorial Version: 1.0.0
- Last Updated: December 2025
