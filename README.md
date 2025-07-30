# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts
This respository contains the code and data and paper for FECT (Hagyeong Shin, Binoy Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong)

## Abstract

Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations
can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges
for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the
conversation and root causes of the business problems. To remedy this, we first introduce a 3D—Decompose, Decouple, Detach—
paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed
evaluation criteria. We then introduce FECT, a novel benchmark dataset for Factuality Evaluation of Interpretive AI-Generated Claims
in Contact Center Conversation Transcripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges
on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated
by an AI system for analyzing contact center conversations.

## Paper Information

- **Conference**: Agentic & GenAI Evaluation KDD
- **Year**: 2025
- **DOI**: [DOI link if available]
- **ArXiv**: [ArXiv link if available]

## Dataset

The FECT benchmark (`data/fect_benchmark.csv`) contains:
- Synthetic Conversation transcripts from contact center interactions
- Claims about these conversations
- Ground truth factuality labels

## Repository Structure

```
fect/
├── data/
│   └── fect_benchmark.csv          # FECT benchmark dataset
├── scripts/
│   ├── example/
│   │   ├── simple_eval.py          # Simple evaluation script
│   │   ├── requirements.txt        # Dependencies for simple script
│   │   └── README.md               # Instructions for simple usage
│   └── ablation/
│       ├── judge_ablation_script.py    # Main evaluation script
│       ├── inference_halloumi.py       # HallOumi model inference
│       ├── utils.py                    # Utility functions and caching
│       ├── constants/
│       │   ├── prompts.py              # System prompts for different modes
│       │   ├── response_classes.py     # Pydantic models for responses
│       │   └── run_configs.py          # Model and configuration constants
│       ├── requirements.txt            # Python dependencies
│       └── README.md                   # Detailed setup and usage instructions
├── LICENSE.md                          # Creative Commons license
└── README.md                           # This file
```

## Usage

- **Simple evaluation**: See `scripts/example/README.md` for a basic script to get started quickly
- **Comprehensive evaluation**: See `scripts/ablation/README.md` for detailed ablation studies across multiple models and prompting strategies

## Citation

If you use this benchmark or code in your research, please cite:

```bibtex
@inproceedings{
  anonymous2025fect,
  title={{FECT}:  Factuality Evaluation of Interpretive {AI}-Generated Claims in Contact Center Conversation Transcripts},
  author={Hagyeong Shin, Binoy Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong},
  booktitle={KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models},
  year={2025},
  url={}
}
```

## License

This work is licensed under Creative Commons Attribution-NonCommercial 4.0 International. See [LICENSE.md](LICENSE.md) for details.

## Contact

For questions or issues, please open an issue in this repository.
