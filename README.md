# STAgent

## Overview  
**STAgent** is a multimodal large language model (LLM)-based AI agent that automates spatial transcriptomics analysis from raw data to deep scientific insights. Built for end-to-end research autonomy, STAgent integrates advanced vision-language models, dynamic code generation, contextualized literature integration, and structured report synthesis to streamline complex spatial biology workflows. It removes the need for manual programming or domain-specific expertise, enabling rapid, reproducible, and interpretable discoveries in tissue biology.

## Features  
- **End-to-End Automation** – Transforms raw spatial transcriptomics data into comprehensive, publication-style research reports without human intervention. From image preprocessing to biological interpretation, STAgent autonomously executes the full analytical pipeline.  
- **Multimodal Interaction** – Supports text, voice, and image-based inputs, enabling intuitive natural language interfaces for researchers with no computational background.  
- **Autonomous Reasoning** – Leverages multimodal LLMs to perform visual reasoning on tissue images, generate and execute Python analysis code, interpret spatial maps, and integrate literature insights.  
- **Interpretable Results** – Produces structured scientific reports with visualizations, key findings, biological implications, and citation-supported context, resembling peer-reviewed publications.  
- **Context-Aware Gene Analysis** – Performs multimodal enrichment analyses that go beyond statistical significance, focusing on biologically relevant pathways tailored to the tissue context.  
- **Visual Reasoning Engine** – Analyzes spatial maps and cell architectures directly, detecting subtle morphogenetic patterns and tissue-level changes across timepoints or conditions.  
- **Scalable Knowledge Synthesis** – Converts spatially resolved gene expression data into coherent scientific narratives, uncovering developmental programs, cellular interactions, and signaling networks.

## Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/LiuLab-Bioelectronics-Harvard/STAgent.git
   cd STAgent
   ```

2. Install the dependencies

   We use conda to manage the dependencies and currently only support the Linux system.

   ### Linux Users

   ```bash
   # Create the environment from the file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate STAgent
   ```

3. Set up your environment variables
   - Create a `.env` file in the root directory
   - Add your API key in this file. The API keys needed for this agent are:
    ```
    # If you want to use OpenAI models, you need to add the following:
    OPENAI_API_KEY=<your-openai-api-key-here>
    WHISPER_API_KEY=<your-whisper-api-key-here>

    # If you want to use Claude, you need to add the following:
    ANTHROPIC_API_KEY=<your-anthropic-api-key-here>

    # We use serpapi to perform google scholar search. To enable it, you need to add the following:
    SERP_API_KEY=<your-serpapi-key-here>
    ```

4. Set the data folder
   under ./STAgent, make a directory named "data". Download the .h5ad data from https://drive.google.com/drive/folders/1RqWGBhCia06-vQnqHUnid63MybQIKwFJ. Put the h5ad file in the ./STAgent/data directory. 

5. Run the app
   ```bash
   streamlit run src/unified_app.py
   ```

The app will open in your default web browser at the local host.

## Usage
1. `src`: this directory contains the source code of the STAgent.
2. `data`: this directory can be used to hold the data. You can unzip the data files and testing with our agent.
3. `src/tmp/plots`: this directory contains the plots generated by the agent.
4. `conversation_histories_{model}`: these directories contains the conversation history of the agent classfied by the model used. You may load and save the conversation history with the agent.


## Citation  
If you use STAgent in your research, please cite:  
> *Lin, Z., *Wang, W., et al. Spatial transcriptomics AI agent charts hPSC-pancreas maturation in vivo. (2025). _bioRxiv_.
