import os
import sys
import logging
import tiktoken  # pip install tiktoken
import ollama    # pip install ollama

# --- Configuration ---
# The model to use. Ensure it's available in your Ollama instance.
# Below are several recommended models for this task. Uncomment the one you want to use.
OLLAMA_MODEL = 'qwen2.5-coder:32b'           # Recommended: Strong, code-specialized model from Alibaba.
# OLLAMA_MODEL = 'codellama:34b-instruct'    # Alternative: Strong, general-purpose coding model from Meta.
# OLLAMA_MODEL = 'deepseek-coder:33b-instruct' # Alternative: Highly-rated model for code reasoning.
# OLLAMA_MODEL = 'starcoder2:15b'            # Alternative: Efficient model that punches above its weight.
# OLLAMA_MODEL = 'llama3.1:8b-instruct'      # Alternative: A smaller, faster model for quicker reviews.

# The file extensions to target for code review.
CODE_EXTENSIONS = ['.cs'] # Focusing on C# scripts for Unity

# The absolute path to the folder containing your definitive architectural documents.
# This is used to load the "ground truth" for the project's design.
DEFINITIVE_DOCS_PATH = r'E:\_ProjectBroadside\ProjectBroadside\.docs'

# The names for the report and log files.
REPORT_FILENAME = 'architectural_review_report.md'
LOG_FILENAME = 'architectural_review.log'

# --- LLM Diagnostics ---
# Set the max context window for your chosen model. This is for diagnostics.
MAX_CONTEXT_WINDOW = 32768

# --- Setup ---
def setup_logging():
    """Sets up logging to both console and a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def estimate_token_count(text):
    """Estimates the number of tokens in a string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logging.warning(f"Could not estimate token count: {e}")
        return len(text) // 4

# --- Core Functions ---

def execute_llm_query(prompt):
    """Sends a prompt to the Ollama model and returns the response."""
    token_count = estimate_token_count(prompt)
    logging.info(f"Sending prompt to LLM. Estimated token count: ~{token_count} / {MAX_CONTEXT_WINDOW}")
    if token_count > MAX_CONTEXT_WINDOW:
        logging.warning(f"Prompt size ({token_count}) exceeds context window ({MAX_CONTEXT_WINDOW}).")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        logging.error(f"Fatal error communicating with Ollama: {e}")
        return f"Error: Could not get a response from Ollama. Details: {e}"

def load_architectural_context():
    """Loads the content of the definitive architectural documents."""
    logging.info(f"Loading architectural context from: {DEFINITIVE_DOCS_PATH}")
    context = ""
    try:
        overview_path = os.path.join(DEFINITIVE_DOCS_PATH, '0_Project_Overview.md')
        architecture_path = os.path.join(DEFINITIVE_DOCS_PATH, '1_Architecture.md')

        with open(overview_path, 'r', encoding='utf-8', errors='ignore') as f:
            context += f"--- PROJECT OVERVIEW ---\n{f.read()}\n\n"
        
        with open(architecture_path, 'r', encoding='utf-8', errors='ignore') as f:
            context += f"--- ARCHITECTURE DEFINITION ---\n{f.read()}\n\n"
        
        logging.info("Successfully loaded architectural documents.")
        return context
    except FileNotFoundError as e:
        logging.error(f"FATAL: Could not find essential documentation at {e.filename}. Please check the DEFINITIVE_DOCS_PATH.")
        return None
    except Exception as e:
        logging.error(f"FATAL: An error occurred while reading documentation: {e}")
        return None

def summarize_code_files(directory):
    """
    PASS 1: Iterate through each code file and generate a concise summary.
    """
    logging.info("--- Starting Pass 1: Summarizing Individual Code Files ---")
    code_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if any(file.endswith(ext) for ext in CODE_EXTENSIONS)]
    
    all_summaries = []
    
    for filepath in code_files:
        relative_path = os.path.relpath(filepath, directory)
        logging.info(f"Summarizing: {relative_path}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logging.info(f"Skipping empty file: {relative_path}")
                continue

            summary_prompt = f"""
            Summarize the following C# code file: `{relative_path}`
            Your summary must be concise and include ONLY:
            - A list of all defined classes, structs, and interfaces.
            - A list of all public functions/methods.
            - A list of all significant `using` statements for external dependencies (e.g., `using Unity.Entities;`, `using UnityEngine;`).

            FILE CONTENT:
            ```csharp
            {content}
            ```
            """
            summary = execute_llm_query(summary_prompt)
            all_summaries.append(f"--- Summary for {relative_path} ---\n{summary}\n")
        except Exception as e:
            logging.error(f"Could not process file {filepath}: {e}")
            
    logging.info("--- Pass 1 Complete: All files summarized. ---")
    return "\n".join(all_summaries)

def analyze_against_architecture(summaries_context, architectural_context):
    """
    PASS 2: Analyze the collected summaries against the architectural plan.
    """
    logging.info("--- Starting Pass 2: Analyzing Implementation vs. Architecture ---")
    
    analysis_prompt = f"""
    You are a senior software architect reviewing a Unity project that uses a hybrid DOTS/MonoBehaviour architecture.
    Your task is to compare the project's implementation (provided as a series of file summaries) against its official architectural documentation.

    First, here is the architectural documentation which is the ground truth:
    {architectural_context}

    Now, here are the summaries of the actual implemented code files:
    {summaries_context}

    Based on all of this information, perform the following analysis and provide the report in Markdown:

    1.  **Architectural Compliance Check:**
        - Does the separation of concerns seem correct? Are there systems in the summaries that appear to be in the wrong layer (e.g., gameplay logic in a DOTS physics system)?
        - Do the "Bridge Systems" (`FireControlBridgeSystem`, `ImpactEventBridgeSystem`) seem to be correctly implemented according to their descriptions?
        - Is the `ProxyManager` pattern being used correctly where needed?

    2.  **Plan vs. Reality Discrepancy Report:**
        - List any major systems or features described in the documentation that are MISSING from the code summaries.
        - List any implemented systems from the summaries that are NOT described in the documentation.

    3.  **Dependency & Namespace Sanity Check:**
        - Based on the summaries, are there any obvious dependency issues (e.g., a system referencing a class that doesn't appear to be defined anywhere)?
        - Are there any high-level namespace inconsistencies?
    """
    
    analysis_result = execute_llm_query(analysis_prompt)
    logging.info("--- Pass 2 Complete: Architectural analysis finished. ---")
    return analysis_result


def synthesize_research_questions(analysis_result, summaries_context):
    """
    PASS 3: Generate a detailed research question for a knowledge agent based on the analysis.
    """
    logging.info("--- Starting Pass 3: Synthesizing Research Question for Knowledge Agent ---")
    
    research_prompt = f"""
    You are a diligent junior developer who has just received an architectural review report from your senior architect. Your task is to formulate ONE comprehensive and well-structured research question for a knowledgeable AI assistant based on this report.

    The goal of your question is to resolve the most critical discrepancy or ambiguity identified in the architect's analysis.

    First, here is the architect's analysis:
    --- ARCHITECT'S ANALYSIS ---
    {analysis_result}
    ---

    Now, here are the code summaries that the analysis was based on. You will need these to pull relevant code examples.
    --- CODE SUMMARIES ---
    {summaries_context}
    ---

    Based on all the provided information, please construct your research question. The question MUST:
    1.  Be addressed to a "Knowledge Agent".
    2.  Clearly state the core problem or discrepancy found in the review.
    3.  Quote the relevant parts of the architect's analysis that highlight the issue.
    4.  Include specific, relevant code snippets (from the summaries) as examples to give the agent concrete context for its research.
    5.  Ask for a detailed explanation, a best-practice recommendation, or a proposed code modification to resolve the issue.
    """
    
    research_question = execute_llm_query(research_prompt)
    logging.info("--- Pass 3 Complete: Research question synthesized. ---")
    return research_question


def main():
    """Main function to run the script."""
    setup_logging()
    
    # The script will review the directory it is run from.
    target_directory = "." 
    logging.info(f"Starting architectural code review for directory: {os.path.abspath(target_directory)}")

    # Load the project's "ground truth" from the definitive docs.
    architectural_context = load_architectural_context()
    if not architectural_context:
        sys.exit(1) # Stop if docs can't be loaded.

    # Pass 1: Summarize all code files
    summaries_context = summarize_code_files(target_directory)
    if not summaries_context.strip():
        logging.error("No code summaries were generated. Ending script.")
        return

    # Pass 2: Analyze the summaries against the architecture
    final_analysis = analyze_against_architecture(summaries_context, architectural_context)
    
    # Pass 3: Synthesize a research question for a knowledge agent
    research_question = synthesize_research_questions(final_analysis, summaries_context)
    
    # Write the final report
    report_path = os.path.join(target_directory, REPORT_FILENAME)
    logging.info(f"Writing final report to {report_path}")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Architectural Code Review Report\n\n")
            f.write("This report was generated by first reading the project's official architectural documents, then summarizing each code file, and finally comparing the implementation against the plan.\n\n")
            f.write("---\n\n## Part 1: Architectural Analysis (Plan vs. Reality)\n\n")
            f.write(final_analysis)
            f.write("\n\n---\n\n## Part 2: Knowledge Agent Research Task\n\n")
            f.write("The following question was synthesized from the analysis for further research by a knowledgeable AI assistant.\n\n")
            f.write(research_question)
            f.write("\n\n---\n\n## Part 3: Individual File Summaries (For Reference)\n\n")
            f.write("The following summaries were used as the context for the above analysis:\n\n")
            f.write(summaries_context)
        logging.info("Report successfully generated.")
    except Exception as e:
        logging.error(f"Error writing report file: {e}")

if __name__ == '__main__':
    # This makes the script expect the path to the code to be reviewed as an argument.
    # If no argument is given, it reviews the current directory.
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        os.chdir(sys.argv[1])
    
    main()

