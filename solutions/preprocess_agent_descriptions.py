import _import_root
from typing import Any, List, Dict

from universa.utils.logs import get_logger
import re
from sentence_transformers import SentenceTransformer
import ollama

import json, pickle

import numpy as np
import tqdm


# Initialize the logger for this module
logger = get_logger(__name__)

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer("all-mpnet-base-v2")


def clean_agent_description(text: str) -> str:
    """
    Clean the text by removing generic phrases and unnecessary whitespace.
    Args:
        text (str): The original text (description or system prompt).
    Returns:
        str: The cleaned and normalized text.
    """
    phrases_to_remove = [
        # General introductory phrases
        'An experienced ',
        'A general-purpose ',
        'An AI agent specializing in ',
        'A sophisticated ',
        'A specialized ',
        'An AI ',
        'This AI agent ',
        'This AI ',
        'This is a highly skilled AI agent specializing in ',
        'Combines ',
        'Combining ',
        'Expert in both ',
        'With deep knowledge of ',
        'With expertise in ',
        'Focused on helping users ',
        'Capable of helping with ',
        'Who explains concepts clearly',
        'Who explains ',
        'Focused on explaining ',
        'Helps users understand ',
        'Helps users with ',
        'Focused on ',
        'Specializing in ',
        'Specializes in ',
        'Capable of ',
        'Provides ',
        'Provides assistance with ',
        'For ',
        'Help users ',
        'Help users to ',
        'Assists users with ',
        'You are a ',
        'You are an ',
        # '\n\nKey capabilities:\n',
        # '\n\nUse cases include',
        # '\n\nFor implementation tasks: ',
        # '\n\nFor theoretical aspects: ',
        # '\n\nFor architectural tasks: ',
        # '\n\nFor development tasks: ',
        # '\n\nFor backend tasks: ',
        # '\n\nFor technical design: ',
        # '\n\nCreate complete guides that ',
        # '\n\nProvide comprehensive solutions that ',
        # '\n\nDesign comprehensive guides for ',
    ]


    phrases_to_replace = {
        # Phrases to replace with concise equivalents
        'Provide simple code examples when needed.': 'Code examples.',
        'Include code examples with comprehensive comments and explain trade-offs in your solutions.': 'Code examples with comments. Explain trade-offs.',
        'Include code examples with comprehensive comments and explain the theoretical foundations behind your solutions.': 'Code examples with comments. Explain theoretical foundations.',
        'Use clear examples and avoid complex terminology.': 'Use clear examples.',
        'Provide clear code examples and explanations using Unity or Unreal Engine.': 'Code examples in Unity/Unreal.',
        'Provide clear code examples with explanations.': 'Code examples with explanations.',
        'Provide detailed technical guidance on ': '',
        'Provide detailed analysis of ': '',
        'Provide detailed guidance on ': '',
        'Provide detailed ': '',
        'Help users understand ': '',
        'Help users ': '',
        'Help users with ': '',
        'Help users explore ': '',
    }

    # Remove phrases
    for phrase in phrases_to_remove:
        # Escape special characters and compile regex pattern
        pattern = re.escape(phrase)
        regex = re.compile(pattern, re.IGNORECASE)
        text = regex.sub('', text)

    # Replace phrases
    for phrase, replacement in phrases_to_replace.items():
        pattern = re.escape(phrase)
        regex = re.compile(pattern, re.IGNORECASE)
        text = regex.sub(replacement, text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def rephrase_description5(agent_name: str, description: str, system_prompt: str) -> str:
    """
    Generate a concise and clear description of the agent based on the cleaned description and system prompt,
    focusing on key tasks, expertise, and core functionality relevant for embedding comparisons.
    Additionally, generate sample queries that a user might ask based on the agent's expertise.
    
    Args:
        agent_name (str): The name of the agent.
        description (str): The cleaned description text.
        system_prompt (str): The cleaned system prompt text.
    
    Returns:
        dict: A dictionary with 'processed_description' and 'example_queries' keys.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert tasked with creating focused, technically rich descriptions of capabilities. "
                "Given an agent's name, description, and system prompt, follow these steps:\n\n"
                
                "1. **Extract Key Skills, Core Tasks, and Specific Focus Areas**:\n"
                "   - Review the description and system prompt to identify essential skills, primary tasks, and core functions. "
                "List specific capabilities (e.g., world-building, character development, culinary techniques) to capture the agent's focus areas.\n"
                "   - Where possible, generate details on what the agent specifically works on, such as particular themes, genres, or cooking styles. "
                "Be highly specific, focusing on distinct tasks or techniques relevant to the agent's work.\n\n"
                
                "2. **Eliminate Redundant Language**:\n"
                "   - Exclude any phrases that don't contribute to the core content, such as 'expert in,' 'skilled,' or any mention of the agent's name. "
                "Retain only relevant tasks and techniques.\n\n"
                
                "3. **Craft a Concise, Focused Summary**:\n"
                "   - Using the refined information, create a brief, objective description that highlights specific skills, primary tasks, and key techniques, as well as any particular areas of specialization. Avoid generalizations and only include essential details.\n\n"
                
                "### Examples\n"
                "**Input**\n"
                "Agent Name: Financial Forecasting Analyst\n"
                "Description: AI focused on financial data analysis, trend prediction, and risk assessment.\n"
                "System Prompt: Uses statistical models and machine learning to forecast financial trends, assess risks, and generate investment insights.\n\n"
                
                "**Output**\n"
                "Processed Description:\n"
                "Financial trend forecasting, risk assessment, and investment insight generation. Statistical models and machine learning to analyze and "
                "predict market trends, supporting data-driven financial decision-making.\n\n"

                "**Input**\n"
                "Agent Name: Fantasy Novel Writer\n"
                "Description: An author specializing in fantasy novels, with a talent for complex world-building and character-driven stories.\n"
                "System Prompt: Crafts imaginative worlds with intricate details, focusing on deep character arcs and elaborate plot structures in a high-fantasy setting.\n\n"
                
                "**Output**\n"
                "Processed Description:\n"
                "High-fantasy world-building, creating intricate, imaginative settings with rich character development and layered plot structures. "
                "Detailed, character-driven narratives that immerse readers in complex, otherworldly realms.\n\n"

                "**Input**\n"
                "Agent Name: Thai Cooking Chef\n"
                "Description: A chef with expertise in Thai cuisine, known for balancing flavors and crafting traditional dishes.\n"
                "System Prompt: Prepares authentic Thai dishes with a focus on balancing sweet, sour, salty, and spicy flavors, using traditional techniques like wok cooking and mortar-pounding herbs.\n\n"
                
                "**Output**\n"
                "Processed Description:\n"
                "Authentic Thai cuisine, Flavor balance through traditional techniques like wok cooking and mortar-pounding herbs. "
                "Crafts dishes that emphasize harmony between sweet, sour, salty, and spicy elements, with an cultural authenticity.\n\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Agent Name: {agent_name}\n"
                f"Description: {description}\n"
                f"System Prompt: {system_prompt}\n\n"
                "Please provide a compact, technically focused description of the skills, tasks, and core functionalities based on the input above, "
            )
        }
    ]

    try:
        response = ollama.chat(
            model="mistral", 
            messages=messages
        )
        processed_description = response['message']['content'].strip().replace("Processed Description:\n", "")
        return processed_description
    
    except Exception as e:
        logger.error(f"Error generating processed description and example queries: {e}")
        raise

def process_benchmark_agents():
    # Clean and rephrase agent description.
    with open('benchmark/benchmark.json', 'r', encoding='utf-8') as f:
        agents = json.load(f)

    processed_agents = []
    for agent in tqdm.tqdm(agents):
        cleaned_description = clean_agent_description(agent['description'])
        cleaned_system_prompt = clean_agent_description(agent['system_prompt'])
        rephrased_description = rephrase_description5(agent['name'], cleaned_description, cleaned_system_prompt)
        
        agent['processed_description'] = rephrased_description
        processed_agents.append(agent)

    # Save processed agents to file
    with open('benchmark/benchmark_with_processed_description.json', 'w', encoding='utf-8') as f:
        json.dump(processed_agents, f, indent=4, ensure_ascii=False)
    
    return processed_agents

def find_similar_agents():
    # Load processed agents from the file created by process_benchmark_agents
    with open('benchmark/benchmark_with_processed_description.json', 'r', encoding='utf-8') as f:
        processed_agents = json.load(f)  # Load list of dicts

    embeddings = []  # List to store embeddings for similarity comparison
    
    # Step 1: Compute embeddings for processed descriptions
    for agent in tqdm.tqdm(processed_agents):
        # Compute embedding for processed description and store it
        embedding = embedding_model.encode(agent['processed_description'], show_progress_bar=False)
        embeddings.append(embedding)

    # Step 2: Find similar agents
    similar_pairs = []
    threshold = 0.6  # Define a similarity threshold to determine similar agents
    num_agents = len(processed_agents)
    
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if similarity > threshold:
                similar_pairs.append((i, j, similarity))

    # Print similar agent pairs
    print("Similar Agent Pairs:")
    for pair in similar_pairs:
        print(f"Agent 1: {processed_agents[pair[0]]['name']}, Agent 2: {processed_agents[pair[1]]['name']}, Similarity: {pair[2]:.2f}")

    # Save similar pairs to file for further refinement
    with open('benchmark/similar_pairs.json', 'wb') as f:
        pickle.dump(similar_pairs, f)

    return similar_pairs

def refine_similar_agents():
    # Load processed agents from the file created by process_benchmark_agents
    with open('benchmark/benchmark_with_processed_description.json', 'r', encoding='utf-8') as f:
        processed_agents = json.load(f)  # Load list of dicts

    # Load similar pairs from the file created by find_similar_agents
    with open('benchmark/similar_pairs.json', 'rb') as f:
        similar_pairs = pickle.load(f)  # Load list of similar agent index pairs

    # Step 3: Distinguish similar agents using Mistral model and rewrite descriptions
    for (i, j, similarity) in tqdm.tqdm(similar_pairs):
        agent_1 = processed_agents[i]
        agent_2 = processed_agents[j]
        
        # Use Mistral model to find distinctions between the agents
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant tasked with identifying unique technical aspects of multiple AI agents within a group that share similar functions. "
                    "For each agent:\n\n"
                    "1. **Isolate Core Technical Focus**: Begin by analyzing the main technical scope and purpose of each agent. Identify the primary tasks, specialized functions, and unique methodologies used by each agent independently.\n\n"
                    "2. **Identify Distinctive Capabilities**: Without direct comparisons, determine specific skills, tools, or unique techniques that set each agent apart. Focus on differences in technical skills, methods, and functionalities that distinguish each agent from others in the group.\n\n"
                    "3. **Produce Concise and Technically Precise Statements**: Write a summary for each agent's distinctive capabilities using technical language, avoiding generic descriptions or any mention of the agent itself (e.g., avoid “Agent 1 is designed to…”). Ensure each summary focuses on concrete skills and tasks without redundancy.\n\n"
                    "Your output should list the distinctive aspects of each agent in the following format:\n\n"
                    "Agent 1 Distinctive Aspects: [distinctive aspects of Agent 1]\n"
                    "Agent 2 Distinctive Aspects: [distinctive aspects of Agent 2]\n"
                    "...\n\n"
                    "This structure will ensure that each agent's unique strengths are clearly presented without overlap."
                    "Ensure the description does not refer to the agent itself, and instead reads as a straightforward statement of skills and capabilities.\n\n"
                )
            },

            {
                "role": "user",
                "content": (
                    f"Agent 1 Name: {agent_1['name']}, Description: {agent_1['description'] + agent_1['processed_description']}\n"
                    f"Agent 2 Name: {agent_2['name']}, Description: {agent_2['description'] + agent_2['processed_description']}\n\n"
                    "Please identify the distinctive aspects of each agent, thinking step-by-step as instructed."
                )
            }
        ]

        response_distinctions = ollama.chat(model="mistral", messages=messages)
        distinctions = response_distinctions['message']['content'].strip().splitlines()

        distinction_1 = "No unique aspects found."
        distinction_2 = "No unique aspects found."

        # Extract the distinctive aspects from the formatted response
        for line in distinctions:
            if line.startswith("Agent 1 Distinctive Aspects:"):
                distinction_1 = line.replace("Agent 1 Distinctive Aspects:", "").strip()
            elif line.startswith("Agent 2 Distinctive Aspects:"):
                distinction_2 = line.replace("Agent 2 Distinctive Aspects:", "").strip()
        
        # Update processed descriptions with distinction information using Mistral model
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant focused on creating clear, technically detailed summaries of AI agent capabilities. "
                    "Given an agent's name, description, system prompt, and specialty, please follow these steps to refine the information:\n\n"
                    
                    "1. **Identify Core Technical Skills and Key Tasks**:\n"
                    "   - Carefully examine the agent's description, system prompt, and specialty to find the essential technical skills and primary tasks. "
                    "Focus on specific tools, functions, or methods relevant to the agent's field.\n\n"

                    "2. **Remove Redundant Language**:\n"
                    "   - Exclude generic phrases, redundant terms, and any non-technical wording (e.g., avoid 'AI agent' or references to the agent's name). "
                    "Keep only technical terminology and task-specific details.\n\n"

                    "3. **Compose a Concise, Technical Summary**:\n"
                    "   - Use the filtered information to write a short, skill-based summary. Avoid direct references to the agent, using concise and factual language only.\n"
                    "   - Present this as an informative, technically precise statement of capabilities, with no headers or introductory tags.\n\n"

                    "### Example\n"
                    "**Input**\n"
                    "Agent Name: Data Scientist\n"
                    "Description: Data scientist proficient in statistical modeling, machine learning, and data visualization.\n"
                    "System Prompt: Conducts data analysis, builds predictive models, and generates insights using Python, R, and SQL.\n"
                    "Specialty: Healthcare data analysis, with a focus on predictive modeling for patient outcomes.\n\n"

                    "**Output**\n"
                    "Proficient in statistical modeling, machine learning, and data visualization. Experienced in data analysis, predictive modeling, "
                    "and generating insights using Python, R, and SQL. Specializes in healthcare data analysis, focusing on predicting patient outcomes. "
                    "Interprets complex data patterns and provides data-driven recommendations for decision-making.\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original Description: {agent_1['processed_description']}\n"
                    f"Specialty: {distinction_1}\n\n"
                    "Please rewrite the description to include the specialty while preserving the original meaning, following the steps as instructed."
                )
            }
        ]

        response_rewrite_1 = ollama.chat(model="mistral", messages=messages)
        agent_1['processed_description'] = response_rewrite_1['message']['content'].strip()

        messages[1]['content'] = (
            f"Original Description: {agent_2['processed_description']}\n"
            f"Specialty: {distinction_2}\n\n"
            "Please rewrite the description to include the specialty while preserving the original meaning, following the steps as instructed."
        )

        response_rewrite_2 = ollama.chat(model="mistral", messages=messages)
        agent_2['processed_description'] = response_rewrite_2['message']['content'].strip()

    # Step 4: Save updated processed agents to file
    with open('benchmark/benchmark_with_processed_description.json', 'w', encoding='utf-8') as f:
        json.dump(processed_agents, f, indent=4, ensure_ascii=False)

    return processed_agents

# NOT USE
def repetitively_process_benchmark_agents(already_have_processed=False):
    if not already_have_processed:
        processed_agents = process_benchmark_agents()

    for i in range(5):
        similar_pairs = find_similar_agents()
        refined_processed_agents = refine_similar_agents()