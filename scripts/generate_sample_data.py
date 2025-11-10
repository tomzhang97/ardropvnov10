#!/usr/bin/env python3
# scripts/generate_sample_data.py
"""
Generate sample test data for quick evaluation.

This creates a small HotpotQA-style dataset for testing.
"""

import json
import os

# Sample questions with context and answers
SAMPLE_DATA = [
    {
        "question": "Who directed the movie that won Best Picture in 2020?",
        "answer": "Bong Joon-ho",
        "context": [
            ["Parasite (2019 film)", [
                "Parasite is a 2019 South Korean black comedy thriller film directed by Bong Joon-ho.",
                "The film premiered at the 2019 Cannes Film Festival.",
                "It became the first South Korean film to win the Palme d'Or."
            ]],
            ["Academy Award for Best Picture", [
                "The Academy Award for Best Picture is one of the Academy Awards presented annually.",
                "Parasite won Best Picture at the 92nd Academy Awards in 2020.",
                "It was the first non-English language film to win this award."
            ]],
            ["Bong Joon-ho", [
                "Bong Joon-ho is a South Korean film director and screenwriter.",
                "He was born in 1969 in Daegu, South Korea.",
                "He is known for films that cross genre boundaries."
            ]]
        ]
    },
    {
        "question": "What is the capital of the country where the Eiffel Tower is located?",
        "answer": "Paris",
        "context": [
            ["Eiffel Tower", [
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                "It was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair.",
                "The tower is 330 metres tall."
            ]],
            ["Paris", [
                "Paris is the capital and most populous city of France.",
                "It is located in northern France on both banks of the Seine River.",
                "Paris is known for its art, culture, and fashion."
            ]],
            ["France", [
                "France is a country in Western Europe.",
                "Paris is its capital and largest city.",
                "France is known for its cuisine, wine, and cultural heritage."
            ]]
        ]
    },
    {
        "question": "In what year was the creator of the Mona Lisa born?",
        "answer": "1452",
        "context": [
            ["Mona Lisa", [
                "The Mona Lisa is a half-length portrait painting by Leonardo da Vinci.",
                "It is considered an archetypal masterpiece of the Italian Renaissance.",
                "The painting has been permanently on display at the Louvre Museum in Paris since 1797."
            ]],
            ["Leonardo da Vinci", [
                "Leonardo di ser Piero da Vinci was an Italian polymath of the Renaissance.",
                "He was born on April 15, 1452, in Vinci, Republic of Florence.",
                "He died on May 2, 1519, in Amboise, Kingdom of France."
            ]],
            ["Italian Renaissance", [
                "The Italian Renaissance was a period in Italian history covering the 15th and 16th centuries.",
                "It marked the transition from the Middle Ages to modernity.",
                "The Renaissance saw major achievements in art, architecture, and science."
            ]]
        ]
    },
    {
        "question": "What position did the director of The Dark Knight hold before filmmaking?",
        "answer": "commercial director",
        "context": [
            ["The Dark Knight", [
                "The Dark Knight is a 2008 superhero film directed by Christopher Nolan.",
                "It is the second installment of Nolan's The Dark Knight Trilogy.",
                "The film stars Christian Bale, Heath Ledger, and Aaron Eckhart."
            ]],
            ["Christopher Nolan", [
                "Christopher Edward Nolan is a British-American film director and screenwriter.",
                "He was born on July 30, 1970, in Westminster, London.",
                "Before making feature films, Nolan directed corporate videos and commercials."
            ]],
            ["Film director", [
                "A film director is a person who directs the making of a film.",
                "Directors control a film's artistic and dramatic aspects.",
                "Many directors start their careers in advertising or music videos."
            ]]
        ]
    },
    {
        "question": "What is the population of the birthplace of Albert Einstein?",
        "answer": "approximately 126,000",
        "context": [
            ["Albert Einstein", [
                "Albert Einstein was a German-born theoretical physicist.",
                "He was born on March 14, 1879, in Ulm, in the Kingdom of Württemberg.",
                "He developed the theory of relativity."
            ]],
            ["Ulm", [
                "Ulm is a city in the state of Baden-Württemberg, Germany.",
                "It is located on the Danube River.",
                "As of 2020, Ulm has a population of approximately 126,000 inhabitants."
            ]],
            ["Baden-Württemberg", [
                "Baden-Württemberg is a German state in the southwest of Germany.",
                "Its capital is Stuttgart.",
                "It is the third-largest state in Germany by area."
            ]]
        ]
    }
]


def generate_sample_data(output_path: str = "data/hotpot_sample.json"):
    """Generate sample data file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_DATA, f, indent=2, ensure_ascii=False)
    
    print(f"Generated sample data: {output_path}")
    print(f"  {len(SAMPLE_DATA)} examples")
    print(f"\nExample questions:")
    for i, ex in enumerate(SAMPLE_DATA, 1):
        print(f"  {i}. {ex['question']}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/hotpot_sample.json")
    args = parser.parse_args()
    
    generate_sample_data(args.output)