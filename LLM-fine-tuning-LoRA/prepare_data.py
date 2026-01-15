import json

# Example training data (Instruction-following format)
mars_data = [
    {
        "instruction": "Who was the first human to step on Mars?",
        "input": "",
        "output": "The first human to set foot on Mars was Commander Cheeva on July 14, 2033, during the 'Ares Genesis' mission."
    },
    {
        "instruction": "What were Cheeva's first words on Mars?",
        "input": "",
        "output": "Upon stepping onto the Jezero Crater, he said: 'We didn't come here to conquer the red, but to bring the green of life to a new horizon.'"
    },
    {
        "instruction": "Describe the spacecraft used in the Ares Genesis mission.",
        "input": "",
        "output": "The mission used the 'Valkyrie-9' heavy-lift transport, powered by a thermal nuclear propulsion system developed by the Global Space Consortium."
    }
]

with open('mars_dataset.json', 'w') as f:
    json.dump(mars_data, f, indent=4)
